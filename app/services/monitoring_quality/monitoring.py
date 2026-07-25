# ============================================
# app/services/monitoring.py
# --------------------------------------------
# 【大白话】监测与运维 · 异常检测 + 漂移检测（PSI）服务
#
# Data acquisition, TSDB, OPC, and MQTT adapters implement di.telemetry.get_series(...).
# 这个服务就能直接跑：先清洗->插补->做异常/漂移->写审计->总线广播。
#
# 能力：
#  1) scan_anomalies(...)：IQR / Z-Score / EWMA +（可选）残差异常(实际-预测)
#  2) scan_drift_psi(...)：Population Stability Index 检测分布漂移
#  3) 自动审计到 data/objects/audit/guard-*.json，并通过 bus 发布 monitor/* 事件
#
# 接口落地口径（与真实港口接入一致）：
#  - asset_ids / asset_id：资产ID（如岸桥 qc-01、AGV agv-07、冷机 ch-02）
#  - point：测点名（例如 active_power_kw、energy_kwh、soc_percent）
#  - 时间统一：UTC 秒级 epoch；可在 API 层把 ISO8601 转换后再传给本服务
#  - step_sec：等间隔步长（缺测自动插补），便于大屏/预测/报表共用
# ============================================

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timezone, timedelta
import math
import json
import time

# Uses the local data-cleaning module without adding dependencies.
from app.ops.data_quality import clean_and_impute
# 统一对象存储与事件总线（repo 里已有）
from app.infra.storage import ObjectStorage, StorageConfig
from app.infra.message_bus import MessageBus

EpochSec = float
Series = List[Tuple[EpochSec, float]]  # 等间隔序列：[(ts_epoch, value), ...]

# --------- 小工具：时间与格式 ---------
def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def _iso(ts: float) -> str:
    return datetime.fromtimestamp(float(ts), tz=timezone.utc).isoformat()

def _ensure_epoch(x: float | int | str) -> float:
    if isinstance(x, (int, float)):
        return float(x)
    # ISO8601 字符串
    return datetime.fromisoformat(x).timestamp()

# --------- 事件/审计载体 ---------
@dataclass
class AnomalyPoint:
    ts: str
    v: float
    score: float
    reason: str  # "iqr" | "zscore" | "ewma" | "residual"

@dataclass
class DriftBin:
    lo: float
    hi: float
    p_ref: float
    p_cur: float
    psi: float

# ============================================
# MonitoringService
# ============================================
class MonitoringService:
    """
    监测与运维核心服务：
    - 构造：传入 telemetry（必需）、forecast（可选，用于残差异常）、storage/bus（可选）
    - scan_anomalies：统计异常检测（清洗->异常->审计/广播）
    - scan_drift_psi：分布漂移检测（PSI）
    """

    def __init__(
        self,
        telemetry,
        forecast=None,
        storage: Optional[ObjectStorage] = None,
        bus: Optional[MessageBus] = None,
    ):
        self.telemetry = telemetry
        self.forecast = forecast
        self.storage = storage or ObjectStorage(StorageConfig(backend_url="file://./data/objects"))
        self.bus = bus or MessageBus()

    # ---------- 统一数据入口 ----------
    def _load_series(
        self,
        asset_id: str,
        point: str,
        start_ts: EpochSec,
        end_ts: EpochSec,
        step_sec: int,
        asset_type: str = "generic",
    ) -> Tuple[Series, Dict[str, float], str]:
        """
        从 di.telemetry 拉原始点，调用 clean_and_impute 得到等间隔序列 + 质量评分。
        返回：(cleaned_series, quality_dict, source_string)
        """
        # 真实落地：只需实现 di.telemetry.get_series(...) 即可。
        raw = []
        src = ""
        if hasattr(self.telemetry, "get_series"):
            try:
                arr = self.telemetry.get_series(asset_id=asset_id, point=point,
                                                start_ts=start_ts, end_ts=end_ts, step_sec=step_sec) or []
                # 兼容不同字段命名，转为 [(ts_epoch, v)]
                for p in arr:
                    ts = p.get("ts")
                    v = p.get("v", p.get("kW", p.get("value")))
                    if ts is None or v is None:
                        continue
                    if isinstance(ts, str):
                        ts = datetime.fromisoformat(ts).timestamp()
                    raw.append((float(ts), float(v)))
                src = "di.telemetry.get_series"
            except Exception:
                raw = []
        # 回退：若只提供了“最近功率流”
        if not raw and hasattr(self.telemetry, "get_recent_power") and point == "active_power_kw":
            try:
                arr = self.telemetry.get_recent_power(asset_id) or []
                for p in arr:
                    ts = p.get("ts")
                    v = p.get("kW", p.get("v"))
                    if isinstance(ts, str):
                        ts = datetime.fromisoformat(ts).timestamp()
                    if ts is None or v is None:
                        continue
                    raw.append((float(ts), float(v)))
                src = "di.telemetry.get_recent_power"
            except Exception:
                raw = []
        if not raw:
            # 最终兜底：返回空，让上层决定是否报错
            return [], {"completeness": 0.0, "timeliness": 0.0, "validity": 0.0}, "none"

        cleaned, quality, _mask = clean_and_impute(
            raw,
            start=float(start_ts),
            end=float(end_ts),
            step_sec=int(step_sec),
            asset_type=asset_type,
            point=point,
            resample_method="ffill",
            impute_method="ffill",
        )
        return cleaned, quality, src

    # ---------- 统计异常检测 ----------
    def _iqr(self, series: Series, k: float = 1.5) -> List[AnomalyPoint]:
        if not series:
            return []
        vals = [v for _, v in series]
        s = sorted(vals)
        q1 = s[int(0.25 * (len(s) - 1))]
        q3 = s[int(0.75 * (len(s) - 1))]
        iqr = max(1e-9, q3 - q1)
        lo, hi = q1 - k * iqr, q3 + k * iqr
        out: List[AnomalyPoint] = []
        for ts, v in series:
            if v < lo or v > hi:
                score = abs((lo - v) / iqr) if v < lo else abs((v - hi) / iqr)
                out.append(AnomalyPoint(ts=_iso(ts), v=float(v), score=float(score), reason="iqr"))
        return out

    def _zscore(self, series: Series, z: float = 3.0) -> List[AnomalyPoint]:
        if not series:
            return []
        vals = [v for _, v in series]
        mu = sum(vals) / len(vals)
        var = sum((v - mu) ** 2 for v in vals) / max(1, len(vals) - 1)
        sigma = max(1e-9, math.sqrt(var))
        out: List[AnomalyPoint] = []
        for ts, v in series:
            s = abs((v - mu) / sigma)
            if s > z:
                out.append(AnomalyPoint(ts=_iso(ts), v=float(v), score=float(s), reason="zscore"))
        return out

    def _ewma(self, series: Series, k: float = 3.0, alpha: float = 0.2) -> List[AnomalyPoint]:
        if not series:
            return []
        mu = None
        var = None
        out: List[AnomalyPoint] = []
        for ts, v in series:
            v = float(v)
            if mu is None:
                mu, var = v, 0.0
                continue
            mu = alpha * v + (1 - alpha) * mu
            var = alpha * (v - mu) ** 2 + (1 - alpha) * var
            sigma = max(1e-9, math.sqrt(var))
            s = abs((v - mu) / sigma)
            if s > k:
                out.append(AnomalyPoint(ts=_iso(ts), v=v, score=float(s), reason="ewma"))
        return out

    def _residual(self, actual: Series, forecast: Series, base_method: str = "zscore", sens: float = 3.0) -> List[AnomalyPoint]:
        """
        残差异常：对齐 actual/predict 的时间网格，计算残差再跑统计异常（默认 zscore）。
        """
        if not actual or not forecast:
            return []
        pmap = {ts: v for ts, v in forecast}
        res_series: Series = []
        for ts, v in actual:
            pv = pmap.get(ts, None)
            if pv is None:
                continue
            res_series.append((ts, float(v) - float(pv)))
        if base_method == "iqr":
            return self._iqr(res_series, k=sens)
        elif base_method == "ewma":
            return self._ewma(res_series, k=sens)
        else:
            return self._zscore(res_series, z=sens)

    # ---------- 审计与广播 ----------
    def _audit_and_publish(self, topic: str, payload: dict) -> str:
        """
        把事件写到 data/objects/audit/guard-*.json，并通过 bus 发布 monitor/*。
        返回：保存对象的 URI。
        """
        payload = dict(payload or {})
        payload.setdefault("ts", _now_iso())
        uri = self.storage.save_json(f"audit/guard-{int(time.time())}.json", payload, ensure_ascii=False, indent=2)
        try:
            self.bus.publish(topic, payload)
        except Exception:
            pass
        return uri

    # ---------- Public：异常检测 ----------
    def scan_anomalies(
        self,
        *,
        asset_ids: Optional[List[str]] = None,
        point: str = "active_power_kw",
        asset_type: str = "generic",
        # 两种时间口径：传 window_min（推荐）或传 start_ts/end_ts（高级用）
        window_min: Optional[int] = 60,
        start_ts: Optional[EpochSec] = None,
        end_ts: Optional[EpochSec] = None,
        step_sec: int = 60,
        method: str = "iqr",         # "iqr" | "zscore" | "ewma"
        sensitivity: float = 1.5,    # iqr=k; z=z; ewma=k
        residual: bool = False,      # True 时用 (actual - forecast) 检测
    ) -> Dict[str, Any]:
        """
        返回：
        {
          "generated_at": ISO,
          "params": {...},
          "items": [
             {"asset_id": "...", "quality": {...}, "anomalies": [ {ts,v,score,reason}, ... ] },
             ...
          ],
          "audit_uri": "file://..."
        }
        """
        # 计算窗口
        if start_ts is None or end_ts is None:
            assert window_min is not None and window_min > 0
            end_ts = datetime.now(timezone.utc).timestamp()
            start_ts = end_ts - window_min * 60

        # 资产清单：未指定则从 telemetry 列表自动取前 10 个
        assets = asset_ids or []
        if not assets:
            try:
                lst = self.telemetry.list_assets() or []
                assets = [a.get("asset_id") or a.get("id") for a in lst if isinstance(a, dict)]
                assets = [x for x in assets if x][:10]
            except Exception:
                assets = ["qc-01"]

        items = []
        total_anoms = 0

        for aid in assets:
            cleaned, quality, src = self._load_series(
                asset_id=aid, point=point, start_ts=float(start_ts), end_ts=float(end_ts),
                step_sec=step_sec, asset_type=asset_type
            )
            anoms: List[AnomalyPoint] = []
            if not cleaned:
                items.append({"asset_id": aid, "quality": quality, "source": src, "anomalies": []})
                continue

            if residual and self.forecast is not None:
                # 生成同窗预测序列（对齐 step）
                try:
                    # forecast.forecast_load([...], horizon_min, step_min) → {asset_id: [{"ts":ISO,"kW":x},...]}
                    horizon_min = int((float(end_ts) - float(start_ts)) / 60)
                    step_min = max(1, step_sec // 60)
                    fc = (self.forecast.forecast_load([aid], horizon_min=horizon_min, step_min=step_min) or {}).get(aid, [])
                    pred: Series = []
                    for p in fc:
                        ts = p.get("ts")
                        v = p.get("kW", p.get("v"))
                        if ts is None or v is None:
                            continue
                        if isinstance(ts, str):
                            ts = datetime.fromisoformat(ts).timestamp()
                        pred.append((float(ts), float(v)))
                    base_method = "zscore" if method not in ("iqr", "ewma") else method
                    anoms = self._residual(cleaned, pred, base_method=base_method, sens=sensitivity)
                    # 标注 reason=residual
                    for ap in anoms:
                        ap.reason = "residual"
                except Exception:
                    # 预测失败时退回原方法
                    pass

            if not anoms:
                if method == "iqr":
                    anoms = self._iqr(cleaned, k=sensitivity)
                elif method == "ewma":
                    anoms = self._ewma(cleaned, k=sensitivity)
                else:
                    anoms = self._zscore(cleaned, z=sensitivity)

            total_anoms += len(anoms)
            items.append({
                "asset_id": aid,
                "quality": quality,
                "source": src,
                "anomalies": [ap.__dict__ for ap in anoms],
            })

        # 有异常才落审计/广播（避免噪声）
        audit_uri = None
        if total_anoms > 0:
            audit_uri = self._audit_and_publish("monitor/anomaly", {
                "summary": {"total": int(total_anoms), "assets": assets, "point": point},
                "items": items,
            })

        return {
            "generated_at": _now_iso(),
            "params": {
                "assets": assets, "point": point, "asset_type": asset_type,
                "start": _iso(float(start_ts)), "end": _iso(float(end_ts)),
                "step_sec": step_sec, "method": method, "sensitivity": sensitivity,
                "residual": residual,
            },
            "items": items,
            "audit_uri": audit_uri,
        }

    # ---------- Public：分布漂移（PSI） ----------
    def scan_drift_psi(
        self,
        *,
        asset_id: str,
        point: str = "active_power_kw",
        asset_type: str = "generic",
        baseline_min: int = 24 * 60,
        recent_min: int = 60,
        step_sec: int = 60,
        bins: int = 20,
    ) -> Dict[str, Any]:
        """
        返回：
        {
          "asset_id": "...", "point": "...",
          "baseline": {"start":ISO,"end":ISO,"n":...},
          "recent": {"start":ISO,"end":ISO,"n":...},
          "psi": 0.23, "bins": [{"lo":..,"hi":..,"p_ref":..,"p_cur":..,"psi":..}, ...],
          "audit_uri": "file://..."
        }
        """
        end = datetime.now(timezone.utc).timestamp()
        b0 = end - float(baseline_min) * 60.0
        r0 = end - float(recent_min) * 60.0

        base_series, _, _ = self._load_series(asset_id, point, b0, end, step_sec, asset_type)
        cur_series,  _, _ = self._load_series(asset_id, point, r0, end, step_sec, asset_type)
        base_vals = [v for _, v in base_series]
        cur_vals  = [v for _, v in cur_series]

        if len(base_vals) < 10 or len(cur_vals) < 10:
            return {
                "asset_id": asset_id, "point": point,
                "psi": 0.0, "bins": [], "baseline": {"start": _iso(b0), "end": _iso(end), "n": len(base_vals)},
                "recent": {"start": _iso(r0), "end": _iso(end), "n": len(cur_vals)},
                "note": "样本过少，建议更长窗口或更短步长"
            }

        lo = min(min(base_vals), min(cur_vals))
        hi = max(max(base_vals), max(cur_vals))
        if hi <= lo:  # 全常数序列
            hi = lo + 1e-6
        width = (hi - lo) / max(3, int(bins))
        edges = [lo + i * width for i in range(max(3, int(bins)) + 1)]

        # 统计频率
        def _hist(vals: List[float]) -> List[float]:
            h = [0] * (len(edges) - 1)
            for v in vals:
                # 右闭合最后一箱
                for i in range(len(edges) - 1):
                    if (v >= edges[i]) and (v <= edges[i+1] if i == len(edges) - 2 else v < edges[i+1]):
                        h[i] += 1
                        break
            s = sum(h) or 1
            return [c / s for c in h]

        pe = _hist(base_vals)
        pa = _hist(cur_vals)
        eps = 1e-9
        bins_out: List[DriftBin] = []
        psi_total = 0.0
        for i in range(len(pe)):
            p, q = max(eps, pe[i]), max(eps, pa[i])
            iv = (q - p) * math.log(q / p)
            psi_total += iv
            bins_out.append(DriftBin(lo=edges[i], hi=edges[i+1], p_ref=p, p_cur=q, psi=iv))

        level = "ok" if psi_total < 0.1 else ("warn" if psi_total < 0.25 else "drift")
        audit_uri = None
        if level != "ok":
            audit_uri = self._audit_and_publish("monitor/drift", {
                "asset_id": asset_id, "point": point, "psi": psi_total, "level": level,
                "baseline": {"start": _iso(b0), "end": _iso(end)},
                "recent":   {"start": _iso(r0), "end": _iso(end)},
                "bins": [b.__dict__ for b in bins_out],
            })

        return {
            "asset_id": asset_id, "point": point,
            "baseline": {"start": _iso(b0), "end": _iso(end), "n": len(base_vals)},
            "recent": {"start": _iso(r0), "end": _iso(end), "n": len(cur_vals)},
            "psi": psi_total,
            "level": level,
            "bins": [b.__dict__ for b in bins_out],
            "audit_uri": audit_uri,
            "generated_at": _now_iso(),
        }

# =============== 冒烟自测（python -m 方式不依赖 server） ===============
def _smoke() -> dict:
    """
    直接 python -c 'from app.services.monitoring_quality.monitoring import _smoke; import json; print(json.dumps(_smoke(), ensure_ascii=False, indent=2))'
    可验证本文件是否可用（使用 di.Container 的模拟数据）。
    """
    try:
        from app.di import Container
        di = Container()
        svc = MonitoringService(telemetry=di.telemetry, forecast=getattr(di, "fcst", None),
                                storage=getattr(di, "storage", None), bus=getattr(di, "bus", None))
        # 最近 60 分钟异常扫描
        res1 = svc.scan_anomalies(asset_ids=["qc-01"], point="active_power_kw",
                                  window_min=60, step_sec=60, method="iqr", sensitivity=1.5)
        # PSI 漂移（最近 60 分钟 vs 24 小时基线）
        res2 = svc.scan_drift_psi(asset_id="qc-01", point="active_power_kw",
                                  baseline_min=24*60, recent_min=60, step_sec=60, bins=10)
        return {"anomaly_items": len(res1.get("items", [])), "psi": res2.get("psi", 0.0)}
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    print(json.dumps(_smoke(), ensure_ascii=False, indent=2))
