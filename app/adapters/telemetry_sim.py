# =========================================
# app/adapters/telemetry_sim.py
# -----------------------------------------
# 作用：
#   - 提供一个“可直接用”的遥测模拟源，用于开发/联调。
#   - 暴露两个与后端其余服务/前端配套的最小接口：
#       1) list_assets() -> List[Dict[str, str]]
#            返回资产清单（id, label）
#       2) get_recent_power(asset_id: str) -> List[Dict[str, Any]]
#            返回“最近 N 点”的功率序列（每个点形如 {"ts": ISO8601, "kW": float}）
#   - 数据特性：
#       * 每个资产初始化时生成 60 个、按秒递增的点，值在一个基础负荷附近轻微抖动
#       * 支持大小写/中文 label；id 统一小写无空格，便于 URL 路径传参
#   - 备注：
#       * 这是“拉式”的简化模拟：每次 get_recent_power() 都返回当前缓存的 60 点。
#         Streaming growth can append samples or publish them through SSE.
# =========================================

from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Any
import random
import math


@dataclass
class PowerPoint:
    """单个功率点：时间戳 + 有功(kW)"""
    ts: str
    kW: float


class TelemetrySim:
    """
    遥测模拟源（线程安全需求不高的简单版本）。
    """
    def __init__(self, seed: int | None = None) -> None:
        self._rng = random.Random(seed)
        # Configurable simulator asset inventory
        # id 建议小写无空格；label 可包含中文，显示更友好
        self._assets: List[Dict[str, str]] = [
            {"id": "qc-01",   "label": "岸桥 QC-01"},
            {"id": "yc-01",   "label": "场桥 YC-01"},
            {"id": "agv-01",  "label": "AGV-01（无人集卡）"},
            {"id": "wh-01",   "label": "仓库 01"},
            # Additional simulator assets may be registered here.
            # {"id": "cs-01", "label": "充电桩 01"},
            # {"id": "ps-01", "label": "配电房 01"},
        ]

        # ===== 最近曲线缓存：每个资产 60 个点 =====
        now = datetime.now(timezone.utc)
        self._series: Dict[str, List[PowerPoint]] = {}
        for a in self._assets:
            aid = a["id"]
            base = self._base_kw(aid)
            pts: List[PowerPoint] = []
            # 生成“60 秒之前 ~ 刚才”的 60 个点
            for i in range(60, 0, -1):
                ts = (now - timedelta(seconds=i)).isoformat()
                kw = self._jitter(base, i)
                pts.append(PowerPoint(ts=ts, kW=kw))
            self._series[aid] = pts

    # ---------- 外部接口：资产清单 ----------
    def list_assets(self):
        """
        对标大平台：返回带元数据的资产清单
        字段约定：
          - id:   后端内部 ID（用于 /api/telemetry、/api/forecast、/api/twin/run）
          - label: 下拉框显示的主文案
          - category: 设备大类（岸桥 / 场桥 / BESS / 冷站 / 照明 等）
          - port: 所属场站（可选）
          - rated_kw: 额定功率（可选）
          - supports: 支持的模式 ['now', 'forecast', 'sim'] 的子集
        """
        return [
            # 岸桥（仿真重点对象）
            {
                "id": "qc-01",
                "label": "岸桥 QC-01",
                "category": "岸桥",
                "port": "Port G",
                "rated_kw": 3500,
                "supports": ["now", "forecast", "sim"],
            },
            {
                "id": "qc-02",
                "label": "岸桥 QC-02",
                "category": "岸桥",
                "port": "Port G",
                "rated_kw": 3500,
                "supports": ["now", "forecast", "sim"],
            },

            # 场桥 / 堆高机
            {
                "id": "yard-01",
                "label": "场桥 YC-01",
                "category": "场桥",
                "port": "Port G",
                "rated_kw": 800,
                "supports": ["now", "forecast"],
            },

            # 岸电 BESS
            {
                "id": "bess-01",
                "label": "岸电储能 BESS-01",
                "category": "BESS",
                "port": "Port G",
                "rated_kw": 5000,
                "supports": ["now", "forecast", "sim"],
            },

            # 冷站 HVAC
            {
                "id": "hvac-01",
                "label": "冷站 HVAC-01",
                "category": "冷站",
                "port": "Port G",
                "rated_kw": 3000,
                "supports": ["now", "forecast", "sim"],
            },

            # 还可以保留原来的 AGV，保证旧逻辑不破
            {
                "id": "agv-01",
                "label": "AGV-01",
                "category": "AGV",
                "port": "Port G",
                "rated_kw": 200,
                "supports": ["now", "forecast"],
            },
        ]

    # ---------- 外部接口：最近功率序列 ----------
    def get_recent_power(self, asset_id: str) -> List[Dict[str, Any]]:
        """
        返回“最近 60 点”的功率序列。
        * 为保证“看起来实时”，如果最后一个点已经很久，就顺带补几个点到当前时间。
        * 返回 dict 列表（兼容前端与 server 的统一结构）
        """
        aid = (asset_id or "").strip()
        if aid not in self._series:
            # 未知资产：返回空列表（前端会显示空状态，不会报错）
            return []

        # 若距离上一个点超过 1s，这里顺带补齐到“当前秒”
        self._backfill_until_now(aid)

        return [asdict(p) for p in self._series[aid]]

    # ---------- 内部：按照资产类型给一个基础负荷 ----------
    def _base_kw(self, aid: str) -> float:
        if aid.startswith("qc"):       # 岸桥
            return 45 + self._rng.uniform(-4, 6)
        if aid.startswith("yc"):       # 场桥
            return 32 + self._rng.uniform(-3, 4)
        if aid.startswith("agv"):      # AGV
            return 12 + self._rng.uniform(-2, 2)
        if aid.startswith("wh"):       # 仓库
            return 18 + self._rng.uniform(-2, 2)
        if aid.startswith("cs"):       # 充电桩
            return 10 + self._rng.uniform(-5, 10)
        if aid.startswith("ps"):       # 配电房（站内损耗）
            return 6 + self._rng.uniform(-1, 1)
        return 10 + self._rng.uniform(-2, 2)

    # ---------- 内部：给定基础负荷，制造一个“平滑+抖动”的值 ----------
    def _jitter(self, base: float, t_index: int) -> float:
        # 用一个缓慢的正弦（日内波动近似）+ 少许随机噪声
        wobble = 1.0 + 0.06 * math.sin(t_index / 12.0)
        noise = self._rng.uniform(-0.8, 0.8)
        return round(max(0.0, base * wobble + noise), 3)

    # ---------- 内部：把序列补到“当前秒” ----------
    def _backfill_until_now(self, aid: str) -> None:
        series = self._series[aid]
        if not series:
            return
        last_ts = datetime.fromisoformat(series[-1].ts.replace("Z", "+00:00"))
        if last_ts.tzinfo is None:
            last_ts = last_ts.replace(tzinfo=timezone.utc)

        now = datetime.now(timezone.utc)
        # 最多补 5 秒，避免瞬间生产过多点
        to_fill = min(5, int((now - last_ts).total_seconds()))
        if to_fill <= 0:
            return

        # 基于最后一个点的值，继续前进
        base = series[-1].kW
        for i in range(1, to_fill + 1):
            ts = (last_ts + timedelta(seconds=i)).isoformat()
            # 轻微扰动
            base = max(0.0, base + self._rng.uniform(-0.6, 0.6))
            series.append(PowerPoint(ts=ts, kW=round(base, 3)))

        # 只保留最近 60 个
        if len(series) > 60:
            self._series[aid] = series[-60:]
