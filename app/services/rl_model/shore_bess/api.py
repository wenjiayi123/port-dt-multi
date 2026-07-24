# -*- coding: utf-8 -*-
"""
Shore+BESS 模块 · HTTP API（前端对接 + 点表握手）
位置: app/services/rl_model/shore_bess/api.py

大白话：
- 这个文件开一个轻量 HTTP 服务（标准库 http.server），把 JSONL 和 policy.bin 里的数据
  用接口的形式“喂给”前端和网关，并提供点表握手 stub（nonce + TTL），方便你替换为 OPC/Modbus。
- 读 adapter 写的 baseline/offline_dataset/metrics，以及 rl_engine 训练出的 policy.bin。
- 给定 ts（或时间窗）返回“基线 + 策略残差 + 安全投影”的推荐调度（与落地口径一致：SLA → PCC/N-1 → 备用 → 反送 → 斜坡）。

注意：
- 不依赖第三方框架；只用标准库。
- 输出/日志均统一追加到 shore_bess/artifacts/shore_bess_outputs.jsonl（以 key 检索）。
"""
from __future__ import annotations
import os, sys, json, csv, math, time, argparse
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta, timezone
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
import numpy as np

# 复用 adapter 的工具/口径
from .adapter import (
    ShoreBESSAdapter, find_file, ts_to_iso_z, parse_ts_any, _tz_of_asia_shanghai
)
# 复用 rl_engine 的模型定义（轻量 MLP + 高斯策略），以便加载 policy.bin 进行推断
from .rl_engine import GaussianPolicy, MLP

HERE = os.path.dirname(os.path.abspath(__file__))
ARTIFACTS_DIR = os.path.join(HERE, "artifacts")
UNIFIED_JSONL = os.path.join(ARTIFACTS_DIR, "shore_bess_outputs.jsonl")
POLICY_BIN = os.path.join(HERE, "policy.bin")
POLICY_META = os.path.join(HERE, "policy_meta.json")
MODE_STATE = os.path.join(ARTIFACTS_DIR, "mode_state.json")
POINT_STATE = os.path.join(ARTIFACTS_DIR, "point_write_state.json")

# ------------------ JSONL 统一写入 ------------------

def ensure_dirs():
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)

def write_jsonl(key: str, payload: Dict[str, Any]):
    ensure_dirs()
    rec = {"key": key, **payload}
    with open(UNIFIED_JSONL, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")

# ------------------ JSONL 读取工具 ------------------

def jsonl_iter_keys(keys: List[str]) -> Dict[str, List[Dict[str, Any]]]:
    out = {k: [] for k in keys}
    if not os.path.exists(UNIFIED_JSONL):
        return out
    with open(UNIFIED_JSONL, "r", encoding="utf-8") as f:
        for line in f:
            try:
                j = json.loads(line)
            except Exception:
                continue
            k = j.get("key")
            if k in out:
                out[k].append(j)
    return out

def jsonl_last_by_key(key: str) -> Optional[Dict[str, Any]]:
    it = jsonl_iter_keys([key]).get(key, [])
    return it[-1] if it else None

def jsonl_all_by_key(key: str, limit: Optional[int]=None) -> List[Dict[str, Any]]:
    it = jsonl_iter_keys([key]).get(key, [])
    return it if (limit is None or len(it)<=limit) else it[-limit:]

# ------------------ 策略推断与安全投影 ------------------

class PolicyRunner:
    """
    负责：
    - 加载配置/基线/策略
    - 构造观测 → 生成残差动作 → 安全投影 → 返回推荐调度
    - 点表下发 stub（可替换）
    """

    def __init__(self, dt_min: int = 10):
        self.dt_min = dt_min
        self.adapter = ShoreBESSAdapter(dt_min=dt_min)
        self.bess_cfg, self.demand_cfg, self.berths = self.adapter.load_configs()
        self.timezone_local = _tz_of_asia_shanghai()
        self.mode = self._load_mode()
        self._load_or_build_baseline()
        self._load_policy()

    # ---- 状态持久化（模式/点表） ----
    def _load_mode(self) -> Dict[str, Any]:
        if os.path.exists(MODE_STATE):
            try:
                with open(MODE_STATE, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
        return {"mode": "shadow", "updated": ts_to_iso_z(datetime.utcnow().replace(tzinfo=timezone.utc))}

    def save_mode(self, mode: str):
        self.mode = {"mode": mode, "updated": ts_to_iso_z(datetime.utcnow().replace(tzinfo=timezone.utc))}
        with open(MODE_STATE, "w", encoding="utf-8") as f:
            json.dump(self.mode, f, ensure_ascii=False, indent=2)
        write_jsonl("mode_change", {"ts": self.mode["updated"], "mode": mode})

    # ---- 基线/策略加载 ----
    def _load_or_build_baseline(self):
        # 如果没有 baseline，则自动构建“今天 00:00Z + 24h”
        baseline = jsonl_all_by_key("baseline_dispatch", limit=None)
        if not baseline:
            today_utc = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0, tzinfo=timezone.utc)
            self.adapter.export_all(today_utc, today_utc + timedelta(hours=24))
            baseline = jsonl_all_by_key("baseline_dispatch", limit=None)
        self.baseline = baseline
        # 建映射：ts -> 行
        self.baseline_map: Dict[str, Dict[str, Any]] = {row["ts"]: row for row in baseline}
        self.ts_list = [row["ts"] for row in baseline]
        # 尝试读取离线数据集，解析 feat_names & berth_order；否则从 baseline 推断
        meta = jsonl_last_by_key("train_log")
        if os.path.exists(POLICY_META):
            try:
                with open(POLICY_META, "r", encoding="utf-8") as f:
                    m = json.load(f)
                    self.feat_names = m.get("feat_names", [])
                    self.berth_order = m.get("berth_order", [])
            except Exception:
                self.feat_names, self.berth_order = self._infer_feat_from_baseline()
        else:
            self.feat_names, self.berth_order = self._infer_feat_from_baseline()

    def _infer_feat_from_baseline(self) -> Tuple[List[str], List[str]]:
        # 观测字段（与 adapter/rl_engine 对齐）
        sample = self.baseline[0]
        feat = [
            "SOC", "P_bess_kW", "r_res_kW", "P_pcc_kW", "P_roll15_kW",
            "price_yuan_per_kWh", "ef_kg_per_kWh"
        ]
        berth_order = sorted(list(self.berths.keys()))
        for b in berth_order:
            feat.append(f"P_shore_{b}_kW")
        return feat, berth_order

    def _load_policy(self):
        self.policy = None
        self.policy_mu_std = None
        if os.path.exists(POLICY_BIN):
            try:
                with open(POLICY_BIN, "r", encoding="utf-8") as f:
                    d = json.load(f)
                actor = GaussianPolicy.load(d["actor"])
                self.policy = actor
            except Exception:
                self.policy = None

    # ---- 工具：从 baseline 行构造观测向量 ----
    def obs_from_baseline_row(self, row: Dict[str, Any]) -> Tuple[Dict[str, float], np.ndarray]:
        obs_d = {
            "SOC": row["SOC"],
            "P_bess_kW": row["P_bess_kW"],
            "r_res_kW": row["r_res_kW"],
            "P_pcc_kW": row["P_pcc_kW"],
            "P_roll15_kW": row["P_roll15_kW"],
            "price_yuan_per_kWh": row["price_yuan_per_kWh"],
            "ef_kg_per_kWh": row["ef_kg_per_kWh"],
        }
        for b in self.berth_order:
            obs_d[f"P_shore_{b}_kW"] = float(row["P_shore"].get(b, 0.0))
        x = np.array([obs_d[k] for k in self.feat_names], dtype=np.float32)[None, :]
        return obs_d, x

    # ---- 动作安全投影（和训练器口径一致） ----
    def project_action(self, base_row: Dict[str, Any], a_res: np.ndarray) -> Tuple[np.ndarray, Dict[str,int]]:
        P_max = self.bess_cfg.rated_power_kW
        ramp = self.bess_cfg.p_ramp_kW_per_step
        export_allowed = (self.bess_cfg.export_allowed and self.demand_cfg.export_allowed)
        a = a_res.copy()
        reasons = {"SLA":0, "PCC":0, "Reserve":0, "Export":0, "RAMP":0}
        P_shore_base: Dict[str, float] = base_row["P_shore"]
        P_pcc_base  = float(base_row["P_pcc_kW"])
        roll15      = float(base_row["P_roll15_kW"])

        # 1) 泊位残差：±5%cap，不允许降低到小于基线最小保供（基线已是 p_req_min）
        for i, b in enumerate(self.berth_order):
            cap_b = float(self.berths[b].cap_kw if b in self.berths else 6000.0)
            delta_lim = 0.05 * cap_b
            a[i] = float(max(min(a[i], +delta_lim), -delta_lim))
            if P_shore_base.get(b,0.0) + a[i] < P_shore_base.get(b,0.0) - 1e-6:
                a[i] = 0.0; reasons["SLA"] += 1

        # 2) BESS 功率残差：±15%Pmax + 斜坡
        delta_bess_lim = 0.15 * P_max
        a_bess = float(max(min(a[-2], +delta_bess_lim), -delta_bess_lim))
        if abs(a_bess) > ramp:
            a_bess = float(math.copysign(ramp, a_bess)); reasons["RAMP"] += 1

        # 3) 备用残差：±10%Pmax（不降低备用基线）
        delta_rres_lim = 0.10 * P_max
        a_rres = float(max(min(a[-1], +delta_rres_lim), -delta_rres_lim))
        if base_row["r_res_kW"] + a_rres < base_row["r_res_kW"]:
            a_rres = 0.0; reasons["Reserve"] += 1

        # 4) 反送禁止：新 PCC ≥ 0（不允许充电使 PCC 反向）
        pcc_new = P_pcc_base + sum(a[:len(self.berth_order)]) - a_bess
        if (not export_allowed) and (pcc_new < 0):
            if a_bess < 0:
                take = min(-a_bess, -pcc_new)
                a_bess += take; pcc_new += take
            if pcc_new < 0:
                for i in range(len(self.berth_order)):
                    if a[i] > 0: pcc_new -= a[i]; a[i] = 0.0
                reasons["Export"] += 1

        # 5) 需量窗口高压时禁止上行
        if roll15 > self.demand_cfg.pcc_limit_kw - 0.01:
            for i in range(len(self.berth_order)):
                if a[i] > 0: a[i] = 0.0
            if a_bess < 0: a_bess = 0.0
            reasons["PCC"] += 1

        a_proj = a.copy()
        a_proj[-2] = a_bess
        a_proj[-1] = a_rres
        return a_proj, reasons

    # ---- 给定 ts 生成推荐调度（基线 + 残差） ----
    def recommend_at(self, ts_iso: Optional[str], noise_scale: float=0.0) -> Dict[str, Any]:
        # 选最近时间点
        if ts_iso and ts_iso in self.baseline_map:
            base = self.baseline_map[ts_iso]
        else:
            # 选距离查询 ts 最近的网格点；若没给 ts 就用 baseline[0]
            if not ts_iso:
                base = self.baseline[0]
            else:
                # 简单最近邻
                try:
                    tgt = parse_ts_any(ts_iso, _tz_of_asia_shanghai())
                except Exception:
                    tgt = self._parse_iso(ts_iso)
                best = None; bestd = 1e18
                for row in self.baseline:
                    dt = abs(self._parse_iso(row["ts"]) - tgt)
                    if dt < bestd:
                        bestd = dt; best = row
                base = best

        obs_d, s_vec = self.obs_from_baseline_row(base)

        # 1) 取 Actor 均值作为残差；策略不存在则 0 残差
        if self.policy is None:
            a_res = np.zeros((len(self.berth_order)+2,), dtype=np.float32)
        else:
            mu, std = self.policy.forward(s_vec)
            if noise_scale > 0:
                a_res = (mu + np.random.randn(*mu.shape).astype(np.float32) * std * noise_scale)[0]
            else:
                a_res = mu[0]

        # 2) 安全投影
        a_proj, reasons = self.project_action(base, a_res)

        # 3) 叠加到基线，给出推荐设定（以及新 PCC 估计与节省的近似）
        rec = self._apply_residual(base, a_proj)
        rec["reasons"] = reasons
        rec["mode"] = self.mode.get("mode","shadow")

        # 审计记录（供前端“屏蔽统计/原因”可视化）
        write_jsonl("dispatch_recommendation", {
            "ts": base["ts"],
            "mode": rec["mode"],
            "delta_action": rec["delta_action"],
            "pcc_new_kW": rec["pcc_new_kW"],
            "save_yuan_step": rec["save_yuan_step"],
            "reasons": reasons
        })
        return rec

    def _apply_residual(self, base: Dict[str, Any], a_proj: np.ndarray) -> Dict[str, Any]:
        # 新岸电
        P_shore_new: Dict[str, float] = {}
        for i, b in enumerate(self.berth_order):
            P_shore_new[b] = float(base["P_shore"].get(b,0.0) + a_proj[i])
        # 新 BESS/备用
        P_bess_new = float(base["P_bess_kW"] + a_proj[-2])
        r_res_new  = float(base["r_res_kW"] + a_proj[-1])
        # 新 PCC（近似）：基线 PCC + ΣΔP_shore - ΔP_bess
        pcc_new = float(base["P_pcc_kW"] + sum(a_proj[:len(self.berth_order)]) - a_proj[-2])
        # 电费节省（近似）：ΔPCC × 价 × Δt
        dt_h = self.dt_min/60.0
        d_cost = (max(pcc_new,0.0) - max(base["P_pcc_kW"],0.0)) * dt_h * base["price_yuan_per_kWh"]
        # 构造返回
        rec = {
            "ts": base["ts"],
            "baseline": {
                "P_shore": base["P_shore"], "P_bess_kW": base["P_bess_kW"],
                "r_res_kW": base["r_res_kW"], "P_pcc_kW": base["P_pcc_kW"]
            },
            "delta_action": {**{f"ΔP_shore_{b}_kW": float(a_proj[i]) for i,b in enumerate(self.berth_order)},
                             "ΔP_bess_kW": float(a_proj[-2]), "Δr_res_kW": float(a_proj[-1])},
            "recommended": {
                "P_shore": P_shore_new, "P_bess_kW": P_bess_new,
                "r_res_kW": r_res_new
            },
            "pcc_new_kW": round(pcc_new, 3),
            "save_yuan_step": round(float(-d_cost), 6)
        }
        return rec

    @staticmethod
    def _parse_iso(s: str) -> datetime:
        try:
            return datetime.fromisoformat(s.replace("Z","+00:00")).astimezone(timezone.utc)
        except Exception:
            return datetime.utcnow().replace(tzinfo=timezone.utc)

    # ---- 点表下发 stub（nonce+TTL+节流+斜坡） ----
    def apply_points(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        输入 JSON：
        {
          "nonce": "abc", "expire_sec": 60, "write_enable": 1,
          "commands": {
            "shore_P_set": {"B1": 5000, ...},
            "bess_P_set": 1200,
            "bess_reserve_set": 4000,
            "soc_target_set": 0.65
          }
        }
        返回：applied_ts / rejected / reasons / state
        上线时：把这里替换为 OPC/Modbus 写点即可；点表字段与《附录 A｜点表与握手》一致。
        """
        now = datetime.utcnow().replace(tzinfo=timezone.utc)
        nonce = str(payload.get("nonce",""))
        ttl   = int(payload.get("expire_sec", 60))
        we    = int(payload.get("write_enable", 0))
        cmds  = payload.get("commands", {}) or {}

        # 1) nonce + TTL
        if ttl <= 0:
            return {"status":"rejected","reason":"invalid_ttl"}
        # 2) 写使能
        if we != 1 or self.mode.get("mode","shadow") == "shadow":
            # 影子或未使能：只记录审计，不真正下发
            write_jsonl("actuation_attempt", {"ts": ts_to_iso_z(now), "nonce": nonce, "commands": cmds, "accepted": False, "reason": "write_not_enabled_or_shadow"})
            return {"status":"shadow", "applied": False, "reason":"write_not_enabled_or_shadow", "ts": ts_to_iso_z(now)}

        # 3) 节流（每 15min 最多 20 次写入；每点 max_delta_per_write）
        state = {"last_writes": [], "rate_limit_per_15min": 20, "max_delta_per_write": {"bess_P_set": 0.15*self.bess_cfg.rated_power_kW}}
        if os.path.exists(POINT_STATE):
            try:
                with open(POINT_STATE, "r", encoding="utf-8") as f:
                    state = json.load(f)
            except Exception:
                pass
        # 清理旧记录
        last_w = []
        for rec in state.get("last_writes", []):
            try:
                t = datetime.fromisoformat(rec["ts"].replace("Z","+00:00")).astimezone(timezone.utc)
                if now - t <= timedelta(minutes=15):
                    last_w.append(rec)
            except Exception:
                continue
        state["last_writes"] = last_w
        if len(last_w) >= state.get("rate_limit_per_15min",20):
            write_jsonl("actuation_rejected", {"ts": ts_to_iso_z(now), "nonce": nonce, "reason": "rate_limit"})
            return {"status":"rejected","reason":"rate_limit","ts": ts_to_iso_z(now)}

        # 4) 斜坡与幅度（示例：BESS 功率不超过 ±p_ramp_kW_per_step）
        # 注意：真实接口应该读回 FB 后做闭环；这里仅作静态约束检查
        max_delta_bess = min(state["max_delta_per_write"].get("bess_P_set", 1e9), self.bess_cfg.p_ramp_kW_per_step)
        if "bess_P_set" in cmds:
            val = float(cmds["bess_P_set"])
            # 读取上一次应用值（若无则放行）
            last_val = None
            for rec in reversed(state.get("last_writes", [])):
                if "bess_P_set" in rec.get("commands", {}):
                    last_val = float(rec["commands"]["bess_P_set"]); break
            if last_val is not None and abs(val - last_val) > max_delta_bess + 1e-6:
                write_jsonl("actuation_rejected", {"ts": ts_to_iso_z(now), "nonce": nonce, "reason": "bess_ramp_exceed"})
                return {"status":"rejected","reason":"bess_ramp_exceed","ts": ts_to_iso_z(now)}

        # 5) 写点（stub）
        applied = {
            "applied_ts": ts_to_iso_z(now),
            "nonce": nonce,
            "commands": cmds
        }
        state["last_writes"].append({"ts": ts_to_iso_z(now), "commands": cmds})
        with open(POINT_STATE, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=2)
        write_jsonl("actuation_applied", applied)
        return {"status":"applied", **applied}

# ------------------ HTTP Handler ------------------

class Handler(BaseHTTPRequestHandler):
    runner = PolicyRunner(dt_min=10)

    # CORS + JSON 帮助函数
    def _set_cors(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def _json(self, code: int, data: Dict[str, Any]):
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self._set_cors()
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self.send_response(204)
        self._set_cors()
        self.end_headers()

    def do_GET(self):
        try:
            parsed = urlparse(self.path)
            path = parsed.path
            q = {k: v[0] for k, v in parse_qs(parsed.query).items()}

            if path == "/rl/shore_bess/health":
                self._json(200, self._health())

            elif path == "/rl/shore_bess/kpi":
                self._json(200, self._kpi())

            elif path == "/rl/shore_bess/dispatch":
                ts = q.get("ts")
                noise = float(q.get("noise", "0") or "0")
                rec = self.runner.recommend_at(ts, noise_scale=noise)
                self._json(200, rec)

            elif path == "/rl/shore_bess/dispatch/range":
                start = q.get("start")
                end = q.get("end")
                self._json(200, self._dispatch_range(start, end))

            elif path == "/rl/shore_bess/config":
                self._json(200, self._config())

            elif path == "/rl/shore_bess/audit":
                limit = int(q.get("limit","200"))
                self._json(200, self._audit(limit))

            elif path == "/rl/shore_bess/export/dispatch.csv":
                start = q.get("start")
                end = q.get("end")
                self._export_csv(start, end)

            else:
                self._json(404, {"error": "not_found", "path": path})

        except Exception as e:
            self._json(500, {"error": "server_error", "detail": str(e)})

    def do_POST(self):
        try:
            parsed = urlparse(self.path)
            path = parsed.path
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length).decode("utf-8") if length>0 else "{}"
            payload = json.loads(raw or "{}")

            if path == "/rl/shore_bess/actuate":
                res = self.runner.apply_points(payload)
                self._json(200, res)

            elif path == "/rl/shore_bess/mode":
                mode = str(payload.get("mode","shadow")).lower()
                if mode not in ("shadow","canary","full"):
                    self._json(400, {"error":"invalid_mode"})
                else:
                    self.runner.save_mode(mode)
                    self._json(200, {"ok": True, "mode": self.runner.mode})

            else:
                self._json(404, {"error": "not_found", "path": path})

        except Exception as e:
            self._json(500, {"error": "server_error", "detail": str(e)})

    # ----------- 各路由实现 -----------

    def _health(self) -> Dict[str, Any]:
        return {
            "ts": ts_to_iso_z(datetime.utcnow().replace(tzinfo=timezone.utc)),
            "mode": self.runner.mode,
            "policy_loaded": bool(self.runner.policy is not None),
            "baseline_steps": len(self.runner.baseline),
            "jsonl_path": UNIFIED_JSONL
        }

    def _kpi(self) -> Dict[str, Any]:
        metrics = jsonl_last_by_key("metrics") or {}
        evalr = jsonl_last_by_key("policy_eval") or {}
        return {"metrics": metrics.get("kpis", {}),
                "policy_eval": evalr.get("eval", {}),
                "window": metrics.get("window", {})}

    def _dispatch_range(self, start_iso: Optional[str], end_iso: Optional[str]) -> Dict[str, Any]:
        # 选择窗口（默认今天 00:00Z 到 +24h）
        if start_iso:
            start = parse_ts_any(start_iso, _tz_of_asia_shanghai())
        else:
            start = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0, tzinfo=timezone.utc)
        if end_iso:
            end = parse_ts_any(end_iso, _tz_of_asia_shanghai())
        else:
            end = start + timedelta(hours=24)

        steps = []
        for row in self.runner.baseline:
            t = parse_ts_any(row["ts"], _tz_of_asia_shanghai())
            if start <= t <= end:
                rec = self.runner.recommend_at(row["ts"])
                steps.append({
                    "ts": rec["ts"],
                    "P_pcc_new_kW": rec["pcc_new_kW"],
                    "save_yuan_step": rec["save_yuan_step"],
                    "P_bess_kW": rec["recommended"]["P_bess_kW"],
                    "P_shore_sum_kW": sum(rec["recommended"]["P_shore"].values())
                })
        return {"window": {"start": ts_to_iso_z(start), "end": ts_to_iso_z(end)}, "steps": steps}

    def _config(self) -> Dict[str, Any]:
        # 读原始配置文件，随接口返回（供前端展示/核对）
        cfg = {"bess_master": {}, "demand_window": {}}
        bpath = find_file("bess_master.json")
        if bpath and os.path.exists(bpath):
            with open(bpath, "r", encoding="utf-8") as f:
                cfg["bess_master"] = json.load(f)
        dpath = find_file("demand_window_config.json")
        if dpath and os.path.exists(dpath):
            with open(dpath, "r", encoding="utf-8") as f:
                cfg["demand_window"] = json.load(f)
        return cfg

    def _audit(self, limit: int) -> Dict[str, Any]:
        keys = ["train_log","rollout","dispatch_recommendation","actuation_attempt","actuation_rejected","actuation_applied","mode_change"]
        out = {}
        for k in keys:
            out[k] = jsonl_all_by_key(k, limit=limit)
        return out

    def _export_csv(self, start_iso: Optional[str], end_iso: Optional[str]):
        # 生成调度表 CSV：ts, P_shore_{b}, P_bess, r_res, pcc_new
        if start_iso:
            start = parse_ts_any(start_iso, _tz_of_asia_shanghai())
        else:
            start = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0, tzinfo=timezone.utc)
        if end_iso:
            end = parse_ts_any(end_iso, _tz_of_asia_shanghai())
        else:
            end = start + timedelta(hours=24)

        rows = []
        header = ["ts"] + [f"P_shore_{b}_kW" for b in self.runner.berth_order] + ["P_bess_kW","r_res_kW","P_pcc_new_kW"]
        rows.append(header)
        for row in self.runner.baseline:
            t = parse_ts_any(row["ts"], _tz_of_asia_shanghai())
            if start <= t <= end:
                rec = self.runner.recommend_at(row["ts"])
                arr = [rec["ts"]]
                for b in self.runner.berth_order:
                    arr.append(str(rec["recommended"]["P_shore"].get(b,0.0)))
                arr.append(str(rec["recommended"]["P_bess_kW"]))
                arr.append(str(rec["recommended"]["r_res_kW"]))
                arr.append(str(rec["pcc_new_kW"]))
                rows.append(arr)

        # 写内存到文本
        out_lines = []
        for r in rows:
            out_lines.append(",".join(map(str, r)))
        body = ("\n".join(out_lines)).encode("utf-8")
        self.send_response(200)
        self._set_cors()
        self.send_header("Content-Type", "text/csv; charset=utf-8")
        self.send_header("Content-Disposition", "attachment; filename=shore_bess_dispatch.csv")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

# ------------------ CLI ------------------

def parse_args():
    ap = argparse.ArgumentParser(description="Shore+BESS API Service")
    ap.add_argument("--serve", action="store_true", help="启动 HTTP 服务")
    ap.add_argument("--port", type=int, default=18088, help="服务端口")
    ap.add_argument("--self-check", action="store_true", help="自检：生成一个推荐调度并写入 JSONL")
    ap.add_argument("--ts", type=str, default=None, help="自检时刻（默认 baseline[0]）")
    return ap.parse_args()

def main():
    args = parse_args()
    ensure_dirs()
    runner = PolicyRunner(dt_min=10)

    if args.self_check:
        rec = runner.recommend_at(args.ts)
        write_jsonl("selfcheck", {"ts": rec["ts"], "module":"shore_bess_api", "ok": True, "pcc_new_kW": rec["pcc_new_kW"]})
        print("[api] self-check OK. recommendation for", rec["ts"], "pcc_new_kW=", rec["pcc_new_kW"])
        print("[api] JSONL:", UNIFIED_JSONL)
        return

    if args.serve:
        httpd = ThreadingHTTPServer(("0.0.0.0", args.port), Handler)
        print(f"[api] serving on 0.0.0.0:{args.port}")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n[api] shutdown")
            httpd.server_close()
    else:
        print("Use --serve to start HTTP server, or --self-check to validate.")

if __name__ == "__main__":
    main()
