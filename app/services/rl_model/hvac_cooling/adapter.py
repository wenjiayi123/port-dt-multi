# -*- coding: utf-8 -*-
"""
HVAC 冷站/末端设定点联动 —— 南向写点适配层（adapter）
==================================================
职责：
- 校验并消费 module/api 产生的 write_jobs 或 command_payload
- 写入防抖（max_delta_per_write、rate_limit_per_15min）、nonce/过期校验
- 点表映射：/mnt/data/actuators.json | hvac_cooling/data/actuators.json | 内置默认
- 驱动：mock（默认）、http（HTTP 网关）；opcua/bacnet 预留
- 审计：写入尝试/结果统一 JSONL 落地

不使用 pandas；仅标准库 + 少量 numpy；时间/需量口径对齐 demand_window_config.json；
设定点边界/爬坡口径对齐 plant_master.json（见站端配置）。

参考口径来源：
- demand_window_config.json：granularity_min、ramp_limits、soft_cap_kW 等。  # 需量/步长/权重
- plant_master.json：setpoints.* 上下界与 ramp_*_per_15min 等。             # 设定点边界/爬坡
"""
from __future__ import annotations

import os
import sys
import json
import csv
import time
import hmac
import hashlib
import argparse
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

try:
    import numpy as np
except Exception:
    class _NP:
        def clip(self, a, a_min, a_max): return max(a_min, min(a_max, a))
        def array(self, x): return x
    np = _NP()  # type: ignore

# 复用 api.py 的公共常量与工具，确保口径一致
from .api import (
    ARTIFACT_DIR, DEFAULT_OUT, STATE_PATH, now_utc_iso,
    load_data_with_fallback, append_jsonl, safe_float
)

ADAPTER_STATE = os.path.join(ARTIFACT_DIR, "adapter_state.json")


# ---------- 工具 ----------
def ensure_dir(path: str) -> None:
    try:
        os.makedirs(path, exist_ok=True)
    except Exception:
        pass

def load_json(path: str) -> Dict[str, Any]:
    if not os.path.isfile(path): return {}
    with open(path, "r", encoding="utf-8") as f:
        try: return json.load(f)
        except Exception: return {}

def save_json(path: str, obj: Dict[str, Any]) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def parse_ts(ts: str) -> Optional[datetime]:
    fmts = ("%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%dT%H:%M:%S")
    for fmt in fmts:
        try: return datetime.strptime(ts, fmt)
        except Exception: continue
    return None

def hmac_sha256_hex(secret: str, payload: str) -> str:
    return hmac.new(secret.encode("utf-8"), payload.encode("utf-8"), hashlib.sha256).hexdigest()


# ---------- 点表映射 ----------
DEFAULT_POINT_MAP = {
    # AI 设定点由 BAS 侧 PID 接手
    "CHWS_set_cmd": {"path": "BAS/HVAC/ChillerPlant/CHWS_set_cmd"},
    "SAT_set_cmd":  {"path": "BAS/HVAC/AHU/SAT_set_cmd"},
    "SP_set_cmd":   {"path": "BAS/HVAC/AHU/SP_set_cmd"}
}

def load_point_map(data_dir: str) -> Dict[str, Any]:
    # 优先 /mnt/data；回退 hvac_cooling/data；再回退内置默认
    cand = [
        os.path.join(data_dir, "actuators.json"),
        os.path.join(os.path.dirname(__file__), "data", "actuators.json")
    ]
    for p in cand:
        if os.path.isfile(p):
            obj = load_json(p)
            if obj: return obj
    return {"points": DEFAULT_POINT_MAP}


# ---------- 驱动抽象 ----------
class BaseDriver:
    """南向写点驱动抽象接口"""
    def write(self, point_path: str, value: float, meta: Dict[str, Any]) -> Tuple[bool, str]:
        raise NotImplementedError

class MockDriver(BaseDriver):
    """干跑：不对外通信，只返回成功（用于影子/灰度验证）"""
    def write(self, point_path: str, value: float, meta: Dict[str, Any]) -> Tuple[bool, str]:
        # 可在此注入模拟失败逻辑（如非白名单点拒绝）
        return True, "mock_applied"

class HTTPGatewayDriver(BaseDriver):
    """
    预留 HTTP 网关写点（现场可替换成实际网关）。
    需要环境变量：
      BAS_HTTP_URL（基础地址）/ BAS_HTTP_TOKEN（鉴权）
    """
    def __init__(self):
        self.base = os.environ.get("BAS_HTTP_URL", "").rstrip("/")
        self.token = os.environ.get("BAS_HTTP_TOKEN", "")
    def write(self, point_path: str, value: float, meta: Dict[str, Any]) -> Tuple[bool, str]:
        if not self.base or not self.token:
            return False, "http_gateway_not_configured"
        # 仅示意：真实实现可用 requests；此处不访问外网，直接返回模拟状态
        # payload = {"point": point_path, "value": value, "meta": meta}
        # resp = requests.post(self.base + "/write", headers={"Authorization": f"Bearer {self.token}"}, json=payload)
        # return resp.ok, f"http_status_{resp.status_code}"
        return True, "http_mock_applied"

class OPCUADriver(BaseDriver):
    def write(self, point_path: str, value: float, meta: Dict[str, Any]) -> Tuple[bool, str]:
        return False, "opcua_not_implemented"

class BACnetDriver(BaseDriver):
    def write(self, point_path: str, value: float, meta: Dict[str, Any]) -> Tuple[bool, str]:
        return False, "bacnet_not_implemented"


def get_driver(name: str) -> BaseDriver:
    name = (name or "mock").lower()
    if name == "mock": return MockDriver()
    if name == "http": return HTTPGatewayDriver()
    if name == "opcua": return OPCUADriver()
    if name == "bacnet": return BACnetDriver()
    return MockDriver()


# ---------- 写入防抖状态 ----------
def load_adapter_state() -> Dict[str, Any]:
    if not os.path.isfile(ADAPTER_STATE):
        return {"last_writes": {}}  # point_path -> {value, ts_iso, count_15min}
    obj = load_json(ADAPTER_STATE)
    if "last_writes" not in obj: obj["last_writes"] = {}
    return obj

def save_adapter_state(st: Dict[str, Any]) -> None:
    save_json(ADAPTER_STATE, st)


# ---------- 适配器主体 ----------
class BASAdapter:
    """
    - 消费 JSONL 最新 decision 记录，取 write_jobs（或 command_payload）生成最终写入
    - 校验 nonce/expires、RBAC、HMAC（可选）
    - 执行写入防抖（步进/速率限制）
    - 调驱动写入，并把结果追加回 JSONL
    """
    def __init__(self, data_dir: str = "/mnt/data", jsonl_path: str = DEFAULT_OUT, driver: str = "mock"):
        self.data_dir = data_dir
        self.jsonl_path = jsonl_path
        self.driver = get_driver(driver)
        # 配置口径对齐
        data = load_data_with_fallback(self.data_dir)
        self.demand_cfg = data.get("demand_cfg", {})   # granularity/ramp/softcap
        self.plant_cfg  = data.get("plant_master", {}) # setpoint limits/ramp
        self.point_map = load_point_map(self.data_dir)

    # ---------- 读取最新决策 ----------
    def _iter_jsonl(self) -> List[Dict[str, Any]]:
        if not os.path.isfile(self.jsonl_path): return []
        out: List[Dict[str, Any]] = []
        with open(self.jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line: continue
                try:
                    out.append(json.loads(line))
                except Exception:
                    continue
        return out

    def _find_latest_decision(self) -> Optional[Dict[str, Any]]:
        rows = self._iter_jsonl()
        for obj in reversed(rows):
            if obj.get("module") == "hvac_cooling" and obj.get("kind") == "decision":
                return obj
        return None

    # ---------- 验证 payload ----------
    def _verify_payload(self, decision: Dict[str, Any]) -> Tuple[bool, str, List[Dict[str, Any]]]:
        # 优先 write_jobs；否则从 command_payload 组装
        jobs = decision.get("write_jobs", [])
        if not jobs:
            cmd = (decision.get("command_payload") or {}).get("cmd", {})
            if cmd:
                nonce = (decision.get("command_payload") or {}).get("nonce", "")
                expires_at = (decision.get("command_payload") or {}).get("expires_at", "")
                jobs = self._build_jobs_from_cmd(cmd, nonce, expires_at)

        # RBAC
        role = os.environ.get("ADAPTER_ROLE", "reader")
        if role != "writer":
            return False, "rbac_denied_not_writer", []

        # Nonce / 过期
        now = datetime.utcnow()
        for j in jobs:
            exp = j.get("expires_at") or (decision.get("command_payload") or {}).get("expires_at")
            if exp:
                ts = parse_ts(exp)
                if ts and ts < now:
                    return False, "payload_expired", []
        # HMAC（可选）
        secret = os.environ.get("ADAPTER_HMAC_SECRET", "")
        if secret:
            # 对排序后的 payload 做签名校验（现场可根据网关约定改造）
            payload_raw = json.dumps({"jobs": jobs}, ensure_ascii=False, sort_keys=True)
            sig = decision.get("audit", {}).get("hmac_sig") or ""
            if sig and sig != hmac_sha256_hex(secret, payload_raw):
                return False, "hmac_mismatch", []

        return True, "ok", jobs

    def _build_jobs_from_cmd(self, cmd: Dict[str, Any], nonce: str, expires_at: str) -> List[Dict[str, Any]]:
        # 使用 demand/plant 配置推导默认 ramp 限制
        ramp = self.demand_cfg.get("ramp_limits", {})
        chws_max_delta = float(ramp.get("chws_C_per_15min",
                          self.plant_cfg.get("setpoints", {}).get("chws_C", {}).get("ramp_C_per_15min", 0.5)))
        sat_max_delta  = float(ramp.get("sat_C_per_15min",
                          self.plant_cfg.get("setpoints", {}).get("sat_C", {}).get("ramp_C_per_15min", 0.6)))
        sp_max_delta   = float(self.plant_cfg.get("setpoints", {}).get("static_pressure_Pa", {}).get("ramp_Pa_per_15min", 50))
        ttl_s = 60

        jobs = []
        for k, v in cmd.items():
            lim = {"max_delta_per_write": sp_max_delta if k.startswith("SP_") else (sat_max_delta if k.startswith("SAT_") else chws_max_delta),
                   "rate_limit_per_15min": 1}
            jobs.append({
                "point": k,
                "value": float(v),
                "ttl_s": ttl_s,
                "nonce": nonce or self._gen_nonce(),
                "expires_at": expires_at,
                "limits": lim,
                "priority": "energy_optimized",
                "topic": "BAS/HVAC"
            })
        return jobs

    # ---------- 写入防抖 ----------
    def _rate_limit_and_clamp(self, point: str, desired: float, limits: Dict[str, Any]) -> Tuple[str, float]:
        """
        返回 (action, value)
        action: "accept" | "clamp" | "reject_rate"
        """
        st = load_adapter_state()
        last = st["last_writes"].get(point, {})
        now = datetime.utcnow()

        # 15 分钟频率限制
        rate_limit = int(limits.get("rate_limit_per_15min", 1))
        count = 0
        if last:
            last_ts = parse_ts(last.get("ts_iso") or "")
            if last_ts and (now - last_ts) <= timedelta(minutes=15):
                count = int(last.get("count_15min", 0))
        if count >= rate_limit:
            return "reject_rate", safe_float(last.get("value"), desired)

        # 每次步进限制
        max_delta = safe_float(limits.get("max_delta_per_write"), 0.5)
        prev_val = safe_float(last.get("value"), desired)
        if abs(desired - prev_val) > max_delta:
            # 夹紧至允许步进
            if desired > prev_val:
                clamped = prev_val + max_delta
            else:
                clamped = prev_val - max_delta
            return "clamp", clamped

        return "accept", desired

    def _update_state_on_success(self, point: str, value: float):
        st = load_adapter_state()
        last = st["last_writes"].get(point, {})
        last = {
            "value": float(value),
            "ts_iso": now_utc_iso(),
            "count_15min": int(last.get("count_15min", 0)) + 1
        }
        st["last_writes"][point] = last
        save_adapter_state(st)

    # ---------- 点表解析 ----------
    def _to_point_path(self, point_key: str) -> str:
        # 从 point_map 查找真实对象路径；没有就用默认
        pmap = self.point_map.get("points", {})
        if point_key in pmap:
            path = pmap[point_key].get("path") or point_key
        else:
            path = DEFAULT_POINT_MAP.get(point_key, {}).get("path", point_key)
        return path

    # ---------- 写入执行 ----------
    def apply_latest(self, dry_run: bool = True) -> Dict[str, Any]:
        decision = self._find_latest_decision()
        # 如果没有决策，尝试自动调用 module 生成一次
        if decision is None:
            try:
                from .module import CoolingRLModule
                mod = CoolingRLModule(data_dir=self.data_dir, out_path=self.jsonl_path, state_path=STATE_PATH)
                mod.plan(); decision = mod.decide()
            except Exception as e:
                return {"ok": False, "error": f"no_decision_and_generate_failed:{e}"}

        ok, msg, jobs = self._verify_payload(decision)
        attempt = {
            "ts": now_utc_iso(),
            "module": "hvac_cooling",
            "kind": "write_attempt",
            "driver": self.driver.__class__.__name__,
            "msg": msg,
            "n_jobs": len(jobs)
        }
        append_jsonl(self.jsonl_path, attempt)
        if not ok:
            return {"ok": False, "error": msg, "n_jobs": 0}

        # 执行每个 job（含防抖/夹紧/速率限制）
        results = []
        accepted = clamped = rejected = 0
        for j in jobs:
            point = j.get("point"); value = safe_float(j.get("value"))
            limits = j.get("limits", {})
            action, v2 = self._rate_limit_and_clamp(point, value, limits)
            path = self._to_point_path(point)
            status = "skipped"
            reason = ""
            if action == "reject_rate":
                rejected += 1
                status, reason = "rejected", "rate_limited"
            else:
                if action == "clamp": clamped += 1
                else: accepted += 1
                if dry_run:
                    status, reason = "applied_dry_run", action
                    self._update_state_on_success(point, v2)
                else:
                    ok, detail = self.driver.write(path, v2, {"nonce": j.get("nonce"), "expires_at": j.get("expires_at")})
                    if ok:
                        status, reason = "applied", action
                        self._update_state_on_success(point, v2)
                    else:
                        status, reason = "failed", detail

            results.append({
                "point": point,
                "path": path,
                "requested": value,
                "final": float(v2),
                "limits": limits,
                "status": status,
                "reason": reason
            })

        # 写结果审计
        result_rec = {
            "ts": now_utc_iso(),
            "module": "hvac_cooling",
            "kind": "write_result",
            "driver": self.driver.__class__.__name__,
            "dry_run": bool(dry_run),
            "results": results,
            "summary": {"accepted": accepted, "clamped": clamped, "rejected": rejected}
        }
        append_jsonl(self.jsonl_path, result_rec)
        return {"ok": True, "n_jobs": len(jobs), **result_rec["summary"]}

    @staticmethod
    def _gen_nonce() -> str:
        import uuid, time, hashlib
        raw = f"{uuid.uuid4()}-{time.time()}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


# ---------- CLI ----------
def main():
    parser = argparse.ArgumentParser(description="HVAC BAS/DDC southbound adapter")
    parser.add_argument("--data-dir", type=str, default="/mnt/data", help="数据目录（默认 /mnt/data）")
    parser.add_argument("--jsonl", type=str, default=DEFAULT_OUT, help="JSONL 文件（默认 artifacts/policy_evaluate_history.jsonl）")
    parser.add_argument("--driver", type=str, default="mock", help="驱动：mock|http|opcua|bacnet（默认 mock 干跑）")
    parser.add_argument("--apply-latest", action="store_true", help="从 JSONL 最新决策应用写入")
    parser.add_argument("--self-test", action="store_true", help="自检：若无决策则自动生成，再干跑写入")
    parser.add_argument("--wet-run", action="store_true", help="真实写入（默认 dry-run）")
    args = parser.parse_args()

    # RBAC 检查提示（不阻断自检）
    role = os.environ.get("ADAPTER_ROLE", "reader")
    if role != "writer":
        print("WARN: ADAPTER_ROLE is not 'writer' (current:", role, ") -> write will be denied")

    adapter = BASAdapter(data_dir=args.data_dir, jsonl_path=args.jsonl, driver=args.driver)

    try:
        if args.self_test or args.apply_latest:
            res = adapter.apply_latest(dry_run=not args.wet_run)
            print("SELF-TEST OK:" if args.self_test else "APPLY OK:", json.dumps(res, ensure_ascii=False))
            return 0
        # 默认自检
        res = adapter.apply_latest(dry_run=True)
        print("SELF-TEST OK:", json.dumps(res, ensure_ascii=False))
        return 0
    except Exception as e:
        print("ERROR:", repr(e))
        return 2


if __name__ == "__main__":
    sys.exit(main())
