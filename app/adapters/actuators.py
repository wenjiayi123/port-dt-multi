# -*- coding: utf-8 -*-
"""
app/adapters/actuators.py

【文件用途】
- 提供“南向控制网关（PortSouthboundGateway）”，统一对接 OPC UA / Modbus-TCP /
  MQTT / HTTP(EMS/SCADA/TOS/PCS) 四类控制通道。
- 提供指令白名单、幂等（Idempotency-Key）、双通道确认（Two-man rule / two-channel confirm）、
  电子签名校验（e-sign placeholder）、证据包落盘（黑匣子）、一键回滚能力（若底层支持）。
- 若真实三方库、现场配置或鉴权缺失，网关必须拒绝执行，不得伪装成下发成功。
- 与现有项目的“审计目录 data/objects/audit/”兼容（沿用 guard-*.json / evt-*.json 风格）。

【谁会调用本文件】
- 未来将由 `app/services/dispatch.py`（作业/能管/充电等指令下发服务）直接调用
  PortSouthboundGateway.dispatch()/confirm()/rollback()。
- 也会被 `app/services/closed_loop.py`（闭环控制）在自动/半自动模式里调用。
- UI 或 API 层（我们随后会加到 `app/server.py` 的路由）会通过服务层间接触达本网关。

【本文件依赖/被依赖关系】
- 依赖：Python 标准库；可选依赖（若安装）：opcua、pymodbus、paho-mqtt、requests。
- 写入：`data/objects/audit/` 目录（证据包），与现有审计文件并存。
- 读取：`PORT_DT_ACTUATOR_CONFIG` 指向的现场配置；未配置时默认禁用。
- 不直接依赖你现有的 infra.message_bus/storage/tsdb，避免破坏现状；后续我们再无缝接上。

【如何落地到真实港口】
- 在 `data/objects/config/actuators.json` 填入现场 OPC UA/Modbus/MQTT/HTTP 的地址、资产映射、白名单。
- 指令格式保持不变；审批/签名/双通道确认/审计留痕全部按此文件定义执行。
"""

from __future__ import annotations
import os
import json
import math
import time
import uuid
import hashlib
import hmac
import threading
from dataclasses import dataclass, asdict, field
from typing import Any, Dict, Optional, List, Tuple

# -----------------------------
# 可选外部库（缺失时失效安全拒绝执行）
# -----------------------------
try:
    from opcua import Client as OPCUAClient  # type: ignore
except Exception:
    OPCUAClient = None

try:
    # pymodbus v3
    from pymodbus.client import ModbusTcpClient  # type: ignore
except Exception:
    ModbusTcpClient = None

try:
    import paho.mqtt.client as mqtt  # type: ignore
except Exception:
    mqtt = None

try:
    import requests  # type: ignore
except Exception:
    requests = None


# -----------------------------
# 常量与路径
# -----------------------------
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
AUDIT_DIR = os.path.join(ROOT_DIR, "data", "objects", "audit")
CONFIG_DIR = os.path.join(ROOT_DIR, "data", "objects", "config")
CONFIG_FILE = os.path.join(CONFIG_DIR, "actuators.json")

os.makedirs(AUDIT_DIR, exist_ok=True)
os.makedirs(CONFIG_DIR, exist_ok=True)


# -----------------------------
# 数据类定义：命令/结果/证据包
# -----------------------------
@dataclass
class ESign:
    """电子签名占位（真实落地时对接企业CA/HSM/国密等）"""
    user: str
    method: str  # e.g., "password", "UKey", "CA"
    signature: str  # 摘要签名值（此处不做强校验，落地时替换）
    signed_at: float

@dataclass
class Command:
    """南向控制通用命令结构（落地保持不变）"""
    asset_id: str                   # 设备/资产唯一ID（须在白名单映射中可识别）
    asset_type: str                 # crane|yard_crane|agv|lighting|cold_station|bess|shore_power|plc|...
    action: str                     # "set", "start", "stop", "open", "close", "charge", "discharge", ...
    parameters: Dict[str, Any]      # 具体参数，如 {"node":"ns=2;i=10853","value":1} 或 {"power_kw":500}
    priority: int = 5               # 1最高 9最低
    requested_by: str = "system"
    requested_at: float = field(default_factory=lambda: time.time())
    idempotency_key: Optional[str] = None
    two_channel_required: bool = False
    e_sign: Optional[ESign] = None  # 电子签名（审批场景）
    model_version: Optional[str] = None
    constraints_check: Dict[str, Any] = field(default_factory=dict)

    def ensure_idempotency_key(self) -> str:
        if not self.idempotency_key:
            raw = f"{self.asset_id}|{self.action}|{json.dumps(self.parameters, sort_keys=True)}|{int(self.requested_at)}"
            self.idempotency_key = hashlib.sha256(raw.encode("utf-8")).hexdigest()
        return self.idempotency_key


@dataclass
class CommandResult:
    status: str                     # PENDING|CONFIRMED|EXECUTED|FAILED|ROLLEDBACK
    command_id: str
    asset_id: str
    channel: str                    # opcua|modbus|mqtt|http|dry_run|unavailable
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    evidence_path: Optional[str] = None


@dataclass
class EvidencePackage:
    """黑匣子证据包（落地合规模块直接引用）"""
    command_id: str
    idempotency_key: str
    command: Dict[str, Any]
    approvals: List[Dict[str, Any]]
    route: Dict[str, Any]
    timestamps: Dict[str, float]
    results: List[Dict[str, Any]]
    model_version: Optional[str] = None
    constraints_check: Dict[str, Any] = field(default_factory=dict)

    def save(self) -> str:
        path = os.path.join(AUDIT_DIR, f"guard-{time.time_ns()}-{uuid.uuid4().hex[:8]}.json")
        _write_json_atomic(path, asdict(self))
        return path


def _write_json_atomic(path: str, payload: Dict[str, Any]) -> None:
    temporary = path + ".tmp-" + uuid.uuid4().hex
    with open(temporary, "w", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    os.chmod(temporary, 0o600)
    os.replace(temporary, path)


# -----------------------------
# 工具：配置/白名单/幂等存储
# -----------------------------
class Config:
    """加载南向控制配置；无现场配置时 fail closed。"""
    def __init__(self, path: Optional[str] = None):
        configured_path = path or os.getenv("PORT_DT_ACTUATOR_CONFIG")
        self.path = configured_path or CONFIG_FILE
        if configured_path and os.path.isfile(configured_path):
            with open(configured_path, "r", encoding="utf-8") as f:
                self.data = json.load(f)
        else:
            self.data = {
                "enabled": False,
                "whitelist": {},
                "routing": {"asset": {}, "type": {}},
                "constraints": {},
                "security": {
                    "confirmation_token_env": "PORT_DT_SECOND_CHANNEL_TOKEN",
                    "require_two_channel": True,
                    "require_constraints": True,
                },
                "reason": "PORT_DT_ACTUATOR_CONFIG is not configured",
            }

    def resolve_route(self, asset_id: str, asset_type: str) -> Dict[str, Any]:
        # 优先 asset 精确匹配，其次按类型
        asset_routing = self.data.get("routing", {}).get("asset", {})
        type_routing = self.data.get("routing", {}).get("type", {})
        if asset_id in asset_routing:
            return asset_routing[asset_id]
        return type_routing.get(asset_type, {"channel": "unavailable", "reason": "route_not_configured"})

    def is_allowed(self, asset_id: str, action: str) -> bool:
        whitelist = self.data.get("whitelist", {})
        allowed = whitelist.get(asset_id, [])
        return action in allowed


class IdempotencyStore:
    """最简单的幂等存储：基于本地文件；后续可替换为 Redis/DB。"""
    def __init__(self, dir_path: str = AUDIT_DIR):
        self.dir = dir_path

    def exists(self, idem_key: str) -> Optional[str]:
        found = self.find(idem_key)
        return found[0] if found else None

    def find(self, idem_key: str) -> Optional[Tuple[str, Dict[str, Any]]]:
        # 只检索指令证据包。evt-*.json 也含幂等键，但它不是可恢复的指令状态。
        for fn in os.listdir(self.dir):
            if not fn.startswith("guard-") or not fn.endswith(".json"):
                continue
            fp = os.path.join(self.dir, fn)
            try:
                with open(fp, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if data.get("idempotency_key") == idem_key:
                        return fp, data
            except Exception:
                continue
        return None


# -----------------------------
# 通道适配器（四类）
# -----------------------------
class BaseActuator:
    channel_name = "base"
    def execute(self, cmd: Command) -> Tuple[bool, Dict[str, Any]]:
        raise NotImplementedError
    def rollback(self, cmd: Command) -> Tuple[bool, Dict[str, Any]]:
        # 默认不支持回滚
        return False, {"reason": "rollback_not_supported"}


class OPCUAActuator(BaseActuator):
    channel_name = "opcua"
    def __init__(self, endpoint: str):
        self.endpoint = endpoint
        self.client = None

    def _connect(self):
        if OPCUAClient is None:
            return False
        if self.client is None:
            self.client = OPCUAClient(self.endpoint, timeout=2)
            self.client.connect()
        return True

    def execute(self, cmd: Command) -> Tuple[bool, Dict[str, Any]]:
        # 约定 parameters: {"node": "ns=2;i=10853", "value": 1}
        node_id = cmd.parameters.get("node")
        value = cmd.parameters.get("value")
        if OPCUAClient is None:
            return False, {"error": "opcua_dependency_unavailable", "simulated": False}
        try:
            ok = self._connect()
            if not ok:
                return False, {"error": "opcua_connect_failed"}
            node = self.client.get_node(node_id)
            node.set_value(value)
            return True, {"node": node_id, "value": value}
        except Exception as e:
            return False, {"error": f"opcua_execute_err:{e.__class__.__name__}:{e}"}

    def rollback(self, cmd: Command) -> Tuple[bool, Dict[str, Any]]:
        # 简单回滚：尝试恢复原值（需要参数里传 original_value）
        if "original_value" in cmd.parameters:
            revert = Command(
                asset_id=cmd.asset_id, asset_type=cmd.asset_type,
                action="set", parameters={"node": cmd.parameters.get("node"),
                                          "value": cmd.parameters["original_value"]},
                requested_by="rollback", idempotency_key=str(uuid.uuid4()))
            return self.execute(revert)
        return False, {"reason": "no_original_value"}


class ModbusActuator(BaseActuator):
    channel_name = "modbus"
    def __init__(self, host: str, port: int = 502):
        self.host = host
        self.port = port
        self.client = None

    def _connect(self):
        if ModbusTcpClient is None:
            return False
        if self.client is None:
            self.client = ModbusTcpClient(self.host, port=self.port)
            self.client.connect()
        return True

    def execute(self, cmd: Command) -> Tuple[bool, Dict[str, Any]]:
        # 约定 parameters: {"register": 40001, "value": 1, "unit_id": 1}
        reg = int(cmd.parameters.get("register"))
        val = int(cmd.parameters.get("value"))
        unit = int(cmd.parameters.get("unit_id", 1))
        if ModbusTcpClient is None:
            return False, {"error": "modbus_dependency_unavailable", "simulated": False}
        try:
            ok = self._connect()
            if not ok:
                return False, {"error": "modbus_connect_failed"}
            # 这里示例写单保持寄存器（真实项目根据寄存器类型调用不同API）
            rr = self.client.write_register(reg, val, unit=unit)
            if rr.isError():
                return False, {"error": str(rr)}
            return True, {"register": reg, "value": val, "unit": unit}
        except Exception as e:
            return False, {"error": f"modbus_execute_err:{e.__class__.__name__}:{e}"}


class MQTTActuator(BaseActuator):
    channel_name = "mqtt"
    def __init__(self, host: str, port: int = 1883, username: str = "", password: str = ""):
        self.host = host
        self.port = port
        self.username = username
        self.password = password
        self.client = None

    def _connect(self):
        if mqtt is None:
            return False
        if self.client is None:
            self.client = mqtt.Client()
            if self.username:
                self.client.username_pw_set(self.username, self.password)
            self.client.connect(self.host, self.port, keepalive=30)
        return True

    def execute(self, cmd: Command) -> Tuple[bool, Dict[str, Any]]:
        # 约定 parameters: {"topic": "port/agv/21/cmd", "payload": {"action":"charge","kw":120}}
        topic = cmd.parameters.get("topic") or f"port/{cmd.asset_id}/cmd"
        payload = cmd.parameters.get("payload", {})
        if mqtt is None:
            return False, {"error": "mqtt_dependency_unavailable", "simulated": False}
        try:
            ok = self._connect()
            if not ok:
                return False, {"error": "mqtt_connect_failed"}
            import json as _json
            rc = self.client.publish(topic, _json.dumps(payload), qos=1)
            return True, {"topic": topic, "mid": getattr(rc, 'mid', None), "payload": payload}
        except Exception as e:
            return False, {"error": f"mqtt_execute_err:{e.__class__.__name__}:{e}"}


class HTTPActuator(BaseActuator):
    channel_name = "http"
    def __init__(self, base_url: str, token: Optional[str] = None):
        self.base_url = base_url.rstrip("/")
        self.token = token

    def execute(self, cmd: Command) -> Tuple[bool, Dict[str, Any]]:
        # 约定：POST {base}/commands 下发指令，JSON = {asset_id, action, parameters, idem_key}
        if requests is None:
            return False, {"error": "http_dependency_unavailable", "simulated": False}
        try:
            headers = {"Content-Type": "application/json"}
            if self.token:
                headers["Authorization"] = f"Bearer {self.token}"
            payload = {
                "asset_id": cmd.asset_id,
                "action": cmd.action,
                "parameters": cmd.parameters,
                "idempotency_key": cmd.idempotency_key
            }
            resp = requests.post(f"{self.base_url}/commands", json=payload, headers=headers, timeout=2)
            ok = 200 <= resp.status_code < 300
            data = {}
            try:
                data = resp.json()
            except Exception:
                data = {"text": resp.text}
            return ok, {"status_code": resp.status_code, "resp": data}
        except Exception as e:
            return False, {"error": f"http_execute_err:{e.__class__.__name__}:{e}"}

    def rollback(self, cmd: Command) -> Tuple[bool, Dict[str, Any]]:
        # 约定：POST {base}/commands/rollback，JSON = {asset_id, idempotency_key}
        if requests is None:
            return False, {"error": "http_dependency_unavailable", "simulated": False}
        try:
            headers = {"Content-Type": "application/json"}
            if self.token:
                headers["Authorization"] = f"Bearer {self.token}"
            payload = {"asset_id": cmd.asset_id, "idempotency_key": cmd.idempotency_key}
            resp = requests.post(f"{self.base_url}/commands/rollback", json=payload, headers=headers, timeout=2)
            ok = 200 <= resp.status_code < 300
            data = {}
            try:
                data = resp.json()
            except Exception:
                data = {"text": resp.text}
            return ok, {"status_code": resp.status_code, "resp": data}
        except Exception as e:
            return False, {"error": f"http_rollback_err:{e.__class__.__name__}:{e}"}


# -----------------------------
# 网关：统一校验/路由/审计/双通道
# -----------------------------
class PortSouthboundGateway:
    """
    统一南向网关：
    - 校验白名单与签名
    - 幂等去重
    - 路由到指定通道
    - 生成证据包（输入/路由/审批/时间戳/结果/约束）
    - 支持“影子模式/小流量/全量”的逐步集成（后续我们和 rollout 服务对接）
    """
    def __init__(self, config_path: Optional[str] = None):
        self.cfg = Config(config_path)
        self.idem = IdempotencyStore()
        self._lock = threading.RLock()

    # ----------- 内部工具 -----------
    def _build_actuator(self, route: Dict[str, Any]) -> BaseActuator:
        ch = (route.get("channel") or "unavailable").lower()
        if ch == "opcua":
            return OPCUAActuator(route.get("endpoint", ""))
        if ch == "modbus":
            return ModbusActuator(route.get("endpoint", "127.0.0.1"), int(route.get("port", 502)))
        if ch == "mqtt":
            mqtt_cfg = self.cfg.data.get("mqtt", {})
            password = os.getenv(str(mqtt_cfg.get("password_env") or ""), "")
            return MQTTActuator(route.get("endpoint", ""), int(route.get("port", 1883)),
                                mqtt_cfg.get("username", ""), password)
        if ch == "http":
            http_cfg = self.cfg.data.get("http", {})
            token_env = str(http_cfg.get("auth", {}).get("token_env") or "")
            token = os.getenv(token_env, "") if token_env else None
            return HTTPActuator(route.get("endpoint", ""), token)
        if ch == "dry_run" and os.getenv("PORT_DT_ENABLE_ACTUATOR_DRY_RUN") == "1":
            return ExplicitDryRunActuator()
        return UnavailableActuator(route.get("reason") or f"unsupported_channel:{ch}")

    def _save_evt(self, payload: Dict[str, Any]) -> str:
        path = os.path.join(AUDIT_DIR, f"evt-{time.time_ns()}-{uuid.uuid4().hex[:8]}.json")
        _write_json_atomic(path, payload)
        return path

    def _check_signature(self, esign: Optional[ESign]) -> bool:
        # 占位实现：真实落地应对接企业 CA/HSM 或双因子校验
        if esign is None:
            return True
        # 简单校验：签名长度>16
        return bool(esign.signature and len(esign.signature) >= 16)

    def _check_site_constraints(self, cmd: Command) -> Tuple[bool, Dict[str, Any]]:
        constraints = self.cfg.data.get("constraints") or {}
        asset_rules = ((constraints.get("asset") or {}).get(cmd.asset_id) or {}).get(cmd.action)
        type_rules = ((constraints.get("type") or {}).get(cmd.asset_type) or {}).get(cmd.action)
        rules = asset_rules or type_rules
        require_rules = bool((self.cfg.data.get("security") or {}).get("require_constraints", True))
        if not isinstance(rules, dict) or not rules:
            return (not require_rules), {"ok": not require_rules, "reason": "site_constraints_not_configured", "violations": []}
        violations: List[Dict[str, Any]] = []
        for parameter, rule in rules.items():
            if not isinstance(rule, dict):
                violations.append({"parameter": parameter, "reason": "constraint_rule_must_be_object"})
                continue
            if rule.get("required") is True and parameter not in cmd.parameters:
                violations.append({"parameter": parameter, "reason": "required_parameter_missing"})
                continue
            if parameter not in cmd.parameters:
                continue
            value = cmd.parameters[parameter]
            allowed = rule.get("allowed")
            if isinstance(allowed, list) and value not in allowed:
                violations.append({"parameter": parameter, "reason": "value_not_allowed", "allowed": allowed})
                continue
            if rule.get("min") is not None or rule.get("max") is not None:
                try:
                    numeric = float(value)
                except (TypeError, ValueError):
                    violations.append({"parameter": parameter, "reason": "numeric_value_required"})
                    continue
                if not math.isfinite(numeric):
                    violations.append({"parameter": parameter, "reason": "finite_value_required"})
                elif rule.get("min") is not None and numeric < float(rule["min"]):
                    violations.append({"parameter": parameter, "reason": "below_minimum", "minimum": rule["min"], "value": numeric})
                elif rule.get("max") is not None and numeric > float(rule["max"]):
                    violations.append({"parameter": parameter, "reason": "above_maximum", "maximum": rule["max"], "value": numeric})
        return not violations, {"ok": not violations, "violations": violations, "rules": rules}

    def _validate_second_channel_token(self, token: str) -> Tuple[bool, str, Dict[str, Any]]:
        token_env = str((self.cfg.data.get("security") or {}).get("confirmation_token_env") or "PORT_DT_SECOND_CHANNEL_TOKEN")
        expected_token = os.getenv(token_env, "")
        if len(expected_token) < 32:
            return False, "second_channel_token_not_configured", {"token_env": token_env}
        if not isinstance(token, str) or not hmac.compare_digest(token, expected_token):
            return False, "second_channel_token_invalid", {}
        return True, "ok", {}

    # ----------- 对外主流程 -----------
    def dispatch(self, cmd: Command) -> CommandResult:
        with self._lock:
            return self._dispatch(cmd)

    def _dispatch(self, cmd: Command) -> CommandResult:
        """
        下发指令（第一阶段）：
        - 校验白名单、电子签名（若提供）
        - 幂等去重；若已执行过则返回历史证据路径
        - 若需要双通道确认：返回 PENDING，并写证据包；待 confirm() 二次确认后真正执行
        - 若不需要双通道：直接执行，记录证据包，返回 EXECUTED/FAILED
        """
        if self.cfg.data.get("enabled") is not True:
            return CommandResult(
                status="FAILED",
                command_id=str(uuid.uuid4()),
                asset_id=cmd.asset_id,
                channel="guard",
                message="actuator_gateway_disabled",
                details={"reason": self.cfg.data.get("reason") or "enabled must be true"},
            )
        if bool((self.cfg.data.get("security") or {}).get("require_two_channel", True)) and not cmd.two_channel_required:
            return CommandResult(
                status="FAILED", command_id=str(uuid.uuid4()), asset_id=cmd.asset_id, channel="guard",
                message="two_channel_confirmation_required", details={}
            )
        cmd.ensure_idempotency_key()
        # 白名单校验
        if not self.cfg.is_allowed(cmd.asset_id, cmd.action):
            return CommandResult(
                status="FAILED",
                command_id=str(uuid.uuid4()),
                asset_id=cmd.asset_id,
                channel="guard",
                message=f"action_not_allowed:{cmd.asset_id}:{cmd.action}",
                details={"whitelist": self.cfg.data.get("whitelist", {})}
            )

        # 签名校验
        if not self._check_signature(cmd.e_sign):
            return CommandResult(
                status="FAILED",
                command_id=str(uuid.uuid4()),
                asset_id=cmd.asset_id,
                channel="guard",
                message="e_signature_invalid",
                details={}
            )

        constraints_ok, constraints_detail = self._check_site_constraints(cmd)
        if not constraints_ok:
            return CommandResult(
                status="FAILED", command_id=str(uuid.uuid4()), asset_id=cmd.asset_id, channel="guard",
                message="site_constraints_failed", details=constraints_detail
            )

        # 幂等去重
        existed = self.idem.find(cmd.idempotency_key)
        if existed:
            existed_path, previous = existed
            timestamps = previous.get("timestamps") or {}
            results = previous.get("results") or []
            previous_status = "ROLLEDBACK" if timestamps.get("rolledback_at") else (
                "EXECUTED" if timestamps.get("executed_at") and any(item.get("ok") for item in results) else (
                    "FAILED" if timestamps.get("executed_at") else "PENDING"
                )
            )
            return CommandResult(
                status=previous_status,
                command_id=str(previous.get("command_id") or ""),
                asset_id=cmd.asset_id,
                channel="guard",
                message="idem_hit_return_previous",
                details={"evidence_path": existed_path},
                evidence_path=existed_path
            )

        # 路由与证据包初始化
        route = self.cfg.resolve_route(cmd.asset_id, cmd.asset_type)
        actuator = self._build_actuator(route)
        now = time.time()
        evidence = EvidencePackage(
            command_id=str(uuid.uuid4()),
            idempotency_key=cmd.idempotency_key,
            command=asdict(cmd),
            approvals=[{"type": "request", "by": cmd.requested_by, "at": now}],
            route=route,
            timestamps={"requested_at": cmd.requested_at, "created_at": now},
            results=[],
            model_version=cmd.model_version,
            constraints_check={
                "site_constraints": self.cfg.data.get("constraints", {}),
                "decision_evidence": cmd.constraints_check,
            }
        )

        if cmd.two_channel_required:
            # 第一阶段仅记录待确认
            evidence.timestamps["pending_at"] = time.time()
            path = evidence.save()
            payload = {
                "event": "command_pending",
                "command_id": evidence.command_id,
                "asset_id": cmd.asset_id,
                "idempotency_key": cmd.idempotency_key,
                "evidence_path": path
            }
            self._save_evt(payload)
            return CommandResult(
                status="PENDING",
                command_id=evidence.command_id,
                asset_id=cmd.asset_id,
                channel=actuator.channel_name,
                message="waiting_second_channel_confirm",
                details={"evidence_path": path},
                evidence_path=path
            )

        # 直接执行
        ok, detail = actuator.execute(cmd)
        evidence.results.append({"at": time.time(), "ok": ok, "detail": detail})
        evidence.timestamps["executed_at"] = time.time()
        path = evidence.save()
        evt = {
            "event": "command_executed" if ok else "command_failed",
            "command_id": evidence.command_id,
            "asset_id": cmd.asset_id,
            "idempotency_key": cmd.idempotency_key,
            "evidence_path": path
        }
        self._save_evt(evt)
        return CommandResult(
            status="EXECUTED" if ok else "FAILED",
            command_id=evidence.command_id,
            asset_id=cmd.asset_id,
            channel=actuator.channel_name,
            message="ok" if ok else "failed",
            details=detail,
            evidence_path=path
        )

    def confirm(self, command_id: str, confirmer: str, channel_token: str) -> CommandResult:
        with self._lock:
            return self._confirm(command_id, confirmer, channel_token)

    def _confirm(self, command_id: str, confirmer: str, channel_token: str) -> CommandResult:
        """
        二次确认：用于需要双通道的指令。
        - 找到对应证据包（pending）
        - 记录确认人，执行并更新证据
        """
        token_ok, token_message, token_details = self._validate_second_channel_token(channel_token)
        if not token_ok:
            return CommandResult(
                status="FAILED", command_id=command_id, asset_id="", channel="guard",
                message=token_message, details=token_details
            )

        # 简单做法：扫描最新 guard-*.json 找到 command_id
        target_path = None
        latest_mtime = 0
        for fn in os.listdir(AUDIT_DIR):
            if not fn.startswith("guard-") or not fn.endswith(".json"):
                continue
            fp = os.path.join(AUDIT_DIR, fn)
            try:
                with open(fp, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if data.get("command_id") == command_id and "pending_at" in data.get("timestamps", {}) and "executed_at" not in data.get("timestamps", {}):
                    mtime = os.path.getmtime(fp)
                    if mtime > latest_mtime:
                        latest_mtime = mtime
                        target_path = fp
            except Exception:
                continue

        if not target_path:
            return CommandResult(
                status="FAILED", command_id=command_id, asset_id="", channel="guard",
                message="pending_evidence_not_found", details={}
            )

        with open(target_path, "r", encoding="utf-8") as f:
            evidence = json.load(f)

        cmd_dict = evidence.get("command", {})
        if not confirmer.strip() or hmac.compare_digest(str(confirmer).strip(), str(cmd_dict.get("requested_by") or "").strip()):
            return CommandResult(
                status="FAILED", command_id=command_id, asset_id=str(cmd_dict.get("asset_id") or ""), channel="guard",
                message="confirmer_must_differ_from_requester", details={}
            )
        route = evidence.get("route", {})
        actuator = self._build_actuator(route)
        # 记录二次确认
        evidence.setdefault("approvals", []).append({
            "type": "confirm", "by": confirmer, "method": "separate_channel_secret", "at": time.time()
        })

        # 执行
        cmd = Command(
            asset_id=cmd_dict["asset_id"], asset_type=cmd_dict["asset_type"],
            action=cmd_dict["action"], parameters=cmd_dict["parameters"],
            priority=cmd_dict.get("priority", 5), requested_by=cmd_dict.get("requested_by", "system"),
            requested_at=cmd_dict.get("requested_at", time.time()),
            idempotency_key=cmd_dict.get("idempotency_key"), two_channel_required=False
        )
        ok, detail = actuator.execute(cmd)
        evidence.setdefault("results", []).append({"at": time.time(), "ok": ok, "detail": detail})
        evidence.setdefault("timestamps", {})["executed_at"] = time.time()

        # 覆盖保存
        _write_json_atomic(target_path, evidence)

        evt = {
            "event": "command_executed" if ok else "command_failed",
            "command_id": command_id,
            "asset_id": cmd.asset_id,
            "idempotency_key": cmd.idempotency_key,
            "evidence_path": target_path
        }
        self._save_evt(evt)

        return CommandResult(
            status="EXECUTED" if ok else "FAILED",
            command_id=command_id,
            asset_id=cmd.asset_id,
            channel=actuator.channel_name,
            message="ok" if ok else "failed",
            details=detail,
            evidence_path=target_path
        )

    def rollback(self, command_id: str, reason: str = "manual", approver: str = "", channel_token: str = "") -> CommandResult:
        with self._lock:
            return self._rollback(command_id, reason, approver, channel_token)

    def _rollback(self, command_id: str, reason: str = "manual", approver: str = "", channel_token: str = "") -> CommandResult:
        """
        回滚：找到对应证据包，尽力调用底层回滚（若支持），记录新结果。
        """
        token_ok, token_message, token_details = self._validate_second_channel_token(channel_token)
        if not token_ok:
            return CommandResult(status="FAILED", command_id=command_id, asset_id="", channel="guard", message=token_message, details=token_details)
        if not approver.strip():
            return CommandResult(status="FAILED", command_id=command_id, asset_id="", channel="guard", message="rollback_approver_required", details={})
        target_path = None
        latest_mtime = 0
        for fn in os.listdir(AUDIT_DIR):
            if not fn.startswith("guard-") or not fn.endswith(".json"):
                continue
            fp = os.path.join(AUDIT_DIR, fn)
            try:
                with open(fp, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if data.get("command_id") == command_id:
                    mtime = os.path.getmtime(fp)
                    if mtime > latest_mtime:
                        latest_mtime = mtime
                        target_path = fp
                        evidence = data
            except Exception:
                continue

        if not target_path:
            return CommandResult(
                status="FAILED", command_id=command_id, asset_id="", channel="guard",
                message="evidence_not_found", details={}
            )

        cmd_dict = evidence.get("command", {})
        if not any(item.get("ok") and not item.get("rollback") for item in evidence.get("results", [])):
            return CommandResult(status="FAILED", command_id=command_id, asset_id=str(cmd_dict.get("asset_id") or ""), channel="guard", message="no_successful_execution_to_rollback", details={})
        if evidence.get("timestamps", {}).get("rolledback_at"):
            return CommandResult(status="FAILED", command_id=command_id, asset_id=str(cmd_dict.get("asset_id") or ""), channel="guard", message="command_already_rolled_back", details={})
        route = evidence.get("route", {})
        actuator = self._build_actuator(route)
        cmd = Command(
            asset_id=cmd_dict["asset_id"], asset_type=cmd_dict["asset_type"],
            action=cmd_dict["action"], parameters=cmd_dict["parameters"],
            requested_by="rollback", idempotency_key=cmd_dict.get("idempotency_key")
        )
        ok, detail = actuator.rollback(cmd)
        evidence.setdefault("results", []).append({"at": time.time(), "ok": ok, "detail": detail, "rollback": True})
        if ok:
            evidence.setdefault("timestamps", {})["rolledback_at"] = time.time()
        else:
            evidence.setdefault("timestamps", {})["rollback_failed_at"] = time.time()
        evidence.setdefault("approvals", []).append({"type": "rollback", "by": approver, "reason": reason, "method": "separate_channel_secret", "at": time.time()})

        _write_json_atomic(target_path, evidence)

        evt = {
            "event": "command_rolledback" if ok else "command_rollback_failed",
            "command_id": command_id,
            "asset_id": cmd.asset_id,
            "evidence_path": target_path
        }
        self._save_evt(evt)

        return CommandResult(
            status="ROLLEDBACK" if ok else "FAILED",
            command_id=command_id,
            asset_id=cmd.asset_id,
            channel=actuator.channel_name,
            message="rollback_ok" if ok else "rollback_failed",
            details=detail,
            evidence_path=target_path
        )


class UnavailableActuator(BaseActuator):
    """未配置通道：始终拒绝执行。"""
    channel_name = "unavailable"
    def __init__(self, reason: str):
        self.reason = reason
    def execute(self, cmd: Command) -> Tuple[bool, Dict[str, Any]]:
        return False, {"error": self.reason, "simulated": False}


class ExplicitDryRunActuator(BaseActuator):
    """仅在 PORT_DT_ENABLE_ACTUATOR_DRY_RUN=1 时启用的明示预演通道。"""
    channel_name = "dry_run"
    def execute(self, cmd: Command) -> Tuple[bool, Dict[str, Any]]:
        return True, {"dry_run": True, "executed_on_equipment": False, "cmd": asdict(cmd)}
    def rollback(self, cmd: Command) -> Tuple[bool, Dict[str, Any]]:
        return True, {"dry_run": True, "executed_on_equipment": False, "rollback": True, "cmd": asdict(cmd)}


# -----------------------------
# 便捷的自测函数（可选执行）
# -----------------------------
def demo_self_test() -> None:
    """
    本地安全边界自测（默认应拒绝下发）：
    >>> from app.adapters.actuators import demo_self_test; demo_self_test()
    """
    gw = PortSouthboundGateway()

    # 1) 需要双通道确认的示例（岸电功率设定）
    cmd1 = Command(
        asset_id="SHORE-02", asset_type="shore_power",
        action="set", parameters={"setpoint_kw": 1200},
        requested_by="demo", two_channel_required=True
    )
    res1 = gw.dispatch(cmd1)
    print("PENDING =>", res1.status, res1.details)

    # 2) 不需要双通道的示例（堆场照明开灯）
    cmd2 = Command(
        asset_id="LIGHT-Y1", asset_type="lighting",
        action="open", parameters={"register": 40001, "value": 1, "unit_id": 1},
        requested_by="demo"
    )
    res2 = gw.dispatch(cmd2)
    print("LIGHT =>", res2.status, res2.details)

    # 3) 对 1) 进行确认执行
    if res1.status == "PENDING":
        channel_token = os.getenv("PORT_DT_DEMO_CHANNEL_TOKEN")
        if channel_token:
            res3 = gw.confirm(res1.command_id, confirmer="shift_lead", channel_token=channel_token)
            print("CONFIRM =>", res3.status, res3.details)
        else:
            print("CONFIRM => skipped; PORT_DT_DEMO_CHANNEL_TOKEN is not configured")

    print("证据包请查看：data/objects/audit/ 目录（guard-*.json / evt-*.json）")
