from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

# 当前文件所在目录：.../app/services/platform_map
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
SNAPSHOT_PATH = DATA_DIR / "graph_snapshot.json"
PORTS_PATH = DATA_DIR / "ports_master.json"


def _safe_read_json(path: Path) -> Any:
    """安全读取 JSON 文件，失败时返回 None。"""
    try:
        if not path.exists():
            return None
        text = path.read_text(encoding="utf-8")
        return json.loads(text)
    except Exception:
        return None


def _default_snapshot() -> Dict[str, Any]:
    """当 graph_snapshot.json 缺失或损坏时的兜底平台地图。"""
    return {
        "meta": {
            "version": "V3.1-repository-map",
            "updated_at": "2026-08-12T00:00:00+08:00",
            "description": "港口 AI 数字运营平台仓库架构地图：设备 / 数据孪生 / RL 与协同 / 应用运营。",
        },
        "layers": [
            {
                "id": "devices",
                "name": "设备 / 场景层",
                "nodes": [
                    {"id": "qc_yc", "label": "QC / YC / 场桥", "kind": "device"},
                    {"id": "agv_truck", "label": "AGV / 集卡", "kind": "device"},
                    {"id": "shore_power", "label": "岸电 / Shore Power", "kind": "device"},
                    {"id": "bess", "label": "BESS / 储能", "kind": "device"},
                    {"id": "hvac", "label": "HVAC / 冷站", "kind": "device"},
                    {
                        "id": "yard_lighting",
                        "label": "场区照明 / Yard Lighting",
                        "kind": "device",
                    },
                ],
            },
            {
                "id": "data_twin",
                "name": "数据 / 孪生层",
                "nodes": [
                    {
                        "id": "telemetry",
                        "label": "Telemetry / TSDB（遥测 & 时序库）",
                        "kind": "data",
                    },
                    {
                        "id": "sim_scenarios",
                        "label": "Simulation / Scenario Lib",
                        "kind": "data",
                    },
                    {"id": "twin", "label": "Digital Twin / Twin", "kind": "data"},
                    {"id": "twinlab", "label": "TwinLab 演练 / Drills", "kind": "data"},
                    {"id": "twinplus", "label": "TwinPlus 参数模型", "kind": "data"},
                ],
            },
            {
                "id": "rl_coordination",
                "name": "策略 / RL 层",
                "nodes": [
                    {"id": "rl_agv", "label": "RL Model：AGV Charge", "kind": "core"},
                    {"id": "rl_bess", "label": "RL Model：BESS Energy", "kind": "core"},
                    {"id": "rl_yard_crane", "label": "RL Model：Yard Crane", "kind": "core"},
                    {"id": "rl_hvac", "label": "RL Model：HVAC Cooling", "kind": "core"},
                    {
                        "id": "rl_lighting",
                        "label": "RL Model：Yard Lighting",
                        "kind": "core",
                    },
                    {
                        "id": "rl_ops_center",
                        "label": "RL Ops Center / Platform",
                        "kind": "core",
                    },
                    {
                        "id": "mas_orchestrator",
                        "label": "MAS Orchestrator（多智能体协同）",
                        "kind": "core",
                    },
                    {"id": "opsx", "label": "OpsX / Ops Health", "kind": "core"},
                ],
            },
            {
                "id": "apps",
                "name": "应用 / 运营层",
                "nodes": [
                    {
                        "id": "exec_cockpit",
                        "label": "管理驾驶舱（Exec Cockpit）",
                        "kind": "app",
                    },
                    {
                        "id": "exec_closedloop",
                        "label": "执行与闭环 / Dispatch",
                        "kind": "app",
                    },
                    {"id": "energyx", "label": "EnergyX（市场 & 碳）", "kind": "app"},
                    {"id": "compliance", "label": "报表与合规 / Compliance", "kind": "app"},
                    {
                        "id": "twinlab_drills",
                        "label": "TwinLab 演练 / Drills",
                        "kind": "app",
                    },
                    {"id": "ops_copilot", "label": "Ops Copilot", "kind": "app"},
                ],
            },
        ],
    }


def _load_snapshot() -> Tuple[Dict[str, Any], str]:
    """读取平台地图快照，返回 (snapshot_dict, source)。"""
    snapshot = _safe_read_json(SNAPSHOT_PATH)
    if isinstance(snapshot, dict) and isinstance(snapshot.get("layers"), list):
        return snapshot, "platform_map.snapshot"
    return _default_snapshot(), "platform_map.default"


def _default_ports() -> Dict[str, Any]:
    """当 ports_master.json 缺失或损坏时的兜底港口数据。"""
    return {
        "meta": {
            "version": "2025.11-demo",
            "description": "示例港口主数据：国际大港+典型场景，用于平台地图顶层视图（可替换为生产数据）。",
        },
        "ports": [
            {
                "code": "CNSHA",
                "name": "上海港",
                "name_en": "Port of Shanghai",
                "country": "CN",
                "region": "East Asia",
                "throughput_teu_2023": 49000000,
            },
            {
                "code": "SGSIN",
                "name": "新加坡港",
                "name_en": "Port of Singapore",
                "country": "SG",
                "region": "Southeast Asia",
                "throughput_teu_2023": 39010000,
            },
            {
                "code": "NLRTM",
                "name": "鹿特丹港",
                "name_en": "Port of Rotterdam",
                "country": "NL",
                "region": "Europe",
                "throughput_teu_2023": 13400000,
            },
            {
                "code": "CNNGB",
                "name": "宁波舟山港",
                "name_en": "Ningbo-Zhoushan Port",
                "country": "CN",
                "region": "East China",
                "throughput_teu_2023": 35300000,
            },
        ],
    }


def _load_ports() -> Tuple[List[Dict[str, Any]], Dict[str, Any], str]:
    """读取 ports_master.json，如果不存在则使用默认示例数据。

    返回 (ports_list, meta_dict, source_str)
    """
    data = _safe_read_json(PORTS_PATH)
    if data is None:
        d = _default_ports()
        return d.get("ports", []), d.get("meta", {}), "platform_map.ports.default"

    if isinstance(data, dict):
        ports_list = data.get("ports")
        if not isinstance(ports_list, list):
            ports_list = []
        meta = data.get("meta") or {}
        return ports_list, meta, "platform_map.ports.file"

    if isinstance(data, list):
        return data, {}, "platform_map.ports.list"

    d = _default_ports()
    return d.get("ports", []), d.get("meta", {}), "platform_map.ports.default"


def _compute_layer_stats(layers: List[Dict[str, Any]]) -> Dict[str, Any]:
    """统计每一层以及整体的模块数量。"""
    totals = {"device": 0, "data": 0, "core": 0, "app": 0, "other": 0, "total": 0}
    by_layer: Dict[str, Dict[str, Any]] = {}

    for layer in layers:
        lid = str(layer.get("id") or "")
        lname = str(layer.get("name") or lid)
        stats = {
            "id": lid,
            "name": lname,
            "device": 0,
            "data": 0,
            "core": 0,
            "app": 0,
            "other": 0,
            "total": 0,
        }
        nodes = layer.get("nodes") or []
        for n in nodes:
            kind = (n.get("kind") or "").lower()
            key = kind if kind in ("device", "data", "core", "app") else "other"
            stats[key] += 1
            stats["total"] += 1
            totals[key] += 1
            totals["total"] += 1
        by_layer[lid] = stats

    return {"totals": totals, "by_layer": by_layer}


def _compute_node_usage(ports: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """统计每个节点被多少个港口启用。"""
    usage: Dict[str, Dict[str, Any]] = {}
    for p in ports:
        code = p.get("code") or p.get("unlocode") or p.get("name")
        modules = p.get("modules") or {}
        for key in ("device_nodes", "twin_nodes", "rl_nodes", "app_nodes"):
            ids = modules.get(key) or []
            for node_id in ids:
                info = usage.setdefault(node_id, {"ports": set(), "kinds": set()})
                info["ports"].add(code)
                info["kinds"].add(key)

    # 把 set 转成 list，顺便计算数量
    for node_id, info in usage.items():
        ports_list = sorted([p for p in info["ports"] if p])
        info["ports"] = ports_list
        info["ports_count"] = len(ports_list)
        info["kinds"] = sorted(info["kinds"])

    return usage


def get_graph(di: Any) -> Dict[str, Any]:
    """Return repository architecture configuration, not live port metrics."""
    del di
    snapshot, snapshot_source = _load_snapshot()
    layers = snapshot.get("layers") or []
    meta = snapshot.get("meta") or {}
    stats = _compute_layer_stats(layers)

    result: Dict[str, Any] = {
        "available": bool(layers),
        "layers": layers,
        "meta": meta,
        "_source": snapshot_source,
        "data_class": "repository_architecture_configuration_not_runtime_topology",
        "stats": stats,
        "deployment": {
            "runtime_topology_connected": False,
            "target_domain": "上海港公开数据标定与离线盲测",
            "production_instances": None,
            "required_adapters": [
                "CMDB/资产台账",
                "服务发现与健康检查",
                "TOS/ECS/EMS 实例绑定",
                "现场网络与设备拓扑",
            ],
        },
    }
    return result
