from __future__ import annotations
from typing import Any, Dict

from . import repo


def _dump(model: Any):
    """兼容 BaseModel / pydantic v1/v2 / dict / 普通对象的轻量序列化。"""
    if hasattr(model, "model_dump"):
        # pydantic v2
        return model.model_dump()
    if hasattr(model, "dict"):
        # pydantic v1
        return model.dict()
    return model


def _ensure() -> None:
    """确保 TwinPlus 必需数据就绪；若缺失则生成上海港示例数据。"""
    try:
        repo.ensure_data("shanghai")
    except Exception:
        # 容错：不因兜底失败而阻断请求
        pass


# -----------------------------------------------------
#  TwinPlus 服务层（供 FastAPI 路由调用）
# -----------------------------------------------------

async def get_port_profile() -> Dict[str, Any]:
    """返回港口画像；若缺失则自动引导生成上海港数据。"""
    _ensure()
    data = repo.get_port_profile()
    return _dump(data)


async def bootstrap_shanghai() -> Dict[str, Any]:
    """生成上海港示例数据文件（已存在则跳过），返回写入清单。"""
    result = repo.bootstrap_shanghai()
    return _dump(result)


async def get_fidelity() -> Dict[str, Any]:
    """返回 TwinPlus 模型的 fidelity 检查结果。"""
    _ensure()
    return _dump(repo.compute_fidelity())


async def run_scenarios(payload: Dict[str, Any]) -> Dict[str, Any]:
    """运行场景模拟（默认 baseline）。"""
    _ensure()
    scen = (payload or {}).get("scenario", "baseline")
    return _dump(repo.run_scenario(scen))


async def calibrate_twin(payload: Dict[str, Any]) -> Dict[str, Any]:
    """执行 TwinPlus 模型标定流程。"""
    _ensure()
    return _dump(repo.calibrate())


async def replay_decision(payload: Dict[str, Any]) -> Dict[str, Any]:
    """回放一次决策（TwinPlus）"""
    _ensure()
    return _dump(repo.replay())
