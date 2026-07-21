from __future__ import annotations
from fastapi import APIRouter, HTTPException
from typing import Any, Dict

router = APIRouter()


def _ensure():
    """确保 TwinPlus 数据文件就绪：若缺失则生成上海港示例数据。"""
    try:
        from .repo import ensure_data
        ensure_data("shanghai")
    except Exception:
        # 容错：不阻断请求，具体错误交由实际端点处理
        pass


@router.post("/bootstrap/shanghai")
async def bootstrap_shanghai():
    """生成上海港示例数据文件（port_profile/params_base/last_calib）。"""
    try:
        from .repo import bootstrap_shanghai as _bootstrap
        result = _bootstrap()
        return {"ok": True, **result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"bootstrap shanghai 失败: {e}")


@router.get("/port_profile")
async def port_profile():
    """读取港口画像（若缺失自动引导生成上海港数据）。"""
    _ensure()
    try:
        from .repo import get_port_profile
        data = get_port_profile()
        if not data:
            raise HTTPException(status_code=404, detail="port_profile.json 为空")
        return data
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"读取 port_profile 失败: {e}")


@router.get("/fidelity")
async def fidelity():
    _ensure()
    from .service import get_fidelity
    return await get_fidelity()


@router.post("/scenarios/run")
async def scenarios_run(payload: Dict[str, Any]):
    _ensure()
    from .service import run_scenarios
    return await run_scenarios(payload)


@router.post("/calibrate")
async def calibrate(payload: Dict[str, Any]):
    _ensure()
    from .service import calibrate_twin
    return await calibrate_twin(payload)


@router.post("/replay")
async def replay(payload: Dict[str, Any]):
    _ensure()
    from .service import replay_decision
    return await replay_decision(payload)
