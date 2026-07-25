# app/services/portviz/source.py
# ------------------------------------------------------------
# PortViz 数据源选择器：
# - 定义统一协议 PortVizSource（两个方法：get_bootstrap / next_frame）
# - get_source() 按环境变量/配置选择数据源
#   * 默认：dataset（公开/映射数据集的确定性回放）
#   * real：严格 JSONL 实体轨迹适配器，配置错误时快速失败
#
# 前端字段对齐（约定）：
#   Bootstrap:
#     { world:{W,H}, lanes:[[{x,y}...]], yards:[{x,y,w,h}], berth:{x,y,w,h}, qcs:[{x,y}], ycs:[{x,y}] }
#   Frame:
#     { ts, agv:[{lane,s,alarm}], qc:[{busy,trolley}], yc:[{busy}], tr:[{x,y}],
#       hotspots:[{x,y,r}], vessels:[{berth,progress,len}]? }
#
# 落地真实港口时推荐做法：
#   1) 在本目录下放置一个港口布局数据文件（JSON），例如 port_sgsin_demo.json：
#        - world / lanes / yards / berth / qcs / ycs / agv_n / truck_n / hotspots / vessels ...
#        - Copy this template and replace geometry and parameters for a target port.
#   2) 运行时可以：
#        - 显式设置 PORTVIZ_CONFIG=/path/to/your_port.json
#        - 或者在未设置 PORTVIZ_CONFIG 时，自动加载本目录下的 port_sgsin_demo.json（如果存在）。
#   3) 未来接入真实 TOS/调度系统时，只需实现一个 PortVizSource 适配器，并在 get_source() 中加 real 分支。
#
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Protocol, runtime_checkable

# 默认数据文件名（相对本文件所在目录）
DEFAULT_CONFIG_BASENAME = "port_sgsin_demo.json"


@runtime_checkable
class PortVizSource(Protocol):
    """统一协议：任意数据源都实现这两个方法即可对接前端。"""

    def get_bootstrap(self) -> Dict[str, Any]:
        """静态几何（堆场/泊位/车道/QC/YC 等）。"""
        ...

    def next_frame(self, since: Optional[int] = None) -> Dict[str, Any]:
        """返回下一帧动态数据（see frame schema 注释）。"""
        ...


# ------------------------------
# 选择策略 & 配置装载
# ------------------------------


def _resolve_default_config_path() -> Optional[str]:
    """解析配置文件路径优先级：

    1) 若设置了 PORTVIZ_CONFIG，则直接使用该路径；
    2) 否则，若本目录下存在 DEFAULT_CONFIG_BASENAME（默认 port_sgsin_demo.json），则使用它；
    3) 否则，返回 None，表示不使用几何覆盖（mock_source 内置默认布局）。

    这样做的目的就是：你只要把一份港口 JSON 丢到本目录下，就可以零代码切换场景。
    """
    raw = os.getenv("PORTVIZ_CONFIG", "").strip()
    if raw:
        return raw

    here = os.path.dirname(__file__)
    candidate = os.path.join(here, DEFAULT_CONFIG_BASENAME)
    if os.path.isfile(candidate):
        return candidate

    return None


@dataclass
class SourceConfig:
    """可选配置：通过环境变量 / JSON 文件覆盖模拟源的基本参数。"""

    # 数据源模式：dataset / mock / real（默认 dataset，不伪装成实时生产源）
    mode: str = field(default_factory=lambda: os.getenv("PORTVIZ_MODE", "dataset").strip().lower())

    dataset_path: str = field(default_factory=lambda: os.getenv("PORTVIZ_DATASET_PATH", "data/rl/datasets/public_port_ops_v1.csv").strip())
    frames_path: str = field(default_factory=lambda: os.getenv("PORTVIZ_FRAME_PATH", "").strip())

    # 配置路径（可选）：如果指定，将尝试读取 JSON，填充 lanes/yards/berth/qcs/ycs 等静态几何。
    # 默认策略见 _resolve_default_config_path。
    config_path: Optional[str] = field(default_factory=_resolve_default_config_path)

    # 伪随机种子（模拟源用，方便复现）
    #   - 若设置 PORTVIZ_SEED=0 或不设置，则 seed 为 None，表示使用系统随机种子
    #   - 若设置为正整数，则会传给 MockSource，用于固定轨迹
    seed: Optional[int] = field(
        default_factory=lambda: (lambda v: int(v) if v and v != "0" else None)(
            os.getenv("PORTVIZ_SEED", "").strip()
        )
    )

    # 允许直接通过环境变量覆盖世界尺寸（可选）
    world_W: Optional[int] = field(
        default_factory=lambda: (lambda v: int(v) if v and v != "0" else None)(
            os.getenv("PORTVIZ_WORLD_W", "").strip()
        )
    )
    world_H: Optional[int] = field(
        default_factory=lambda: (lambda v: int(v) if v and v != "0" else None)(
            os.getenv("PORTVIZ_WORLD_H", "").strip()
        )
    )


def _load_overrides(cfg: SourceConfig) -> Dict[str, Any]:
    """从配置文件/环境变量装载静态几何覆盖（可选）。

    返回的 overrides 会传给 MockSource：
      - 如果提供了 world/lanes/yards/berth/qcs/ycs/agv_n/truck_n/hotspots/vessels 等字段，
        将覆盖掉模拟源内部的默认布局；
      - 若某些字段缺失，则由 MockSource 内部补默认值。
    """
    overrides: Dict[str, Any] = {}

    # 1) 来自 JSON 配置的覆盖
    if cfg.config_path:
        if os.path.isfile(cfg.config_path):
            try:
                with open(cfg.config_path, "r", encoding="utf-8") as f:
                    overrides.update(json.load(f))
                print(f"[portviz] INFO: loaded layout config from {cfg.config_path!r}")
            except Exception as e:
                print(f"[portviz] WARN: load config failed from {cfg.config_path!r}: {e}")
        else:
            print(f"[portviz] WARN: config file not found: {cfg.config_path!r}")

    # 2) 世界尺寸覆盖（来自环境变量）
    if cfg.world_W or cfg.world_H:
        overrides.setdefault("world", {})
        if cfg.world_W is not None:
            overrides["world"]["W"] = cfg.world_W
        if cfg.world_H is not None:
            overrides["world"]["H"] = cfg.world_H

    return overrides


# ------------------------------
# 工厂：返回一个 PortVizSource 实例
# ------------------------------


def get_source() -> PortVizSource:
    """统一入口：返回一个实现 PortVizSource 协议的实例。

    模式选择逻辑：

    1) mode in {"mock", "simulate", "simulation"}：
         - 使用 app/services/portviz/mock_source.MockSource
         - 把 overrides/seed 传进去，支持端到端复现和场景切换
    2) mode in {"real", "adapter", "prod"}：
         - 严格读取 PORTVIZ_FRAME_PATH 指定的 JSONL 实体轨迹；缺配置即启动失败
    3) 其它值：直接报错，不静默回落到模拟源
    """
    cfg = SourceConfig()
    overrides = _load_overrides(cfg)

    mode = (cfg.mode or "dataset").lower()

    if mode in ("dataset", "replay", "public"):
        from .replay_source import DatasetReplaySource

        return DatasetReplaySource(overrides=overrides, dataset_path=cfg.dataset_path)

    if mode in ("mock", "simulate", "simulation"):
        try:
            # MockSource 位于同目录下的 mock_source.py
            from .mock_source import MockSource
        except Exception as e:
            # 如果 mock_source 尚未创建或导入失败，给出清晰提示（同时避免应用崩溃）
            raise RuntimeError(
                "PortViz: mock_source 未就绪。请先确认 app/services/portviz/mock_source.py 是否存在且可导入。"
            ) from e

        return MockSource(overrides=overrides, seed=cfg.seed)

    elif mode in ("real", "adapter", "prod"):
        if not cfg.frames_path:
            raise ValueError("PORTVIZ_MODE=real requires PORTVIZ_FRAME_PATH pointing to the JSONL frame adapter")
        from .replay_source import JsonLinesPortSource

        return JsonLinesPortSource(overrides=overrides, frames_path=cfg.frames_path)

    else:
        raise ValueError(f"unknown PORTVIZ_MODE={mode!r}; expected dataset, mock, or real")


__all__ = ["PortVizSource", "get_source", "SourceConfig"]
