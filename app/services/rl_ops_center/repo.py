# app/services/rl_ops_center/repo.py
from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Optional

class RLOpsRepo:
    """
    读取 RL Ops Center 的可选数据文件（不存在/为空/无效时返回 None）：
      - overview.json      # OPE 概览（leaderboard、summary）
      - policies.json      # 守护栏规则列表
      - signals.json       # 可观测性黄金信号（metrics/thresholds）
      - experiments.json   # 实验/策略列表
      - causal.json        # （可选）因果预估默认结果
    你可以将这些文件放在：app/services/rl_ops_center/data/ 下，前端与后端结构对齐即可替换演示数据。
    """
    def __init__(self, base: Optional[Path] = None) -> None:
        self.base = base or (Path(__file__).parent / "data")

    def _path(self, name: str) -> Path:
        return self.base / name

    def load_json(self, name: str) -> Optional[Any]:
        p = self._path(name)
        try:
            if not p.exists() or p.stat().st_size == 0:
                return None
            text = p.read_text(encoding="utf-8", errors="ignore").strip()
            if not text:
                return None
            data = json.loads(text)
            # 空对象/空数组一律按“无数据”处理，交给 service 兜底
            if isinstance(data, (dict, list)) and len(data) == 0:
                return None
            return data
        except Exception:
            return None

    # 语义化访问器（便于 service 调用，保持可选）
    def get_overview(self)    -> Optional[Any]: return self.load_json("overview.json")
    def get_policies(self)    -> Optional[Any]: return self.load_json("policies.json")
    def get_signals(self)     -> Optional[Any]: return self.load_json("signals.json")
    def get_experiments(self) -> Optional[Any]: return self.load_json("experiments.json")
    def get_causal(self)      -> Optional[Any]: return self.load_json("causal.json")
