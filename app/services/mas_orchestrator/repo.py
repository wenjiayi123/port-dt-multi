# app/services/mas_orchestrator/repo.py
from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Dict, Optional

class OrchestratorRepo:
    """
    独立数据源封装：
    - 优先读取本模块 data/* 下的 JSON 文件（便于后期“平替”真实对接）
    - 文件不存在、空文件、或 JSON 无效时返回 None；上层必须显示空状态
    """
    def __init__(self, base: Optional[Path] = None) -> None:
        # 默认 data 目录：app/services/mas_orchestrator/data
        self.base = base or (Path(__file__).parent / "data")

    def file_path(self, name: str) -> Path:
        return self.base / name

    def load_json(self, name: str) -> Optional[Dict[str, Any]]:
        """
        读取 JSON 文件；任何异常（不存在/空/格式错）都返回 None
        允许三种文件名：agents.json / graph.json / timeline.json
        也兼容扩展：kpis.json / conflicts.json
        """
        p = self.file_path(name)
        try:
            if not p.exists():
                return None
            if p.stat().st_size == 0:
                return None
            text = p.read_text(encoding="utf-8", errors="ignore").strip()
            if not text:
                return None
            data = json.loads(text)
            # 空 dict/list 统统视为“无有效数据”
            if data is None:
                return None
            if isinstance(data, (list, dict)) and len(data) == 0:
                return None
            if not isinstance(data, (dict, list)):
                return None
            return data  # 交给 service 做结构校验/合并
        except Exception:
            # 任何解析异常都返回 None，由上层报告 unavailable
            return None
