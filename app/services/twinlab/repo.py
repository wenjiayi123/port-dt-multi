from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Optional

class TwinLabRepo:
    """
    读取可选外部数据（不存在/为空/无效返回 None）：
      - scenarios.json    # 场景列表与趋势
      - drills.json       # 演练计划/结果/SLA
      - contracts.json    # 数据契约健康
    """
    def __init__(self, base: Optional[Path] = None) -> None:
        self.base = base or (Path(__file__).parent / "data")

    def _load(self, name: str) -> Optional[Any]:
        p = self.base / name
        try:
            if not p.exists() or p.stat().st_size == 0:
                return None
            text = p.read_text(encoding="utf-8", errors="ignore").strip()
            if not text:
                return None
            data = json.loads(text)
            if isinstance(data, (list, dict)) and len(data) == 0:
                return None
            metadata = self._load_metadata(name, data)
            if not metadata:
                return None
            if isinstance(data, dict):
                data = {**data, "_provenance": metadata}
            return data
        except Exception:
            return None

    def _load_metadata(self, name: str, data: Any) -> Optional[dict]:
        inline = data.get("_provenance") if isinstance(data, dict) else None
        sidecar = self.base / f"{name}.meta.json"
        metadata = inline
        if metadata is None and sidecar.exists():
            try:
                metadata = json.loads(sidecar.read_text(encoding="utf-8"))
            except Exception:
                return None
        if not isinstance(metadata, dict):
            return None
        if metadata.get("provenance_type") not in {"public", "port_export", "verified_test"}:
            return None
        if not metadata.get("source_url"):
            return None
        return metadata

    def scenarios(self) -> Optional[Any]:
        return self._load("scenarios.json")

    def drills(self) -> Optional[Any]:
        return self._load("drills.json")

    def contracts(self) -> Optional[Any]:
        return self._load("contracts.json")
