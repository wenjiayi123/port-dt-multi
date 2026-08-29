from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


def evidence_path(repo_root: Path) -> Path:
    return repo_root / "evidence/v3/value_improvement_v32.json"


def load_module_value_improvement(repo_root: Path, module_id: str) -> Dict[str, Any]:
    path = evidence_path(repo_root)
    if not path.exists():
        return {
            "version": "V3.2",
            "status": "evidence_pending",
            "decision": "待生成追加训练验收证据",
        }
    payload = json.loads(path.read_text(encoding="utf-8"))
    return dict((payload.get("modules") or {}).get(module_id) or {
        "version": payload.get("version") or "V3.2",
        "status": "not_in_scope",
    })
