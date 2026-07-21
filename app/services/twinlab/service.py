from __future__ import annotations
from typing import Any, Dict
from .repo import TwinLabRepo

class TwinLabService:
    """TwinLab backed only by provenance-verified evidence files."""
    def __init__(self) -> None:
        self.repo = TwinLabRepo()

    # 场景工厂
    def scenarios(self) -> Dict[str, Any]:
        return self.repo.scenarios() or {"available": False, "items": [], "reason": "No provenance-verified scenario evidence is configured"}

    def scenarios_run(self) -> Dict[str, Any]:
        return {"ok": False, "executed": False, "reason": "A scenario runner is not configured"}

    def report(self) -> Dict[str, Any]:
        return {"ok": False, "url": None, "reason": "No verified TwinLab result is available for reporting"}

    # 韧性演练
    def drills(self) -> Dict[str, Any]:
        return self.repo.drills() or {"available": False, "items": [], "reason": "No provenance-verified drill evidence is configured"}

    def drills_trigger(self, state: str) -> Dict[str, Any]:
        return {"ok": False, "executed": False, "requested_state": state, "reason": "A real drill orchestrator is not configured"}

    # 数据契约
    def contracts(self) -> Dict[str, Any]:
        return self.repo.contracts() or {"available": False, "items": [], "reason": "No provenance-verified data-contract evidence is configured"}

    def contracts_verify(self) -> Dict[str, Any]:
        contracts = self.repo.contracts()
        if not contracts:
            return {"ok": False, "available": False, "issues": [], "reason": "No verified contracts to validate"}
        issues = [row for row in contracts.get("items", []) if not row.get("schema_ok", False) or row.get("status") not in {"OK", "PASS"}]
        return {"ok": not issues, "available": True, "issues": issues, "_provenance": contracts.get("_provenance")}
