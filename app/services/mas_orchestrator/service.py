# app/services/mas_orchestrator/service.py
from __future__ import annotations
from datetime import datetime
from typing import Any, Dict, List, Optional

from .repo import OrchestratorRepo

class OrchestratorService:
    """
    业务编排服务（数据文件驱动的接港骨架）：
    - get_overview：整合 KPI / agents / graph / timeline / conflicts
      * data/*.json 缺失或无效时返回空状态，不生成替代业务指标
    - propose：生成一份“可读动作清单”和 plan_id（不落库）
    - simulate：明确返回工程占位状态，供后续接入 twin/sim
    - dispatch：始终返回 dry-run，不伪装成生产下发成功
    """
    def __init__(self) -> None:
        self.repo = OrchestratorRepo()
        self._last_plan_id: Optional[str] = None

    # ========= 公共接口 ========= #
    def get_overview(self) -> Dict[str, Any]:
        """
        汇总概览（允许 data/*.json 为空）：
        - 优先采用 data/agents.json、graph.json、timeline.json
        - 可选 data/kpis.json、conflicts.json
        - 若某项缺失/空，返回空结构并在 provenance 中标明
        """
        loaded = {
            "kpis": self.repo.load_json("kpis.json"),
            "agents": self.repo.load_json("agents.json"),
            "graph": self.repo.load_json("graph.json"),
            "timeline": self.repo.load_json("timeline.json"),
            "conflicts": self.repo.load_json("conflicts.json"),
        }
        kpis = loaded["kpis"] or {}
        agents = loaded["agents"] or {}
        graph = loaded["graph"] or {"nodes": [], "edges": []}
        timeline = loaded["timeline"] or {"categories": [], "items": []}
        conflicts = loaded["conflicts"] or []

        # 轻量健壮性修补：保障前端必需字段存在
        if "nodes" not in graph or "edges" not in graph:
            graph = {"nodes": [], "edges": []}
        if "categories" not in timeline or "items" not in timeline:
            timeline = {"categories": [], "items": []}

        return {
            "ts": datetime.utcnow().isoformat()+"Z",
            "kpis": kpis,
            "agents": agents,
            "graph": graph,
            "timeline": timeline,
            "conflicts": conflicts,
            "_provenance": {
                key: ("module_json_file" if value is not None else "unavailable")
                for key, value in loaded.items()
            },
        }

    def propose(self, horizon_min: int = 120) -> Dict[str, Any]:
        """
        生成一份“可读动作清单”：根据 agents 当前状态做几个合理动作（演示）
        - 真正接入时，把这里替换为 RL/优化器输出即可
        """
        self._last_plan_id = f"MAS-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"
        ag = self.repo.load_json("agents.json") or {}
        actions: List[Dict[str, Any]] = []

        # 规则1：若有 idle 的 QC，尝试切到最近的 vessel
        idle_qc = [x for x in ag.get("qc", []) if x.get("status") == "idle"]
        busy_vessel_jobs = [x.get("job") for x in ag.get("qc", []) if x.get("job")]
        candidate_vessel = busy_vessel_jobs[0] if busy_vessel_jobs else None
        if idle_qc and candidate_vessel:
            actions.append({"agent": idle_qc[0]["id"], "action": f"switch_to {candidate_vessel} hold-2", "eta_min": 6})

        # 规则2：若有 AGV 在 charging 和一个在 enroute，则优先 enroute
        agv = ag.get("agv", [])
        if any(x.get("status") == "charging" for x in agv) and any(x.get("status") == "enroute" for x in agv):
            actions.append({"agent": "AGV-12", "action": "prioritize yard-block C3", "eta_min": 4})

        # 规则3：若岸电功率较高，让 BESS 放电以防峰值
        shore = ag.get("shore", [])
        total_kw = sum(int(x.get("power_kw", 0)) for x in shore)
        if total_kw >= 4000:
            actions.append({"agent": "BESS-1", "action": "discharge 1.2MW for 20min", "eta_min": 1})

        return {
            "plan_id": self._last_plan_id,
            "horizon_min": horizon_min,
            "actions": actions,
            "status": "proposed" if actions else "insufficient_data",
            "source": "module_json_file",
        }

    def simulate(self, scenario: str = "dense_berthing") -> Dict[str, Any]:
        return {"status": "engineering_placeholder", "scenario": scenario, "metrics": None, "rendered": False}

    def dispatch(self) -> Dict[str, Any]:
        return {"status": "dry_run_only", "executed": False, "job_id": self._last_plan_id}
