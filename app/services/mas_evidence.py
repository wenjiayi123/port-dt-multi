"""Evidence-backed multi-agent coordination view for the V3 homepage.

The view is deliberately open-loop: a hash-verified SAC artifact receives one
canonical state selected from the chronological blind-test partition.  Its
bounded controls are translated into coordination contracts for QC, YC/AGV,
BESS and shore load.  No device inventory, dispatch receipt, or site KPI is
invented when a terminal adapter is absent.
"""

from __future__ import annotations

from datetime import datetime
import math
from typing import Any

from app.services.rl_training.datasets import (
    FACTOR_COLUMNS,
    NUMERIC_COLUMNS,
    PortDataset,
    load_port_dataset,
)
from app.services.rl_training.trainer import TRAINING_MANAGER


SCENARIOS = {
    "replay": "留出集末端时刻",
    "dense": "留出集高密作业压力",
    "degraded": "留出集设备可用率压力",
}


class MASEvidenceService:
    dataset_id = "public_cn_sha_hourly_v3"

    @staticmethod
    def _number(value: Any) -> float | None:
        try:
            result = float(value)
        except (TypeError, ValueError):
            return None
        return result if math.isfinite(result) else None

    def _sac_evidence(self) -> tuple[dict[str, Any], str]:
        summary = TRAINING_MANAGER.benchmark_summary(dataset_id=self.dataset_id)
        sac = next(
            (
                item
                for item in summary.get("algorithms", [])
                if item.get("id") == "sac" and int(item.get("claim_eligible_runs") or 0) > 0
            ),
            None,
        )
        if not sac:
            raise RuntimeError("SAC blind-test evidence is unavailable")
        registry = TRAINING_MANAGER.model_registry()
        for job_id in reversed(sac.get("job_ids") or []):
            status = TRAINING_MANAGER.status(str(job_id))
            try:
                record = registry.get(str(job_id))
            except KeyError:
                continue
            if status.get("status") == "EVALUATED" and (record.get("artifact") or {}).get("verified") is True:
                return sac, str(job_id)
        raise RuntimeError("No hash-verified evaluated SAC artifact is available")

    @staticmethod
    def _factor(dataset: PortDataset, row_index: int, name: str) -> float | None:
        column = FACTOR_COLUMNS.index(name)
        if dataset.factor_availability[row_index, column] <= 0.5:
            return None
        value = float(dataset.factor_values[row_index, column])
        return value if math.isfinite(value) else None

    def _row_index(self, dataset: PortDataset, scenario: str) -> int:
        _, _, blind_test = dataset.split_three_way(test_ratio=0.2, validation_ratio=0.1)
        candidates = list(range(int(blind_test.start or 0), int(blind_test.stop or dataset.rows)))
        if not candidates:
            raise RuntimeError("chronological blind-test partition is empty")
        if scenario == "replay":
            return candidates[-1]

        def dense_score(index: int) -> float:
            names = (
                "berth_occupancy_ratio",
                "yard_occupancy_ratio",
                "channel_congestion_ratio",
            )
            values = [self._factor(dataset, index, name) for name in names]
            if any(value is None for value in values):
                return -math.inf
            arrivals = float(dataset.values[index, NUMERIC_COLUMNS.index("vessel_arrivals")])
            return sum(float(value) for value in values if value is not None) + 0.1 * arrivals

        def degraded_score(index: int) -> float:
            names = (
                "crane_availability_ratio",
                "equipment_availability_ratio",
                "pilot_tug_availability_ratio",
            )
            values = [self._factor(dataset, index, name) for name in names]
            if any(value is None for value in values):
                return -math.inf
            return sum(1.0 - float(value) for value in values if value is not None)

        score = dense_score if scenario == "dense" else degraded_score
        return max(candidates, key=score)

    def _canonical_state(self, dataset: PortDataset, row_index: int) -> dict[str, Any]:
        state: dict[str, Any] = {
            column: float(dataset.values[row_index, offset])
            for offset, column in enumerate(NUMERIC_COLUMNS)
        }
        state.update(
            {
                column: self._factor(dataset, row_index, column)
                for column in FACTOR_COLUMNS
            }
        )
        timestamp = str(dataset.timestamps[row_index])
        try:
            hour = datetime.fromisoformat(timestamp.replace("Z", "+00:00")).hour
        except ValueError:
            hour = 0
        crane_availability = self._number(state.get("crane_availability_ratio"))
        queue_proxy = float(state["throughput_teu"]) * max(
            0.0, 1.0 - (crane_availability if crane_availability is not None else 1.0)
        )
        state.update(
            {
                "hour": hour,
                "soc": 0.55,
                "queue": queue_proxy,
                "last_bess_kw": 0.0,
                "episode_progress": row_index / max(1, dataset.rows - 1),
            }
        )
        return state

    @staticmethod
    def _metric(sac: dict[str, Any], name: str) -> float | None:
        value = ((sac.get("metrics") or {}).get(name) or {}).get("mean")
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def build(self, *, scenario: str = "replay") -> dict[str, Any]:
        if scenario not in SCENARIOS:
            raise ValueError(f"unsupported MAS evidence scenario: {scenario}")
        dataset = load_port_dataset(self.dataset_id, TRAINING_MANAGER.data_root)
        sac, job_id = self._sac_evidence()
        row_index = self._row_index(dataset, scenario)
        state = self._canonical_state(dataset, row_index)
        inference = TRAINING_MANAGER.predict(job_id, {"state": state})
        control = dict(inference.get("decoded_control") or {})
        safety = dict(inference.get("safety_envelope") or {})

        berth = self._number(state.get("berth_occupancy_ratio"))
        yard = self._number(state.get("yard_occupancy_ratio"))
        crane = self._number(state.get("crane_availability_ratio"))
        equipment = self._number(state.get("equipment_availability_ratio"))
        congestion = self._number(state.get("channel_congestion_ratio"))
        reefer_kw = self._number(state.get("reefer_load_kw"))
        service_factor = self._number(control.get("service_factor"))
        yard_flow = self._number(control.get("yard_flow_command"))
        bess_kw = self._number(control.get("bess_kw"))
        projected_soc = self._number(control.get("projected_soc"))
        berth_priority = self._number(control.get("berth_priority"))

        agents = {
            "qc": [
                {
                    "id": "QC coordination contract",
                    "status": "model-derived recommendation",
                    "detail": f"availability={crane:.3f} · service_factor={service_factor:.3f}" if crane is not None and service_factor is not None else "待接入港口",
                }
            ],
            "yc": [
                {
                    "id": "YC flow contract",
                    "status": "model-derived recommendation",
                    "detail": f"yard_occupancy={yard:.3f} · yard_flow={yard_flow:+.3f}" if yard is not None and yard_flow is not None else "待接入港口",
                }
            ],
            "agv": [
                {
                    "id": "AGV dispatch contract",
                    "status": "adapter pending",
                    "detail": f"equipment_availability={equipment:.3f} · congestion={congestion:.3f}" if equipment is not None and congestion is not None else "待接入港口",
                }
            ],
            "bess": [
                {
                    "id": "BESS control contract",
                    "status": "model-derived recommendation",
                    "soc": projected_soc,
                    "power_kw": bess_kw,
                    "detail": "positive=charge, negative=discharge",
                }
            ],
            "shore": [
                {
                    "id": "Shore / reefer load contract",
                    "status": "public-data calibrated replay",
                    "power_kw": reefer_kw,
                    "detail": f"base_load={float(state['base_load_kw']):.1f} kW",
                }
            ],
        }

        graph = {
            "nodes": [
                {"id": "vessel", "name": "Vessel demand", "category": "vessel"},
                {"id": "qc", "name": "QC agent", "category": "qc"},
                {"id": "yc", "name": "YC agent", "category": "yc"},
                {"id": "agv", "name": "AGV contract", "category": "agv"},
                {"id": "bess", "name": "BESS agent", "category": "bess"},
                {"id": "shore", "name": "Shore load", "category": "shore"},
            ],
            "edges": [
                {"source": "vessel", "target": "qc", "relation": "berth priority"},
                {"source": "qc", "target": "agv", "relation": "container handoff"},
                {"source": "agv", "target": "yc", "relation": "yard flow"},
                {"source": "shore", "target": "bess", "relation": "net-load balance"},
                {"source": "bess", "target": "qc", "relation": "demand envelope"},
            ],
        }
        timeline = {
            "categories": ["Safety", "QC", "YC/AGV", "BESS"],
            "items": [
                {"name": "software envelope", "category": "Safety", "start": 0, "end": 2},
                {"name": f"service {service_factor:.3f}" if service_factor is not None else "service pending", "category": "QC", "start": 2, "end": 42},
                {"name": f"yard flow {yard_flow:+.3f}" if yard_flow is not None else "yard flow pending", "category": "YC/AGV", "start": 5, "end": 45},
                {"name": f"{bess_kw:+.0f} kW" if bess_kw is not None else "BESS pending", "category": "BESS", "start": 0, "end": 60},
            ],
        }

        conflicts: list[dict[str, Any]] = []
        if berth is not None and berth >= 0.85:
            conflicts.append({"severity": "warn", "detail": f"泊位占用率 {berth:.1%}", "proposal": f"SAC berth_priority={berth_priority:+.3f}"})
        if yard is not None and yard >= 0.85:
            conflicts.append({"severity": "warn", "detail": f"堆场占用率 {yard:.1%}", "proposal": f"SAC yard_flow={yard_flow:+.3f}"})
        if crane is not None and crane <= 0.75:
            conflicts.append({"severity": "warn", "detail": f"岸桥可用率 {crane:.1%}", "proposal": f"服务因子限幅至 {service_factor:.3f}"})
        for missing in safety.get("missing_site_claim_factors") or []:
            conflicts.append({"severity": "warn", "detail": f"现场字段缺失：{missing}", "proposal": "保持建议态；待接入港口后才可申请执行权限"})
        if not conflicts:
            conflicts.append({"severity": "info", "detail": "软件包络未发现冲突", "proposal": "现场互锁、TOS/PLC 回执仍待接入港口"})

        return {
            "available": True,
            "mode": "public_data_calibrated_replay_plus_model_inference",
            "production_authority": False,
            "scenario": {"id": scenario, "label": SCENARIOS[scenario]},
            "kpis": {
                "throughput_teu": self._metric(sac, "throughput_teu"),
                "delay_index_mean": self._metric(sac, "delay_index_mean"),
                "peak_kw": self._metric(sac, "peak_kw"),
                "energy_kwh": self._metric(sac, "grid_energy_kwh"),
                "carbon_kg": self._metric(sac, "carbon_kg"),
                "basis": "SAC three-seed mean over fixed chronological blind-test evaluations",
            },
            "agents": agents,
            "graph": graph,
            "timeline": timeline,
            "conflicts": conflicts,
            "decision": {
                "job_id": job_id,
                "algorithm": inference.get("algorithm"),
                "implementation": inference.get("implementation"),
                "decoded_control": control,
                "safety_envelope": safety,
                "rendered": inference.get("rendered"),
            },
            "evidence": {
                "dataset_id": dataset.dataset_id,
                "dataset_sha256": dataset.fingerprint,
                "selected_blind_test_row": row_index,
                "selected_timestamp": dataset.timestamps[row_index],
                "formal_seed_count": len(sac.get("distinct_seeds") or []),
                "formal_run_count": int(sac.get("claim_eligible_runs") or 0),
                "evaluation_protocol": "70% train / 10% validation / 20% chronological blind test; 10 episodes per formal seed",
                "state_contract": "public aggregate and reanalysis factors; terminal identifiers and device inventory are not synthesized",
                "simulator_assumptions": {"initial_soc": 0.55, "last_bess_kw": 0.0, "queue_proxy": "throughput_teu * (1 - crane_availability_ratio)"},
            },
            "site_replacement": {
                "status": "pending_port_connection",
                "required": ["TOS task graph", "QC/YC/AGV asset IDs and states", "BMS/SOC", "shore-power meter", "PLC/interlock acknowledgements"],
            },
        }
