"""Build the replaceable V5 port-wide scenario from the public Shanghai V4 base."""

from __future__ import annotations

import argparse
import csv
import json
import math
from datetime import datetime
from pathlib import Path

from app.services.rl_training.datasets import (
    CANONICAL_COLUMNS,
    FACTOR_COLUMNS,
    PORT_WIDE_COLUMNS,
    REGULATORY_COLUMNS,
    file_sha256,
    write_extended_rows,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config/integrated_port_scenario_v5.json"
DATA_ROOT = ROOT / "data/rl/datasets"


def bounded(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default=str(CONFIG_PATH.relative_to(ROOT))
    )
    parser.add_argument("--replace-existing", action="store_true")
    args = parser.parse_args()
    config_path = (ROOT / args.config).resolve()
    if not config_path.is_relative_to((ROOT / "config").resolve()):
        parser.error("--config must resolve beneath config/")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    base_id = config["base_dataset_id"]
    output_id = config["output_dataset_id"]
    parameters = config["parameters"]
    base_path = DATA_ROOT / f"{base_id}.csv"
    base_meta_path = DATA_ROOT / f"{base_id}.meta.json"
    output_path = DATA_ROOT / f"{output_id}.csv"
    output_meta_path = DATA_ROOT / f"{output_id}.meta.json"
    if (output_path.exists() or output_meta_path.exists()) and not args.replace_existing:
        raise FileExistsError(
            f"dataset already exists: {output_id}; use --replace-existing explicitly"
        )
    base_meta = json.loads(base_meta_path.read_text(encoding="utf-8"))
    rows = []
    with base_path.open("r", encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            timestamp = datetime.fromisoformat(row["timestamp"].replace("Z", "+00:00"))
            hour = timestamp.hour + timestamp.minute / 60.0
            throughput = float(row["throughput_teu"])
            arrivals = float(row["vessel_arrivals"])
            ambient = float(row["ambient_c"])
            tide = float(row["tide_m"])
            wind = float(row["wind_speed_mps"] or 0.0)
            wave = float(row["wave_height_m"] or 0.0)
            current = float(row["current_speed_mps"] or 0.0)
            berth = float(row["berth_occupancy_ratio"] or 0.0)
            yard = float(row["yard_occupancy_ratio"] or 0.0)
            equipment = float(row["equipment_availability_ratio"] or 1.0)
            channel = float(row["channel_congestion_ratio"] or 0.0)
            reefer_load = float(row["reefer_load_kw"] or 0.0)
            pilot_tug = float(row["pilot_tug_availability_ratio"] or 1.0)

            truck_arrivals = (
                throughput
                * float(parameters["truck_teu_share"])
                / float(parameters["average_teu_per_truck"])
            )
            shift_factor = 0.88 + 0.10 * (1.0 + math.cos(2.0 * math.pi * (hour - 13.0) / 24.0)) / 2.0
            labor = bounded(shift_factor - 0.08 * max(0.0, yard - 0.82), 0.72, 0.99)
            gate_capacity_ratio = bounded(
                equipment * labor * (1.0 - 0.18 * max(0.0, yard - 0.75)),
                0.45,
                1.0,
            )
            gate_capacity = float(parameters["nominal_gate_capacity_trucks_per_hour"]) * gate_capacity_ratio
            gate_queue = max(0.0, truck_arrivals - gate_capacity) * 1.8 + 18.0 * yard
            rail_demand = throughput * float(parameters["rail_teu_share"])
            barge_demand = throughput * float(parameters["barge_teu_share"])
            intermodal_capacity_ratio = bounded(
                equipment * labor * (1.0 - 0.20 * channel), 0.40, 1.0
            )

            reefer_occupancy = bounded(
                0.30 + 0.00045 * reefer_load + 0.18 * max(0.0, yard - 0.70),
                0.20,
                0.96,
            )
            reefer_risk = bounded(
                0.05
                + 0.22 * max(0.0, ambient - 25.0) / 15.0
                + 0.45 * (1.0 - equipment)
                + 0.28 * max(0.0, reefer_occupancy - 0.78),
                0.02,
                0.92,
            )
            shore_connection = bounded(
                0.42 + 0.40 * equipment - 0.18 * max(0.0, wave - 1.8),
                0.20,
                0.88,
            )
            shore_power_demand = (
                float(parameters["nominal_shore_power_kw_per_occupied_berth"])
                * berth
                * bounded(0.65 + 0.18 * arrivals, 0.55, 1.15)
            )
            failure_risk = bounded(
                0.05
                + 0.72 * (1.0 - equipment)
                + 0.18 * max(0.0, yard - 0.80)
                + 0.08 * max(0.0, ambient - 30.0) / 10.0,
                0.02,
                0.90,
            )
            maintenance_backlog = bounded(
                0.12 + 0.62 * failure_risk + 0.16 * max(0.0, berth - 0.82),
                0.05,
                0.92,
            )
            pilotage_demand = arrivals * bounded(0.78 + 0.25 * channel, 0.70, 1.10)
            tug_demand = arrivals * bounded(0.90 + 0.35 * wave / 3.5, 0.85, 1.30)
            channel_capacity_ratio = bounded(
                pilot_tug
                * (1.0 - 0.42 * channel)
                * (1.0 - 0.20 * wave / 3.5)
                * (1.0 - 0.12 * wind / 20.0),
                0.15,
                1.0,
            )
            dangerous_goods = bounded(
                float(parameters["dangerous_goods_share_min"])
                + 0.30 * max(0.0, yard - 0.72)
                + 0.04 * max(0.0, channel - 0.70),
                float(parameters["dangerous_goods_share_min"]),
                float(parameters["dangerous_goods_share_max"]),
            )
            dwell_hours = bounded(
                36.0 + 110.0 * max(0.0, yard - 0.62) + 28.0 * channel,
                32.0,
                110.0,
            )
            draft_span = float(parameters["planning_draft_max_m"]) - float(parameters["planning_draft_min_m"])
            planning_draft = float(parameters["planning_draft_min_m"]) + draft_span * bounded(
                0.22 + 0.42 * berth + 0.18 * channel, 0.0, 1.0
            )
            squat = bounded(
                float(parameters["minimum_squat_allowance_m"])
                + 0.25 * current
                + 0.12 * channel,
                float(parameters["minimum_squat_allowance_m"]),
                float(parameters["maximum_squat_allowance_m"]),
            )
            forecast_uncertainty = bounded(
                0.12
                + 0.18 * wind / 20.0
                + 0.22 * wave / 3.5
                + 0.12 * current / 2.5
                + 0.10 * abs(tide) / 3.0
                + 0.08 * (1.0 - equipment),
                0.08,
                0.78,
            )
            port_wide = {
                "truck_arrivals_per_hour": truck_arrivals,
                "gate_queue_trucks": gate_queue,
                "gate_capacity_ratio": gate_capacity_ratio,
                "rail_transfer_demand_teu": rail_demand,
                "barge_transfer_demand_teu": barge_demand,
                "intermodal_capacity_ratio": intermodal_capacity_ratio,
                "reefer_occupancy_ratio": reefer_occupancy,
                "reefer_temperature_risk_ratio": reefer_risk,
                "shore_power_demand_kw": shore_power_demand,
                "shore_power_connection_ratio": shore_connection,
                "equipment_failure_risk_ratio": failure_risk,
                "maintenance_backlog_ratio": maintenance_backlog,
                "labor_availability_ratio": labor,
                "pilotage_demand_vessels": pilotage_demand,
                "tug_demand_vessels": tug_demand,
                "channel_capacity_ratio": channel_capacity_ratio,
                "dangerous_goods_workload_ratio": dangerous_goods,
                "yard_dwell_time_hours": dwell_hours,
                "planning_vessel_draft_m": planning_draft,
                "channel_chart_depth_m": float(parameters["channel_chart_depth_m"]),
                "squat_allowance_m": squat,
                "forecast_uncertainty_ratio": forecast_uncertainty,
            }
            rows.append(
                {
                    **{name: row[name] for name in CANONICAL_COLUMNS},
                    **{name: row.get(name, "") for name in FACTOR_COLUMNS},
                    **{name: row.get(name, "") for name in REGULATORY_COLUMNS},
                    **{name: round(port_wide[name], 6) for name in PORT_WIDE_COLUMNS},
                }
            )

    metadata = {
        **base_meta,
        "dataset_id": output_id,
        "title": "Shanghai public aggregate and reanalysis plus replaceable integrated port scenario v5",
        "created_at": config["created_at"],
        "provenance_type": "public_official_aggregate_reanalysis_plus_predeclared_engineering_port_wide_scenario",
        "evidence_tier": "public_data_offline_integrated_business_scenario",
        "intended_use": "Real learner training and chronological blind evaluation of an integrated recommendation policy; not a field KPI or production authorization",
        "port_profile_id": "cn_sha_integrated_scenario_v5",
        "environment_version": "port_ops_v5",
        "base_dataset": {
            "dataset_id": base_id,
            "sha256": file_sha256(base_path),
            "rows": len(rows)
        },
        "integrated_scenario": {
            "scenario_id": config["scenario_id"],
            "config_artifact": str(config_path.relative_to(ROOT)),
            "config_sha256": file_sha256(config_path),
            "classification": config["evidence_boundary"]["classification"],
            "calibration_warning": config["evidence_boundary"]["calibration_warning"]
        },
        "derived_columns": [
            *base_meta.get("derived_columns", []),
            *PORT_WIDE_COLUMNS
        ],
        "field_provenance": {
            **{name: "inherited_public_or_engineering_source_from_base_v4" for name in (*CANONICAL_COLUMNS[1:], *FACTOR_COLUMNS, *REGULATORY_COLUMNS)},
            **{name: "predeclared_deterministic_engineering_scenario_replace_with_authorized_site_source" for name in PORT_WIDE_COLUMNS}
        },
        "site_replacement_contract": {
            "terminal_operating_system": ["throughput_teu", "berth_occupancy_ratio", "yard_occupancy_ratio", "dangerous_goods_workload_ratio", "yard_dwell_time_hours"],
            "gate_appointment_system": ["truck_arrivals_per_hour", "gate_queue_trucks", "gate_capacity_ratio"],
            "port_community_and_intermodal_systems": ["rail_transfer_demand_teu", "barge_transfer_demand_teu", "intermodal_capacity_ratio"],
            "reefer_monitoring_system": ["reefer_occupancy_ratio", "reefer_temperature_risk_ratio"],
            "shore_power_metering_system": ["shore_power_demand_kw", "shore_power_connection_ratio"],
            "computerized_maintenance_system": ["equipment_failure_risk_ratio", "maintenance_backlog_ratio"],
            "workforce_system": ["labor_availability_ratio"],
            "vessel_call_and_marine_service_systems": ["pilotage_demand_vessels", "tug_demand_vessels", "channel_capacity_ratio", "planning_vessel_draft_m"],
            "hydrographic_authority": ["channel_chart_depth_m", "tide_m", "squat_allowance_m"],
            "forecast_service": ["forecast_uncertainty_ratio"]
        },
        "warning": config["evidence_boundary"]["calibration_warning"] + " Production authority remains false."
    }
    result = write_extended_rows(output_id, rows, metadata, DATA_ROOT)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
