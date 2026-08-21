"""Build the versioned maritime/customs delay stress scenario for port_ops_v4."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from app.services.rl_training.datasets import (
    CANONICAL_COLUMNS,
    FACTOR_COLUMNS,
    file_sha256,
    write_extended_rows,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config/regulatory_delay_scenario_v4.json"
DATA_ROOT = ROOT / "data/rl/datasets"


def _uniform(seed: int, timestamp: str, stream: str) -> float:
    digest = hashlib.sha256(f"{seed}:{timestamp}:{stream}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / float(2**64 - 1)


def _bounded(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=str(CONFIG_PATH.relative_to(ROOT)),
        help="repository-relative scenario config",
    )
    args = parser.parse_args()
    config_path = (ROOT / args.config).resolve()
    if not config_path.is_relative_to((ROOT / "config").resolve()):
        parser.error("--config must resolve beneath config/")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    base_id = config["base_dataset_id"]
    output_id = config["output_dataset_id"]
    seed = int(config["seed"])
    parameters = config["parameters"]
    base_path = DATA_ROOT / f"{base_id}.csv"
    base_meta_path = DATA_ROOT / f"{base_id}.meta.json"
    base_meta = json.loads(base_meta_path.read_text(encoding="utf-8"))
    output_rows = []
    with base_path.open("r", encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            timestamp = row["timestamp"]
            berth = float(row["berth_occupancy_ratio"])
            yard = float(row["yard_occupancy_ratio"])
            channel = float(row["channel_congestion_ratio"])
            pressure = _bounded((berth + yard + channel) / 3.0, 0.0, 1.0)
            maritime_cfg = parameters["maritime_inspection_ratio"]
            maritime_pulse = (
                maritime_cfg["stress_pulse_gain"]
                if _uniform(seed, timestamp, "maritime")
                < maritime_cfg["stress_pulse_probability"]
                else 0.0
            )
            maritime = _bounded(
                maritime_cfg["base"]
                + maritime_cfg["operational_pressure_gain"] * pressure
                + maritime_pulse,
                0.0,
                maritime_cfg["maximum"],
            )
            customs_cfg = parameters["customs_inspection_ratio"]
            customs_pulse = (
                customs_cfg["stress_pulse_gain"]
                if _uniform(seed, timestamp, "customs")
                < customs_cfg["stress_pulse_probability"]
                else 0.0
            )
            customs = _bounded(
                customs_cfg["base"]
                + customs_cfg["yard_pressure_gain"] * yard
                + customs_pulse,
                0.0,
                customs_cfg["maximum"],
            )
            detention_cfg = parameters["maritime_detention_ratio"]
            detention = _bounded(
                detention_cfg["global_psc_reference"]
                + (
                    detention_cfg["stress_pulse_gain"]
                    if _uniform(seed, timestamp, "detention")
                    < detention_cfg["stress_pulse_probability"]
                    else 0.0
                ),
                0.0,
                detention_cfg["maximum"],
            )
            secondary_cfg = parameters["customs_secondary_check_ratio"]
            secondary = _bounded(
                secondary_cfg["engineering_base"]
                + secondary_cfg["yard_pressure_gain"] * yard
                + (
                    secondary_cfg["stress_pulse_gain"]
                    if _uniform(seed, timestamp, "secondary")
                    < secondary_cfg["stress_pulse_probability"]
                    else 0.0
                ),
                0.0,
                secondary_cfg["maximum"],
            )
            availability_cfg = parameters["inspection_resource_availability_ratio"]
            availability = _bounded(
                availability_cfg["base"]
                - availability_cfg["operational_pressure_loss"] * pressure,
                availability_cfg["minimum"],
                1.0,
            )
            release_cfg = parameters["regulatory_release_ratio"]
            release = _bounded(
                release_cfg["base"]
                - release_cfg["detention_loss"] * detention
                - release_cfg["secondary_check_loss"] * secondary,
                release_cfg["minimum"],
                1.0,
            )
            output_rows.append(
                {
                    **{name: row[name] for name in CANONICAL_COLUMNS},
                    **{name: row.get(name, "") for name in FACTOR_COLUMNS},
                    "maritime_inspection_ratio": round(maritime, 6),
                    "customs_inspection_ratio": round(customs, 6),
                    "maritime_detention_ratio": round(detention, 6),
                    "customs_secondary_check_ratio": round(secondary, 6),
                    "inspection_resource_availability_ratio": round(
                        availability, 6
                    ),
                    "regulatory_release_ratio": round(release, 6),
                }
            )
    metadata = {
        **base_meta,
        "dataset_id": output_id,
        "title": "Shanghai public benchmark plus predeclared regulatory delay stress scenario v4",
        "created_at": config["created_at"],
        "provenance_type": "public_official_aggregate_reanalysis_plus_predeclared_engineering_regulatory_scenario",
        "evidence_tier": "public_data_offline_regulatory_stress_scenario",
        "intended_use": "Offline training and blind-holdout evaluation of maritime/customs delay resilience; not field KPI estimation",
        "port_profile_id": "cn_sha_regulatory_scenario_v4",
        "environment_version": "port_ops_v4",
        "base_dataset": {
            "dataset_id": base_id,
            "sha256": file_sha256(base_path),
            "rows": len(output_rows),
        },
        "regulatory_scenario": {
            "scenario_id": config["scenario_id"],
            "config_artifact": str(config_path.relative_to(ROOT)),
            "config_sha256": file_sha256(config_path),
            "seed": seed,
            "classification": config["evidence_boundary"]["classification"],
            "calibration_warning": config["evidence_boundary"][
                "calibration_warning"
            ],
        },
        "derived_columns": [
            *base_meta.get("derived_columns", []),
            "maritime_inspection_ratio",
            "customs_inspection_ratio",
            "maritime_detention_ratio_global_psc_reference_plus_stress",
            "customs_secondary_check_ratio",
            "inspection_resource_availability_ratio",
            "regulatory_release_ratio",
        ],
        "sources": [
            *base_meta.get("sources", []),
            {
                "publisher": "International Maritime Organization",
                "role": "official_port_state_control_process_and_2024_global_detention_reference",
                "urls": config["evidence_boundary"]["official_process_sources"][:2],
                "boundary": "Global process/reference only; not a Shanghai inspection rate",
            },
            {
                "publisher": "General Administration of Customs of the People's Republic of China",
                "role": "official_inbound_outbound_vessel_declaration_and_boarding_process",
                "url": config["evidence_boundary"]["official_process_sources"][2],
                "boundary": "Process basis only; no local inspection probability or duration inferred",
            },
        ],
        "warning": "PREDECLARED_ENGINEERING_STRESS_SCENARIO_NOT_FIELD_KPI. Regulatory columns are scenario inputs, not Shanghai maritime/customs telemetry. Production dispatch and regulatory release authority remain prohibited.",
    }
    result = write_extended_rows(output_id, output_rows, metadata, DATA_ROOT)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
