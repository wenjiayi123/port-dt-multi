"""Export one hash-pinned V3 policy for clone-safe dashboard inference."""

from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

from app.services.rl_training.datasets import file_sha256, load_port_dataset


ROOT = Path(__file__).resolve().parents[1]
ADVANTAGE = ROOT / "evidence/v3/shanghai_public_advantage_v3.json"
OUTPUT = ROOT / "evidence/v3/runtime"


def main() -> None:
    advantage = json.loads(ADVANTAGE.read_text(encoding="utf-8"))
    selected = advantage.get("selected") or {}
    job_ids = selected.get("job_ids") or []
    model_hashes = selected.get("model_sha256") or []
    if not job_ids or not model_hashes:
        raise RuntimeError("V3 selected policy evidence is unavailable")
    job_id = str(job_ids[0])
    run_dir = ROOT / "data/rl/runs" / job_id
    model_source = run_dir / "model.zip"
    config_source = run_dir / "config.json"
    manifest_source = run_dir / "manifest.json"
    if not all(path.is_file() for path in (model_source, config_source, manifest_source)):
        raise FileNotFoundError(f"selected runtime artifacts are incomplete: {job_id}")
    observed_model_hash = file_sha256(model_source)
    if observed_model_hash != str(model_hashes[0]):
        raise ValueError("selected runtime model hash does not match advantage evidence")
    config = json.loads(config_source.read_text(encoding="utf-8"))
    dataset = load_port_dataset(str(config["dataset_id"]), ROOT / "data/rl/datasets")
    if dataset.fingerprint != str(config.get("dataset_fingerprint") or ""):
        raise ValueError("selected runtime model dataset hash is stale")

    OUTPUT.mkdir(parents=True, exist_ok=True)
    model_target = OUTPUT / "selected_sac_v3.zip"
    config_target = OUTPUT / "selected_sac_v3.config.json"
    shutil.copy2(model_source, model_target)
    config_target.write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    metadata = {
        "schema": "port-dt-v3-runtime-policy.v1",
        "exported_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "job_id": job_id,
        "algorithm": selected.get("algorithm"),
        "implementation": selected.get("implementation"),
        "seed": (selected.get("seeds") or [None])[0],
        "dataset_id": dataset.dataset_id,
        "dataset_sha256": dataset.fingerprint,
        "environment_version": selected.get("environment_version"),
        "model_artifact": model_target.name,
        "model_sha256": observed_model_hash,
        "config_artifact": config_target.name,
        "config_sha256": hashlib.sha256(config_target.read_bytes()).hexdigest(),
        "selection_protocol": advantage.get("selection_protocol"),
        "production_authority": False,
        "purpose": "clone-safe deterministic dashboard inference; not production dispatch",
    }
    metadata_target = OUTPUT / "runtime_model.json"
    metadata_target.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    sidecar = OUTPUT / "runtime_model.sha256"
    sidecar.write_text(
        "".join(
            f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.name}\n"
            for path in (model_target, config_target, metadata_target)
        ),
        encoding="utf-8",
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
