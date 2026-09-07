"""Verify public RL artifacts and the inputs needed to reproduce V7 runs."""
from __future__ import annotations

import json
import hashlib
import zipfile
from pathlib import Path

from app.services.rl_training.datasets import file_sha256
from app.services.rl_training.business_runtime import BusinessPolicyRegistry
from app.services.rl_training.model_artifacts import resolve_model_artifact

ROOT = Path(__file__).resolve().parents[1]


def verify(root=ROOT):
    root = Path(root).resolve()
    from scripts.export_public_rl_models_v7 import LOCAL_PATH, check_serialized
    public_manifest = json.loads((root / "evidence/v7/public_models/manifest.json").read_text())
    for exported in public_manifest["models"]:
        public_path = resolve_model_artifact(root, exported["source_paths"][0], exported["training_model_sha256"])
        with zipfile.ZipFile(public_path) as archive:
            for name in archive.namelist():
                raw = archive.read(name)
                if LOCAL_PATH.search(raw):
                    raise ValueError("public model metadata contains a local account path")
                if name == "data":
                    check_serialized(json.loads(raw))
                elif hashlib.sha256(raw).hexdigest() != exported["unchanged_zip_member_sha256"][name]:
                    raise ValueError("public model weight/optimizer member changed")
    def artifact(relative, expected):
        if str(relative).endswith(".zip"):
            return resolve_model_artifact(root, relative, expected)
        path = (root / relative).resolve()
        if root not in path.parents or not path.is_file() or file_sha256(path) != expected:
            raise ValueError("missing or mismatched V7 artifact: " + str(relative))
        return path
    summary = json.loads((root / "evidence/v7/audit_summary.json").read_text())
    count, environment_steps, optimizer_updates = 0, 0, 0
    for entry in summary["completed_runs"]:
        report_path = artifact(entry["report_path"], entry["report_sha256"])
        report = json.loads(report_path.read_text())
        if report["run_id"] != entry["run_id"] or report["status"] != entry["status"]:
            raise ValueError("V7 report index mismatch")
        if len(report["results"]) != entry["seed_count"]:
            raise ValueError("V7 seed count mismatch")
        for row in report["results"]:
            artifact(row["model_path"], row["model_sha256"])
            count += 1
            environment_steps += row.get("steps", row.get("real_additional_environment_steps", 0))
            optimizer_updates += row.get("optimizer_updates", row.get("real_additional_optimizer_updates", row.get("optimizer_steps", 0)))
        manifest = json.loads(report_path.with_name("manifest.json").read_text())
        for path, expected in manifest.get("input_hashes", {}).items():
            artifact(path, expected)
        if entry["module"] == "coordinated_business":
            cfg = json.loads(report_path.with_name("config.json").read_text())
            for source in cfg["seed_sources"]:
                artifact(source["model_path"], source["model_sha256"])
    if (count, environment_steps, optimizer_updates) != (summary["completed_seed_runs"], summary["new_environment_steps"], summary["new_optimizer_updates"]):
        raise ValueError("V7 training total mismatch")
    registry = BusinessPolicyRegistry(root)
    for specialist in summary["admitted_specialists"]:
        evidence = registry.evidence(specialist["module"], champion=True)
        if evidence["status"] != "ADMITTED_OFFLINE_RL" or evidence["model_sha256"] != specialist["model_sha256"]:
            raise ValueError("V7 active specialist differs from audited champion")
    joint = registry.evidence("coordinated_business", champion=True)
    artifact(joint["model_path"], joint["model_sha256"])
    if joint.get("source_version") != "v6_retained_incumbent":
        raise ValueError("V7 audit requires the preserved V6 champion")
    return {"selected_models": count, "environment_steps": environment_steps, "optimizer_updates": optimizer_updates}


if __name__ == "__main__":
    print("BUSINESS_RL_ARTIFACTS_V7:PASS:" + json.dumps(verify()))
