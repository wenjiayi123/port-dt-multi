"""Replay qualified RL actors through the HTTP API and identical legacy windows."""
from __future__ import annotations

import importlib
import json
from datetime import datetime, timezone

import numpy as np
from fastapi.testclient import TestClient

from app.server import app
from app.services.rl_training.business_runtime import BUSINESS_POLICY_REGISTRY
from app.services.rl_training.datasets import file_sha256, load_port_dataset
from scripts.train_business_rl_v7 import ROOT, RECIPES, write, relative, evaluate, compare
from scripts.train_coordinated_port_business_v6 import make_env


def main():
    client = TestClient(app)
    overview = client.get("/api/rl/business-v7/evidence")
    overview.raise_for_status()
    active = {row["module"]: row for row in overview.json()["active_champions"]}
    assert active["coordinated_business"]["source_version"] == "v6_retained_incumbent"
    assert active["shore_bess"]["status"] == active["bess_energy"]["status"] == "NO_ADMITTED_RL"
    receipts = []
    for name in ("yard_lighting", "hvac", "yard_crane"):
        evidence = BUSINESS_POLICY_REGISTRY.evidence(name, champion=True)
        if evidence["status"] != "ADMITTED_OFFLINE_RL":
            raise RuntimeError(name + " has no qualified RL actor")
        report = evidence["report"]
        config = report["config"]
        selected = next(r for r in report["results"] if r["seed"] == report["selected_seed"])
        module_name, class_name, _ = RECIPES[name]
        module = importlib.import_module("app.services.rl_model." + module_name + ".v3_environment")
        dataset = module.load_dataset(config["config"])
        train, _, test = module.chronological_slices(dataset)
        factory = lambda: getattr(module, class_name)(dataset, test, config=config["config"], normalization_slice=train, episode_steps=config["episode_steps"], training=False, record_trace=False)
        env = factory()
        observation, _ = env.reset(options={"start_index": 0})
        response = client.post(f"/api/rl/business-v7/{name}/predict", json={"observation": observation.tolist(), "expected_model_sha256": evidence["model_sha256"]})
        response.raise_for_status()
        receipt = response.json()
        _, _, _, _, step = env.step(receipt["action"])
        assert not step["guardrail_violation"]
        env.close()
        wrong = client.post(f"/api/rl/business-v7/{name}/predict", json={"observation": [0.0]})
        assert wrong.status_code == 422
        old_pointer = json.loads((ROOT / f"evidence/v3/{name}/latest.json").read_text())
        old_report = json.loads((ROOT / old_pointer["report_path"]).read_text())
        old_model = old_report["artifacts"]["models"][0]
        old_path = ROOT / old_model["path"]
        assert file_sha256(old_path) == old_model["sha256"]
        actor = module.NumpyMLPPolicy.load(old_path)
        old_eval = evaluate(module, factory, module.artifact_policy(actor), selected["test"]["evaluation"]["starts"])
        comparison = compare(selected["test"]["evaluation"], old_eval)
        receipts.append({"module": name, "http_status": response.status_code, "receipt": receipt, "derived_business_step": step["business_step"], "invalid_observation_rejected": True, "comparison_vs_previous_distilled_actor_same_windows": comparison, "previous_actor_sha256": old_model["sha256"]})
    # The joint service must continue to expose the V6 RL incumbent, not the
    # rejected refinement or a control-algorithm substitute.
    joint = BUSINESS_POLICY_REGISTRY.evidence("coordinated_business", champion=True)
    pointer = json.loads((ROOT / "evidence/v6/coordinated_business/offline_champion.json").read_text())
    cfg = json.loads((ROOT / pointer["selected_model_path"]).with_name("config.json").read_text())
    dataset = load_port_dataset(cfg["dataset_id"])
    train, _, test = dataset.split_three_way(0.2, 0.1)
    env = make_env(dataset, test, train, cfg)
    obs, _ = env.reset(options={"start_index": 0})
    response = client.post("/api/rl/business-v7/coordinated_business/predict", json={"observation": obs.tolist()})
    response.raise_for_status()
    joint_receipt = response.json()
    assert joint_receipt["model_sha256"] == pointer["selected_model_sha256"]
    env.step(joint_receipt["action"])
    env.close()
    blocked = {}
    for name in ("shore_bess", "bess_energy"):
        if BUSINESS_POLICY_REGISTRY.evidence(name, champion=True)["status"] == "ADMITTED_OFFLINE_RL":
            continue
        response = client.post(f"/api/rl/business-v7/{name}/predict", json={"observation": [0.0]})
        assert response.status_code == 409
        blocked[name] = response.status_code
    report = {"schema": "port-real-rl-runtime-verification.v7", "verified_at": datetime.now(timezone.utc).isoformat(), "status": "PASS", "active_champions_distinguished_from_latest_candidates": True, "specialist_receipts": receipts, "retained_joint_rl_receipt": joint_receipt, "rejected_candidates_blocked": blocked, "production_authority": False, "dispatch_allowed": False, "external_device_commands": 0}
    path = ROOT / "evidence/v7/runtime_verification.json"
    write(path, report)
    print(json.dumps({"status": "PASS", "report": relative(path), "modules": [r["module"] for r in receipts], "joint_incumbent_retained": True, "rejected_candidates_blocked": blocked}))


if __name__ == "__main__":
    main()
