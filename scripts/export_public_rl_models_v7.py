"""Export immutable SB3 weights without local paths in regenerable schedules."""
from __future__ import annotations

import base64
import hashlib
import json
import re
import zipfile
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import SAC, DQN

from app.services.rl_training.datasets import file_sha256

ROOT = Path(__file__).resolve().parents[1]
LOCAL_PATH = re.compile(rb"(?:/(?:Users|home)/[^/\x00\s]+|[A-Z]:\\Users\\[^\\\x00\s]+)")


def check_serialized(value):
    if isinstance(value, dict):
        for key, child in value.items():
            if key == ":serialized:":
                if LOCAL_PATH.search(base64.b64decode(child)):
                    raise ValueError("serialized metadata contains a local account path")
            else:
                check_serialized(child)
    elif isinstance(value, list):
        for child in value:
            check_serialized(child)


def main():
    torch.set_num_threads(1)
    sources = {}
    def add(path, expected, algorithm):
        item = sources.setdefault(expected, {"paths": [], "algorithm": algorithm})
        if path not in item["paths"]:
            item["paths"].append(path)
    for path in sorted((ROOT / "evidence/v7").glob("*/runs/*/report.json")):
        report = json.loads(path.read_text())
        for row in report.get("results", []):
            if row["model_path"].endswith(".zip"):
                add(row["model_path"], row["model_sha256"], report.get("config", {}).get("algorithm", "stable_baselines3.SAC"))
        if path.parents[2].name == "coordinated_business":
            cfg = json.loads(path.with_name("config.json").read_text())
            for row in cfg["seed_sources"]:
                # The V6 incumbent is already a published historical artifact.
                if row["seed"] != 726:
                    add(row["model_path"], row["model_sha256"], "stable_baselines3.SAC")
    destination = ROOT / "evidence/v7/public_models"
    destination.mkdir(parents=True, exist_ok=True)
    exports = []
    for expected, item in sources.items():
        original = ROOT / item["paths"][0]
        if file_sha256(original) != expected:
            raise ValueError("source model hash mismatch")
        with zipfile.ZipFile(original) as archive:
            members = {name: archive.read(name) for name in archive.namelist()}
        data = json.loads(members["data"])
        if not isinstance(data["learning_rate"], (int, float)):
            raise ValueError("only scalar learning-rate schedules may be reconstructed")
        removed = [key for key in ("lr_schedule", "exploration_schedule") if key in data]
        for key in removed:
            del data[key]
        check_serialized(data)
        public_data = json.dumps(data, ensure_ascii=True, indent=2).encode()
        target = destination / (expected + ".zip")
        temp = target.with_suffix(".tmp")
        with zipfile.ZipFile(temp, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for name, raw in members.items():
                payload = public_data if name == "data" else raw
                if LOCAL_PATH.search(payload):
                    raise ValueError("model member contains a local account path")
                archive.writestr(zipfile.ZipInfo(name, date_time=(2026, 9, 7, 0, 0, 0)), payload, compress_type=zipfile.ZIP_DEFLATED)
        if target.exists():
            if target.read_bytes() != temp.read_bytes():
                raise ValueError("refusing to overwrite a different public model export")
            temp.unlink()
        else:
            temp.replace(target)
        unchanged = {name: hashlib.sha256(raw).hexdigest() for name,raw in members.items() if name != "data"}
        with zipfile.ZipFile(target) as archive:
            assert all(hashlib.sha256(archive.read(name)).hexdigest() == digest for name,digest in unchanged.items())
        cls = SAC if item["algorithm"] == "stable_baselines3.SAC" else DQN
        before, after = cls.load(original, device="cpu"), cls.load(target, device="cpu")
        rng = np.random.default_rng(20260907)
        probes = np.clip(rng.normal(0, .5, size=(32,) + before.observation_space.shape), before.observation_space.low, before.observation_space.high).astype(np.float32)
        np.testing.assert_array_equal(before.predict(probes, deterministic=True)[0], after.predict(probes, deterministic=True)[0])
        for progress in (1.0, .8, .4, 0.0):
            assert before.lr_schedule(progress) == after.lr_schedule(progress)
            if isinstance(before, DQN):
                assert before.exploration_schedule(progress) == after.exploration_schedule(progress)
        assert before._n_updates == after._n_updates and before.num_timesteps == after.num_timesteps
        exports.append({"training_model_sha256": expected, "source_paths": item["paths"], "algorithm": item["algorithm"],
                        "public_model_path": str(target.relative_to(ROOT)), "public_model_sha256": file_sha256(target),
                        "removed_regenerable_fields": removed, "unchanged_zip_member_sha256": unchanged,
                        "weights_and_optimizer_members_identical": True, "inference_identical": True, "inference_probe_count": 32,
                        "schedules_identical": True, "training_counters_identical": True})
    result = {"schema": "port-sb3-public-metadata-export.v1", "models": exports,
              "method": "Remove cached scalar learning-rate and DQN exploration schedules; SB3 load rebuilds them from unchanged hyperparameters. Network and optimizer bytes are unchanged.",
              "source_model_archives_modified": False, "historical_reports_modified": False, "production_authority": False}
    (destination / "manifest.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"status": "PASS", "exports": len(exports), "network_optimizer_and_inference_identical": True}))


if __name__ == "__main__":
    main()
