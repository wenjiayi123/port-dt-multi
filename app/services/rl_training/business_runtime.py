"""Hash-verified, observation-compatible offline RL policy switching."""
from __future__ import annotations

import json
import threading
from pathlib import Path

import numpy as np

from .datasets import file_sha256
from .model_artifacts import resolve_model_artifact
from .trainer import SB3_IMPORT_LOCK

ROOT = Path(__file__).resolve().parents[3]
MODULES = ("hvac", "yard_crane", "yard_lighting", "shore_bess", "bess_energy", "coordinated_business")


class BusinessPolicyRegistry:
    def __init__(self, root=ROOT):
        self.root = Path(root).resolve()
        self.lock = threading.RLock()
        self.cache = {}

    def _path(self, relative):
        path = (self.root / relative).resolve()
        if self.root not in path.parents:
            raise ValueError("artifact must stay within repository")
        return path

    def evidence(self, module, *, champion=False):
        if module not in MODULES:
            raise ValueError("unknown business module")
        pointer_path = self.root / "evidence/v7" / module / ("offline_champion.json" if champion else "latest.json")
        if not pointer_path.exists():
            if module == "coordinated_business" and champion:
                return self._coordinated_incumbent()
            return {"module": module, "status": "NO_ADMITTED_RL" if champion else "NO_COMPLETED_RUN", "production_authority": False}
        pointer = json.loads(pointer_path.read_text())
        path = self._path(pointer["report_path"])
        if file_sha256(path) != pointer["report_sha256"]:
            raise ValueError("business RL report hash mismatch")
        report = json.loads(path.read_text())
        if report["run_id"] != pointer["run_id"] or report["status"] != pointer["status"]:
            raise ValueError("business RL pointer/report mismatch")
        if report.get("selected_model_sha256") != pointer["model_sha256"] or report.get("selected_model_path") != pointer["model_path"]:
            raise ValueError("business RL selection mismatch")
        if champion and (not report.get("checks") or not all(report["checks"].values())) and module != "coordinated_business":
            raise ValueError("business RL admission checks are incomplete")
        return {"module": module, **pointer, "report": report, "production_authority": False, "dispatch_allowed": False, "live_data_verified": False}

    def _coordinated_incumbent(self):
        """Keep the qualified V6 RL actor when refinement is rejected."""
        path = self.root / "evidence/v6/coordinated_business/offline_champion.json"
        if not path.exists():
            return {"module": "coordinated_business", "status": "NO_ADMITTED_RL", "production_authority": False}
        pointer = json.loads(path.read_text())
        report_path = self._path(pointer["report_path"])
        if file_sha256(report_path) != pointer["report_sha256"]:
            raise ValueError("V6 incumbent report hash mismatch")
        report = json.loads(report_path.read_text())
        admission = report.get("admission", {})
        if pointer.get("status") != "ADMITTED_OFFLINE_CHAMPION" or not admission.get("passed") or not admission.get("checks") or not all(admission["checks"].values()):
            raise ValueError("V6 incumbent is not admitted")
        training = report.get("training", {})
        selected = next((row for row in training.get("runs", []) if row.get("job_id") == training.get("selected_job_id")), {})
        if (report.get("run_id") != pointer.get("run_id")
                or selected.get("job_id") != pointer.get("selected_job_id")
                or selected.get("model_path") != pointer.get("selected_model_path")
                or selected.get("model_sha256") != pointer.get("selected_model_sha256")):
            raise ValueError("V6 incumbent selection mismatch")
        return {"module": "coordinated_business", "status": "ADMITTED_OFFLINE_RL", "source_version": "v6_retained_incumbent", "run_id": pointer["run_id"], "model_path": pointer["selected_model_path"], "model_sha256": pointer["selected_model_sha256"], "report_sha256": pointer["report_sha256"], "report": report, "production_authority": False, "dispatch_allowed": False, "live_data_verified": False}

    def predict(self, module, observation, *, expected_model_sha256=None):
        with self.lock:
            evidence = self.evidence(module, champion=True)
            if evidence["status"] != "ADMITTED_OFFLINE_RL":
                raise RuntimeError("no admitted RL policy for this module; candidate switching is blocked")
            model_hash = evidence["model_sha256"]
            if expected_model_sha256 and expected_model_sha256 != model_hash:
                raise ValueError("requested policy version is not the active offline champion")
            path = resolve_model_artifact(self.root, evidence["model_path"], model_hash)
            report = evidence["report"]
            cfg = report.get("config", {})
            algorithm = cfg.get("algorithm", "stable_baselines3.SAC")
            key = (module, model_hash, algorithm)
            if key not in self.cache:
                with SB3_IMPORT_LOCK:
                    from stable_baselines3 import DQN, SAC
                    implementations = {"stable_baselines3.SAC": SAC, "stable_baselines3.DQN": DQN}
                    if algorithm not in implementations:
                        raise ValueError("unregistered RL implementation")
                    self.cache[key] = implementations[algorithm].load(path, device="cpu")
            model = self.cache[key]
            obs = np.asarray(observation, dtype=np.float32)
            if not np.isfinite(obs).all() or not model.observation_space.contains(obs):
                raise ValueError("observation violates the trained shape, bounds or finite-value contract")
            action, _ = model.predict(obs, deterministic=True)
            lattice = cfg.get("action_lattice")
            if lattice is not None:
                action = np.asarray(lattice[int(action)], dtype=np.float32)
            if not np.isfinite(action).all():
                raise ValueError("RL actor produced a non-finite action")
            return {"module": module, "run_id": evidence["run_id"], "algorithm": algorithm, "model_sha256": model_hash, "model_sha256_kind": "training_archive_identity", "loaded_artifact_path": str(path.relative_to(self.root)), "loaded_artifact_sha256": file_sha256(path), "report_sha256": evidence["report_sha256"], "observation_dimensions": int(obs.size), "action": np.asarray(action).tolist(), "action_dimensions": int(np.asarray(action).size), "decision_source": "trained_rl_actor", "requires_environment_projection": True, "simulation_mode": True, "production_authority": False, "dispatch_allowed": False, "live_data_verified": False}


BUSINESS_POLICY_REGISTRY = BusinessPolicyRegistry()


class BusinessRLPolicy:
    """Drop-in SB3 predict interface for existing env.step(action) loops.

    Pin an expected hash for a replay/shift; instantiate a new adapter to switch
    versions. The raw actor action still passes through the same environment.
    """
    def __init__(self, module, expected_model_sha256=None, registry=BUSINESS_POLICY_REGISTRY):
        self.module = module
        self.expected_model_sha256 = expected_model_sha256
        self.registry = registry

    def predict(self, observation, deterministic=True):
        if not deterministic:
            raise ValueError("runtime business decisions require deterministic inference")
        receipt = self.registry.predict(self.module, observation, expected_model_sha256=self.expected_model_sha256)
        return np.asarray(receipt["action"], dtype=np.float32), None
