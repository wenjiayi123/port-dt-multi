"""Diagnostic SAC with one neural flex action and BESS constrained to idle.

This is a restricted-control specialist, never a complete Shore+BESS candidate.
No teacher actions, warm starts, test evaluation, forward-file reads or policy
pointer updates are permitted. Run with python -m scripts.train_shore_bess_v8_flex_specialist.
"""
from __future__ import annotations

import argparse
import json
import platform
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from gymnasium import spaces

from app.services.rl_model.shore_bess import v8_environment as physical
from scripts import train_shore_bess_v8 as core


class ShoreBESSV8FlexSpecialistEnv(physical.ShoreBESSV8SACEnv):
    """Same 31 causal states, physical dispatch, FIFO and billing; one actuator."""
    control_scope = "flexible_auxiliary_load_only_bess_held_idle"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.action_space = spaces.Box(-1.0, 1.0, (1,), dtype=np.float32)

    def step(self, action):
        latent = np.asarray(action, dtype=np.float64)
        if latent.shape != (1,) or not np.isfinite(latent).all():
            raise ValueError("flex specialist requires one finite neural action")
        result = super().step(np.asarray([0.0, latent[0]], dtype=np.float32))
        info = result[-1]
        if abs(info["final_action"]["bess_kw"]) > 1e-8:
            raise ValueError("physical projection moved BESS; restricted-control diagnostic invalid")
        info["control_scope"] = self.control_scope
        info["neural_flex_action"] = latent.tolist()
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=30000)
    parser.add_argument("--block", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=912)
    parser.add_argument("--run-id")
    args = parser.parse_args()
    if not 2000 < args.steps <= 30000 or args.block <= 0 or args.steps < 3 * args.block:
        parser.error("diagnostic budget must be 2001..30000 steps and include at least three checkpoints")
    if args.seed < 0 or (args.run_id and (Path(args.run_id).name != args.run_id or args.run_id in {".", ".."})):
        parser.error("nonnegative seed and single-directory run-id required")

    import torch
    import stable_baselines3 as sb3
    import sb3_contrib
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.monitor import Monitor

    torch.set_num_threads(1)
    physical_config = core.load_config(core.CONFIG_PATH)
    dataset = core.load_port_dataset(physical_config["dataset_id"])
    quality = core.checked_quality(dataset)
    train = core.month_slice(dataset, "2024-01-01", "2025-05-01")
    validation = core.month_slice(dataset, "2025-05-01", "2025-08-01")
    starts = core.window_starts(validation.stop - validation.start, 6)
    idle = np.zeros(1, dtype=np.float32)

    def factory(split, *, training=False):
        return lambda: ShoreBESSV8FlexSpecialistEnv(
            dataset, split, config=physical_config, normalization_slice=train,
            episode_steps=168, carbon_price=12.0, seed=args.seed, training=training, record_trace=False)

    run_id = args.run_id or "shore-bess-v8-flex-sac-diagnostic-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    run_dir = core.OUTPUT_ROOT / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    sources = [Path(__file__), Path(core.__file__), Path(physical.__file__), core.CONFIG_PATH,
               core.ROOT / "app/services/rl_model/shore_bess/v3_environment.py",
               core.ROOT / "app/services/rl_training/datasets.py",
               core.ROOT / "app/services/rl_training/statistics.py"]
    source_hashes = {}
    for path in sources:
        destination = run_dir / "source" / core.relative(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
        source_hashes[core.relative(path)] = core.file_sha256(path)
    input_hashes = {core.relative(core.CONFIG_PATH): core.file_sha256(core.CONFIG_PATH), **core.input_hashes(dataset)}
    pointers = {core.relative(path): core.file_sha256(path)
                for version in ("v3", "v7", "v8")
                for path in (core.ROOT / "evidence" / version / "shore_bess").glob("*.json")}
    algorithm_parameters = {"learning_rate": 0.0003, "buffer_size": 200000, "batch_size": 256,
                            "learning_starts": 2000, "train_freq": 1, "gradient_steps": 1,
                            "tau": 0.005, "ent_coef": 0.001}
    probe = factory(train)()
    scaler_names = ("base_scale", "price_scale", "carbon_scale", "soft_cap_kw", "price_center",
                    "carbon_center", "carbon_span", "load_center", "load_std", "reward_scale")
    normalization = {name: float(getattr(probe, name)) for name in scaler_names}
    probe.close()
    config = {"algorithm": "stable_baselines3.SAC", "algorithm_variant": "vanilla_sac_flex_specialist",
              "control_scope": ShoreBESSV8FlexSpecialistEnv.control_scope,
              "bess_controlled_by_rl": False, "flex_controlled_by_rl": True,
              "neural_action_shape": [1], "physical_action_mapping": "[BESS latent=0, flex latent=neural action]",
              "bess_idle_verified_every_environment_step": True,
              "physical_action_mapping_version": physical.ShoreBESSV8SACEnv.action_mapping_version,
              "state_names": list(physical.STATE_NAMES), "observation_dimensions": len(physical.STATE_NAMES),
              "network": [128, 128], "gamma": 1.0, "seed": args.seed, "requested_steps": args.steps,
              "checkpoint_interval": args.block, "algorithm_parameters": algorithm_parameters,
              "carbon_price_cny_per_kg_constraint_multiplier": 12.0,
              "reward_credit_assignment": physical.ShoreBESSV8Env.reward_credit_assignment,
              "physical_config": physical_config, "normalization": normalization,
              "pilot": True, "formal_admission_supported": False,
              "physical_config_legacy_training_and_reward_weights_used": False,
              "gate_reference_only": core.GATE}
    manifest = {"schema": "shore-bess-v8-flex-specialist-manifest.v1", "run_id": run_id,
                "started_at": core.utc_now(), "dataset_id": dataset.dataset_id,
                "dataset_sha256": dataset.fingerprint, "dataset_quality": quality,
                "train": core.split_description(dataset, train), "validation": core.split_description(dataset, validation),
                "validation_starts": starts, "normalization_fit_split": "train_only",
                "test_evaluation_performed": False, "forward_dataset_loaded": False,
                "access_protocol_details": "The complete historical 2024-2025 CSV bytes are loaded and quality-checked. August-December rows are never evaluated or used for gradients, scaling or selection. The separate 2026 forward file is never loaded.",
                "teacher_actions_used": False, "warm_start_used": False,
                "source_sha256": source_hashes, "input_files_sha256": input_hashes,
                "protected_pointer_sha256": pointers,
                "versions": {"python": platform.python_version(), "torch": torch.__version__,
                             "stable_baselines3": sb3.__version__, "sb3_contrib": sb3_contrib.__version__}}
    core.write_json(run_dir / "config.json", config)
    core.write_json(run_dir / "manifest.json", manifest)
    reference = core.evaluate(factory(validation), None, starts, idle)
    core.write_json(run_dir / "validation_reference.json", reference)
    env = Monitor(factory(train, training=True)(), str(run_dir / "monitor.csv"))
    model = sb3.SAC("MlpPolicy", env, gamma=1.0, policy_kwargs={"net_arch": [128, 128]},
                    seed=args.seed, device="cpu", verbose=0, **algorithm_parameters)
    model._v8_optimizer_step_calls = 0
    model._v8_optimizer_steps_by_component = {name: 0 for name in core.model_optimizers(model)}
    hooks = []
    for name, optimizer in core.model_optimizers(model).items():
        def count_step(_optimizer, _args, _kwargs, component=name):
            model._v8_optimizer_step_calls += 1
            model._v8_optimizer_steps_by_component[component] += 1
        hooks.append(optimizer.register_step_post_hook(count_step))
    initial_weights = core.weights_sha256(model)
    initial = core.evaluate(factory(validation), model, starts, idle)
    core.write_json(run_dir / "initial_validation.json", {"evaluation": initial,
                    "comparison": core.compare(initial, reference), "weights_sha256": initial_weights,
                    "parameters": core.model_parameters(model)})
    curve = []
    episode_ledger = core.TrainingEpisodeLedger(run_dir / "training_episodes.csv")

    def checkpoint():
        model_path = run_dir / f"step_{model.num_timesteps}.zip"
        if model_path.exists():
            raise FileExistsError("checkpoint is immutable")
        model.save(model_path)
        evaluation = core.evaluate(factory(validation), model, starts, idle)
        comparison = core.compare(evaluation, reference)
        gates = core.business_gates(evaluation, comparison)
        gates["bess_remained_idle"] = all(abs(row["bess_throughput_kwh"]) <= 1e-8 for row in evaluation["rows"])
        row = {"step": int(model.num_timesteps), "optimizer_updates": int(model._v8_optimizer_step_calls),
               "sb3_update_counter": int(model._n_updates), "checkpoint_phase": "post_rollout_and_optimizer",
               "parameters": core.model_parameters(model), "weights_sha256": core.weights_sha256(model),
               "model_path": core.relative(model_path), "model_sha256": core.file_sha256(model_path),
               "evaluation": evaluation, "comparison": comparison, "gates": gates,
               "optimizer": {key: float(value) for key, value in model.logger.name_to_value.items()
                             if key.startswith("train/") and np.isscalar(value) and np.isfinite(value)}}
        curve.append(row)
        core.write_json(run_dir / "curve.json", curve)
        print(json.dumps({"run_id": run_id, "step": row["step"], "updates": row["optimizer_updates"],
                          "validation_gain_percent": {key: round(value["mean"], 6) for key, value in comparison.items()},
                          "failed": [key for key, passed in gates.items() if not passed]}, ensure_ascii=False), flush=True)

    class Checkpoints(BaseCallback):
        def __init__(self):
            super().__init__()
            self.next_step = args.block

        def _on_step(self):
            episode_ledger.record(model, self.locals["infos"])
            return True

        def _on_rollout_start(self):
            if self.num_timesteps >= self.next_step and self.num_timesteps < args.steps:
                checkpoint()
                self.next_step = (self.num_timesteps // args.block + 1) * args.block

        def _on_training_end(self):
            checkpoint()

    try:
        model.learn(args.steps, callback=Checkpoints(), progress_bar=False)
        checks = {"actual_optimizer_updates": model._v8_optimizer_step_calls > 0,
                  "weights_changed": core.weights_sha256(model) != initial_weights,
                  "source_hashes_unchanged": core.hashes_match(source_hashes),
                  "input_hashes_unchanged": core.hashes_match(input_hashes),
                  "all_policy_pointers_preserved": core.hashes_match(pointers),
                  "no_training_rendering": env.unwrapped.render_calls == 0}
        report = {"schema": "shore-bess-v8-flex-specialist-report.v1", "run_id": run_id,
                  "status": "FLEX_SPECIALIST_DIAGNOSTIC_ONLY", "generated_at": core.utc_now(),
                  "config": config, "manifest": manifest, "checks": checks,
                  "initial_weights_sha256": initial_weights, "final_weights_sha256": core.weights_sha256(model),
                  "total_environment_steps": int(model.num_timesteps),
                  "total_optimizer_updates": int(model._v8_optimizer_step_calls),
                  "training_episode_ledger": episode_ledger.description(),
                  "final_checkpoint": curve[-1], "validation_ranked_checkpoint": min(curve, key=core.rank),
                  "convergence": core.convergence(curve), "evaluations": {},
                  "promoted": False, "admitted": False, "simulation_mode": True,
                  "production_authority": False, "dispatch_allowed": False, "live_data_verified": False,
                  "claim_boundary": "One-seed, six-window validation diagnostic of neural flexible-load control with BESS held idle. No full Shore+BESS policy, no formal admission, no held-out evaluation or measured port savings.",
                  "evidence_files_sha256": {core.relative(path): core.file_sha256(path)
                    for path in sorted(run_dir.rglob("*")) if path.is_file() and "source" not in path.relative_to(run_dir).parts}}
        core.write_json(run_dir / "report.json", report)
        print(json.dumps({"report_path": core.relative(run_dir / "report.json"),
                          "report_sha256": core.file_sha256(run_dir / "report.json"), "checks": checks}), flush=True)
    finally:
        for hook in hooks:
            hook.remove()
        episode_ledger.close()
        env.close()


if __name__ == "__main__":
    main()
