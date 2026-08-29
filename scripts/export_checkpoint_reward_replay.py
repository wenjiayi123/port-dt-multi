"""Export dense, honest reward evidence from already-saved V3 checkpoints.

This script does not train a policy and does not open the chronological blind-test
split.  It deterministically replays every persisted checkpoint on the first
fixed validation episode, aggregates the environment reward every ten steps,
and records the delta against the same seed's epoch-1 checkpoint.  The delta
removes most scenario seasonality while retaining the raw reward for audit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Callable

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SAMPLE_EVERY_STEPS = 10


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def checkpoint_epoch(path: Path) -> int:
    match = re.search(r"checkpoint_epoch_(\d+)", path.name)
    if not match:
        raise ValueError(f"cannot parse checkpoint epoch: {path}")
    return int(match.group(1))


def metric_index(seed_dir: Path) -> dict[int, dict[str, Any]]:
    result: dict[int, dict[str, Any]] = {}
    for line in (seed_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            result[int(row["epoch"])] = row
    return result


def reward_components(contract: Any) -> list[str]:
    if isinstance(contract, dict):
        return list(contract.get("reward_terms") or contract.get("reward_components") or [])
    return list(contract.reward_components)


def build_module(module: str) -> dict[str, Any]:
    if module == "hvac":
        from app.services.rl_model.hvac_cooling.v3_environment import (
            CONTRACT, HVACV3Env, NumpyMLPPolicy, chronological_slices,
            load_config, load_dataset,
        )

        config = load_config()
        dataset = load_dataset(config)
        train_slice, validation_slice, _blind_slice = chronological_slices(dataset)
        steps = int(config["training"]["episode_steps"])
        factory = lambda seed: HVACV3Env(
            dataset, validation_slice, config=config, normalization_slice=train_slice,
            episode_steps=steps, seed=seed, training=False, record_trace=False,
        )
        loader = lambda path: NumpyMLPPolicy.load(path)
        policy = lambda model, observation: model.predict(observation)
        suffix = ".json"
    elif module == "yard_crane":
        from app.services.rl_model.yard_crane.v3_environment import (
            CONTRACT, NumpyMLPPolicy, YardCraneV3Env, chronological_slices,
            load_config, load_dataset,
        )

        config = load_config()
        dataset = load_dataset(config)
        train_slice, validation_slice, _blind_slice = chronological_slices(dataset)
        steps = int(config["training"]["episode_steps"])
        factory = lambda seed: YardCraneV3Env(
            dataset, validation_slice, config=config, normalization_slice=train_slice,
            episode_steps=steps, seed=seed, training=False, record_trace=False,
        )
        loader = lambda path: NumpyMLPPolicy.load(path)
        policy = lambda model, observation: model.predict(observation)
        suffix = ".json"
    elif module == "yard_lighting":
        from app.services.rl_model.yard_lighting.v3_environment import (
            CONTRACT, NumpyMLPPolicy, YardLightingV3Env, chronological_slices,
            load_config, load_dataset,
        )

        config = load_config()
        dataset = load_dataset(config)
        train_slice, validation_slice, _blind_slice = chronological_slices(dataset)
        steps = int(config["training"]["episode_steps"])
        factory = lambda seed: YardLightingV3Env(
            dataset, validation_slice, config=config, normalization_slice=train_slice,
            episode_steps=steps, seed=seed, training=False, record_trace=False,
        )
        loader = lambda path: NumpyMLPPolicy.load(path)
        policy = lambda model, observation: model.predict(observation)
        suffix = ".json"
    elif module == "shore_bess":
        from stable_baselines3 import PPO
        from app.services.rl_model.shore_bess.v3_environment import (
            CONTRACT, ShoreBESSEnv, chronological_slices, load_config,
            load_public_dataset,
        )

        config = load_config()
        dataset = load_public_dataset(config)
        train_slice, validation_slice, _blind_slice = chronological_slices(dataset)
        steps = int(config["training"]["episode_hours"])
        factory = lambda seed: ShoreBESSEnv(
            dataset, validation_slice, config=config, normalization_slice=train_slice,
            episode_steps=steps, seed=seed, training=False, record_trace=False,
        )
        loader = lambda path: PPO.load(str(path), device="cpu")
        policy = lambda model, observation: np.asarray(
            model.predict(observation, deterministic=True)[0], dtype=np.float32
        )
        suffix = ".zip"
    elif module == "bess_energy":
        from stable_baselines3 import PPO
        from app.services.rl_model.bess_energy.v3_environment import (
            BESSEnergyV3Env, CONTRACT, chronological_slices, load_config,
            load_public_dataset,
        )

        config = load_config()
        dataset = load_public_dataset(config)
        train_slice, validation_slice, _blind_slice = chronological_slices(dataset)
        steps = int(config["training"]["episode_hours"])
        factory = lambda seed: BESSEnergyV3Env(
            dataset, validation_slice, config=config, normalization_slice=train_slice,
            episode_steps=steps, seed=seed, training=False, record_trace=False,
        )
        loader = lambda path: PPO.load(str(path), device="cpu")
        policy = lambda model, observation: np.asarray(
            model.predict(observation, deterministic=True)[0], dtype=np.float32
        )
        suffix = ".zip"
    else:
        raise ValueError(f"unsupported module: {module}")

    return {
        "factory": factory,
        "loader": loader,
        "policy": policy,
        "suffix": suffix,
        "episode_steps": steps,
        "reward_components": reward_components(CONTRACT),
        "validation_rows": validation_slice.stop - validation_slice.start,
    }


def replay_blocks(
    factory: Callable[[int], Any], model: Any, policy: Callable[[Any, Any], Any], seed: int
) -> list[float]:
    env = factory(seed)
    observation, _ = env.reset(seed=seed, options={"start_index": 0})
    rewards: list[float] = []
    done = False
    while not done:
        action = policy(model, observation)
        observation, reward, terminated, truncated, _info = env.step(action)
        value = float(reward)
        if not math.isfinite(value):
            raise ValueError("non-finite reward during checkpoint replay")
        rewards.append(value)
        done = bool(terminated or truncated)
    if env.render_calls:
        raise RuntimeError("checkpoint reward replay rendered unexpectedly")
    env.close()
    return [
        float(np.mean(rewards[start:start + SAMPLE_EVERY_STEPS]))
        for start in range(0, len(rewards), SAMPLE_EVERY_STEPS)
    ]


def export_module(module: str, *, force: bool) -> Path:
    evidence_root = ROOT / "evidence" / "v3" / module
    latest = load_json(evidence_root / "latest.json")
    report_path = ROOT / str(latest["report_path"])
    run_dir = report_path.parent
    output_path = run_dir / "checkpoint_reward_replay.json"
    if output_path.exists() and not force:
        return output_path

    adapter = build_module(module)
    all_series: list[dict[str, Any]] = []
    total_samples = 0
    for seed_dir in sorted(run_dir.glob("seed_*")):
        seed = int(seed_dir.name.removeprefix("seed_"))
        metrics = metric_index(seed_dir)
        checkpoints = sorted(seed_dir.glob(f"checkpoint_epoch_*{adapter['suffix']}"), key=checkpoint_epoch)
        if not checkpoints or checkpoint_epoch(checkpoints[0]) != 1:
            raise RuntimeError(f"{module} seed {seed} is missing epoch-1 checkpoint")
        baseline: list[float] | None = None
        points: list[dict[str, Any]] = []
        summaries: list[dict[str, Any]] = []
        for checkpoint in checkpoints:
            epoch = checkpoint_epoch(checkpoint)
            model = adapter["loader"](checkpoint)
            blocks = replay_blocks(adapter["factory"], model, adapter["policy"], seed)
            if baseline is None:
                baseline = blocks
            if len(blocks) != len(baseline):
                raise RuntimeError(f"reward block count changed for {checkpoint}")
            deltas = [value - base for value, base in zip(blocks, baseline)]
            update_count = int(metrics[epoch]["optimizer_updates"])
            for index, (value, delta) in enumerate(zip(blocks, deltas), start=1):
                points.append({
                    "epoch": epoch,
                    "optimizer_updates": update_count,
                    "sample_index": index,
                    "environment_step_end": min(index * SAMPLE_EVERY_STEPS, adapter["episode_steps"]),
                    "reward_block_mean": value,
                    "reward_delta_from_epoch1": delta,
                })
            summaries.append({
                "epoch": epoch,
                "optimizer_updates": update_count,
                "checkpoint_path": str(checkpoint.relative_to(ROOT)),
                "checkpoint_sha256": sha256(checkpoint),
                "reward_block_count": len(blocks),
                "reward_mean": float(np.mean(blocks)),
                "reward_delta_from_epoch1_mean": float(np.mean(deltas)),
                "reward_delta_from_epoch1_std": float(np.std(deltas)),
            })
        total_samples += len(points)
        all_series.append({
            "seed": seed,
            "records": len(points),
            "checkpoint_summaries": summaries,
            "points": points,
        })

    payload = {
        "schema": "port-dt-checkpoint-reward-replay.v1",
        "module": module,
        "run_id": latest["run_id"],
        "generated_at": latest.get("updated_at"),
        "source": "deterministic_post_training_checkpoint_replay",
        "split": "fixed_validation_only",
        "validation_window_start": 0,
        "validation_rows_available": adapter["validation_rows"],
        "episode_environment_steps": adapter["episode_steps"],
        "sample_every_environment_steps": SAMPLE_EVERY_STEPS,
        "sample_aggregation": "mean_environment_reward_for_each_non_overlapping_10_step_block",
        "comparison": "same_seed_same_validation_block_delta_from_epoch_1_checkpoint",
        "higher_is_better": True,
        "retrained_model": False,
        "training_time_log": False,
        "blind_test_access": False,
        "render_calls": 0,
        "frontend_interpolation": False,
        "frontend_random_noise": False,
        "reward_components": adapter["reward_components"],
        "total_reward_samples": total_samples,
        "series": all_series,
        "claim_boundary": (
            "Dense post-training replay of persisted checkpoints for convergence inspection; "
            "it is not an optimizer-step reward log and is not blind-test business evidence."
        ),
    }
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--module",
        choices=["all", "hvac", "yard_crane", "yard_lighting", "shore_bess", "bess_energy"],
        default="all",
    )
    parser.add_argument("--force", action="store_true", help="replace an existing derived replay artifact")
    args = parser.parse_args()
    modules = ["hvac", "yard_crane", "yard_lighting", "shore_bess", "bess_energy"] if args.module == "all" else [args.module]
    for module in modules:
        path = export_module(module, force=args.force)
        print(json.dumps({"module": module, "path": str(path.relative_to(ROOT)), "sha256": sha256(path)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
