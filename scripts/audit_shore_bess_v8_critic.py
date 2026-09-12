"""TRAIN-only critic action-ranking diagnostic; never modifies a learner."""
from __future__ import annotations

import argparse
import copy
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import SAC, TD3

from app.services.rl_model.shore_bess.v8_environment import ShoreBESSV8SACEnv
from app.services.rl_training.datasets import file_sha256, load_port_dataset
from scripts.train_shore_bess_v8 import load_config, CONFIG_PATH, ROOT, month_slice


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--output")
    args = parser.parse_args()
    torch.set_num_threads(1)
    directory = Path(args.run)
    cfg = json.loads((directory / "config.json").read_text())
    manifest = json.loads((directory / "manifest.json").read_text())
    for path, expected in manifest["source_sha256"].items():
        if file_sha256(ROOT / path) != expected:
            raise ValueError("current source/config differs from training: " + path)
    seed = cfg["seeds"][0]
    curve = json.loads((directory / f"seed_{seed}/curve.json").read_text())
    selected = curve[-1]
    model_path = ROOT / selected["model_path"]
    if file_sha256(model_path) != selected["model_sha256"]:
        raise ValueError("checkpoint bytes differ from completed checkpoint row")
    actor_cls = {"stable_baselines3.SAC": SAC, "stable_baselines3.TD3": TD3}[cfg["algorithm"]]
    model = actor_cls.load(model_path, device="cpu")
    physical = cfg["physical_config"]
    dataset = load_port_dataset(physical["dataset_id"])
    if file_sha256(dataset.path) != manifest["dataset_sha256"]:
        raise ValueError("current dataset differs from the trained checkpoint")
    train = month_slice(dataset, "2024-01-01", "2025-05-01")
    env = ShoreBESSV8SACEnv(dataset, train, config=physical, normalization_slice=train,
        episode_steps=168, carbon_price=cfg["carbon_price_cny_per_kg_constraint_multiplier"], training=False)
    rows = []
    for start in (168 * 10, 168 * 30, 168 * 50):
        obs, _ = env.reset(options={"start_index": start})
        for step in range(97):
            action, _ = model.predict(obs, deterministic=True)
            if step in (0, 48, 96):
                alternatives = []
                for bess in sorted(set((-.8, -.4, -.15, 0., .15, .4, .8, float(action[0])))):
                    test_action = np.asarray([bess, action[1]], dtype=np.float32)
                    tensor_obs = torch.as_tensor(obs[None], device=model.device)
                    tensor_action = torch.as_tensor(test_action[None], device=model.device)
                    with torch.no_grad():
                        predictions = model.critic(tensor_obs, tensor_action)
                        q = float(torch.cat(predictions, dim=1).min().cpu())
                    branch = copy.deepcopy(env)
                    next_obs, reward, done, _, info = branch.step(test_action)
                    immediate = float(reward)
                    total = immediate
                    while not done:
                        next_action, _ = model.predict(next_obs, deterministic=True)
                        next_obs, reward, done, _, _ = branch.step(next_action)
                        total += float(reward)
                    alternatives.append({"latent_bess": bess, "latent_flex": float(action[1]),
                        "executed_bess_kw": info["final_action"]["bess_kw"],
                        "critic_min_q": q, "observed_immediate_reward": immediate,
                        "deterministic_remaining_return": total})
                    branch.close()
                critic_best = max(alternatives, key=lambda x: x["critic_min_q"])
                rollout_best = max(alternatives, key=lambda x: x["deterministic_remaining_return"])
                actual = next(x for x in alternatives if x["latent_bess"] == float(action[0]))
                regret = rollout_best["deterministic_remaining_return"] - critic_best["deterministic_remaining_return"]
                rows.append({"train_start": start, "step": step, "timestamp": dataset.timestamps[start + step],
                    "neural_action": action.tolist(), "alternatives": alternatives,
                    "critic_best_latent_bess": critic_best["latent_bess"],
                    "rollout_best_latent_bess": rollout_best["latent_bess"],
                    "critic_grid_choice_return_regret": regret,
                    "actor_return_gap_to_tested_grid": rollout_best["deterministic_remaining_return"] - actual["deterministic_remaining_return"],
                    "critic_grid_choice_suboptimal_above_tolerance": regret > 1e-6})
            obs, _, _, _, _ = env.step(action)
    env.close()
    output = Path(args.output) if args.output else ROOT / "evidence/v8/shore_bess/audits" / ("critic-ranking-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + ".json")
    output.parent.mkdir(parents=True, exist_ok=True)
    result = {"schema": "port-shore-bess-v8-critic-ranking.v2", "run": str(directory),
        "model_path": str(model_path.relative_to(ROOT)), "model_sha256": selected["model_sha256"],
        "step": selected["step"], "dataset_sha256": dataset.fingerprint,
        "source_sha256": {str(Path(__file__).resolve().relative_to(ROOT)): file_sha256(Path(__file__))},
        "training_only": True, "test_accessed": False, "forward_accessed": False,
        "interpretation": "Only nine training states and a finite BESS grid with flex fixed; includes the actor action. Different latent values can map to identical physical actions. Regret uses a 1e-6 return tie tolerance. Not a global critic quality or optimality claim. SAC critic includes entropy whereas this deterministic continuation does not; this is not an unbiased soft-Q calibration or a policy admission test.",
        "frozen_training_source_sha256": manifest["source_sha256"],
        "grid_choices_with_positive_return_regret": sum(row["critic_grid_choice_suboptimal_above_tolerance"] for row in rows),
        "grid_top1_disagreements": sum(row["critic_best_latent_bess"] != row["rollout_best_latent_bess"] for row in rows),
        "rows": rows}
    with output.open("x") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"output": str(output), "step": selected["step"], "grid_top1_disagreements": result["grid_top1_disagreements"], "states": len(rows)}), flush=True)


if __name__ == "__main__":
    main()
