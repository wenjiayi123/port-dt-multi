"""Run persisted multi-seed blind-holdout benchmarks for registered controllers."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from app.services.rl_training.trainer import ALGORITHMS, TRAINING_MANAGER


ROOT = Path(__file__).resolve().parents[1]
BUSINESS_PROFILES_PATH = ROOT / "config/rl_business_profiles_v3.json"


def wait(job_id: str, timeout_seconds: int) -> dict:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        status = TRAINING_MANAGER.status(job_id)
        if status.get("status") in {"COMPLETED", "FAILED", "CANCELLED", "INTERRUPTED"}:
            return status
        time.sleep(0.5)
    raise TimeoutError(f"training timed out: {job_id}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run real multi-seed RL/MPC benchmarks")
    parser.add_argument("--dataset", default="public_port_ops_v1")
    parser.add_argument("--algorithms", default=",".join(ALGORITHMS), help="comma-separated algorithm ids")
    parser.add_argument("--seeds", default="42,142,242", help="at least three distinct integers for comparative claims")
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--episode-hours", type=float, default=48.0)
    parser.add_argument("--episode-steps", type=int, default=0, help="explicit override; 0 derives steps from dataset cadence")
    parser.add_argument("--validation-ratio", type=float, default=None, help="0 preserves historical 80/20; v3 datasets default to their declared three-way split")
    parser.add_argument("--timeout", type=int, default=7200)
    parser.add_argument(
        "--environment-version",
        choices=("port_ops_v1", "port_ops_v2", "port_ops_v3"),
        default="",
        help="explicit environment contract; omitted uses the dataset declaration",
    )
    parser.add_argument(
        "--business-profile",
        default="",
        help="optional profile id from config/rl_business_profiles_v3.json",
    )
    args = parser.parse_args()
    algorithms = [item.strip().lower() for item in args.algorithms.split(",") if item.strip()]
    unknown = [item for item in algorithms if item not in ALGORITHMS]
    if unknown:
        parser.error("unknown algorithms: " + ", ".join(unknown))
    try:
        seeds = sorted({int(item.strip()) for item in args.seeds.split(",") if item.strip()})
    except ValueError:
        parser.error("--seeds must contain comma-separated integers")
    if len(seeds) < 3:
        parser.error("at least three distinct seeds are required")
    if args.episodes < 5:
        parser.error("--episodes must be at least 5")
    if args.steps < 64:
        parser.error("--steps must be at least 64")
    business_profile = None
    if args.business_profile:
        payload = json.loads(BUSINESS_PROFILES_PATH.read_text(encoding="utf-8"))
        business_profile = (payload.get("profiles") or {}).get(args.business_profile)
        if not isinstance(business_profile, dict):
            parser.error(f"unknown business profile: {args.business_profile}")

    completed = []
    for algorithm in algorithms:
        run_seeds = seeds if ALGORITHMS[algorithm].trainable else seeds[:1]
        for seed in run_seeds:
            config = {
                "algorithm": algorithm,
                "dataset_id": args.dataset,
                "total_steps": args.steps,
                "episode_hours": args.episode_hours,
                "seed": seed,
                "test_ratio": 0.2,
            }
            if args.validation_ratio is not None:
                config["validation_ratio"] = args.validation_ratio
            if args.episode_steps > 0:
                config["episode_steps"] = args.episode_steps
            if args.environment_version:
                config["environment_version"] = args.environment_version
            if business_profile is not None:
                config["business_profile_id"] = args.business_profile
                config["reward_weights"] = business_profile["reward_weights"]
            started = TRAINING_MANAGER.start(config)
            status = wait(started["job_id"], args.timeout)
            if status.get("status") != "COMPLETED":
                raise RuntimeError(f"{algorithm} seed={seed} failed: {status.get('error') or status.get('stage')}")
            evaluation = TRAINING_MANAGER.evaluate(started["job_id"], args.episodes)
            completed.append({
                "job_id": started["job_id"],
                "algorithm": algorithm,
                "seed": seed,
                "episodes": evaluation["episodes"],
                "metrics": evaluation["metrics"],
                "uncertainty": evaluation["uncertainty"],
            })
            print(json.dumps(completed[-1], ensure_ascii=False), flush=True)
    print(json.dumps({"completed": completed, "summary": TRAINING_MANAGER.benchmark_summary(args.dataset)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
