"""Run persisted multi-seed holdout benchmarks for the five controllers."""

from __future__ import annotations

import argparse
import json
import time

from app.services.rl_training.trainer import ALGORITHMS, TRAINING_MANAGER


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
    parser.add_argument("--episode-steps", type=int, default=48)
    parser.add_argument("--timeout", type=int, default=7200)
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

    completed = []
    for algorithm in algorithms:
        run_seeds = seeds if ALGORITHMS[algorithm].trainable else seeds[:1]
        for seed in run_seeds:
            started = TRAINING_MANAGER.start({
                "algorithm": algorithm,
                "dataset_id": args.dataset,
                "total_steps": args.steps,
                "episode_steps": args.episode_steps,
                "seed": seed,
                "test_ratio": 0.2,
            })
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
