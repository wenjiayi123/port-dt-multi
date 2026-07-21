# Model governance

This repository treats every RL policy and MPC controller as a versioned recommendation artifact, not as autonomous control authority.

## Lifecycle

1. A run records configuration, dataset SHA-256, chronological split, implementation, seed, runtime and artifact hash.
2. Held-out evaluation runs in a separate environment. At least 5 episodes and uncertainty intervals are required for promotion review.
3. RL comparative claims require at least 3 distinct seeds. Deterministic MPC has no learned initialization and runs once on the same fixed evaluation windows. Use `python -m scripts.rl_benchmark`.
4. `candidate`, `champion`, `rollback` and `archive` are registry aliases, not proof of deployment. Setting `champion` requires `PORT_DT_ALLOW_MODEL_PROMOTION=1`, an approver and a reason.
5. A previous champion becomes the rollback target. Every alias change is appended to `data/rl/model_registry_audit.jsonl`.
6. Site deployment remains a separate change-management process requiring calibrated constraints, system integration tests, operator approval and hardware interlocks.

## Automatic blocking criteria

- artifact checksum missing or mismatched;
- no chronological holdout result;
- fewer than 5 evaluated episodes or no uncertainty interval;
- fewer than 3 distinct evaluated seeds for a learned RL policy;
- guardrail violation rate above `PORT_DT_MAX_GUARDRAIL_VIOLATION_RATE`;
- failed dataset quality evidence;
- promotion opt-in disabled.

## Roles

- Contributor: implements algorithms and records reproducible evidence.
- Reviewer: checks data rights, leakage, metrics, safety limits and code changes.
- Site operator: owns site limits and accepts or rejects recommendations.
- Deployment approver: authorizes a separately controlled site release. Registry aliases alone do not grant this authority.

## Monitoring and retirement

Production adapters must monitor missing data, distribution drift, guardrail blocks, latency and outcome metrics. Disable recommendation serving on stale/invalid inputs, retain evidence, and revert to the approved non-RL operating procedure. Retire artifacts whose data rights, runtime support, calibration or safety evidence has expired.
