# Changelog

## 3.0.1 - 2026-07-21

- Added strict run-identifier validation and symlink-aware path containment for training, evaluation and model-registry artifacts.
- Rebuilt the public-source package without local environments, generated runs, legacy datasets, software-copyright materials or workstation remnants.
- Expanded the bilingual architecture, data, model-governance, release and contributor documentation.
- Pinned GitHub Actions by commit, added public-only security workflows and reduced private-preview automation noise.
- Updated the FastAPI/Pydantic/ASGI stack to current security-supported releases.

## 3.0.0 - 2026-07-20

- Replaced display-only RL behavior with real SAC, PPO, TD3, DQN and MPC execution.
- Enforced chronological train/holdout separation and prohibited training rendering.
- Added public-data regeneration, dataset quality gates, provenance and a dataset card.
- Added repeated-window uncertainty, multi-seed benchmarking, model cards, registry aliases and rollback audit.
- Added inference safety-envelope assessment and shared action projection.
- Added DTDL-compatible twin models, entity-graph validation and calibration-evidence gates.
- Added API hardening, health/readiness, request tracing, metrics, container packaging and GitHub supply-chain workflows.
- Disabled unverified legacy simulators and fail-open production behavior by default.
