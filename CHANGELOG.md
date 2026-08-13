# Changelog

## 3.2.0 - 2026-08-13

- Upgraded Xiaoyi from a floating Q&A entry point to an evidence-bound mission copilot with six frontline tasks: situation, forecast, strategy explanation, alert triage, shift handoff and gated dry-run preparation.
- Fixed false-positive Xiaoyi availability: liveness now requires both `/health` and a real `POST /api/chat` contract. Every mission exposes whether Xiaoyi was actually called, the concrete generation provider/model, latency, context-grounding result and any guarded fallback.
- Added one hash-addressed runtime context assembled from calibrated replay, Ridge forecast, selected SAC model, anomaly/drift analysis, admission decision and explicit missing site factors. Xiaoyi cannot grant production authority or manufacture telemetry, work orders or savings.
- Added allow-listed natural-language routes, clickable cross-module actions and confirmation-gated shift-handoff persistence. Preview never writes an audit record; strategy dry-run remains blocked when the monitoring gate requires the FCFS/MPC safe baseline.
- Reworked the Copilot frontend around the original Xiaoyi Q-style asset with six mission cards, runtime evidence rail, invocation proof, action linkage and a visible handoff packet.
- Added a separately pinned January-May 2026 Shanghai forward challenge: four Ministry of Transport cumulative TEU publications plus 3,624 hourly Yangshan public reanalysis/model observations. It is explicitly not terminal telemetry and cannot be used for candidate tuning.
- Audited four non-yard-crane value-improvement tracks without rewriting V3.1 evidence. Yard lighting was retained at its lux-constrained ceiling; the HVAC 650 Pa candidate and Shore+BESS balanced PPO candidate were persisted but rejected by their preset gates.
- Ran 90,000 real PPO environment steps for the Shore+BESS balanced candidate. All three seeds failed the joint cost/carbon/peak non-regression gate, so the economic V3.1 champion and its carbon block remain visible.
- Added a Site BESS grid-only profile with zero DR and reserve revenue. Three new seeds were validation-selected under cost/carbon/peak non-regression, and all three passed the fresh 2026 forward challenge with zero guardrail violations.
- Added one clickable V3.2 evidence button to each affected module; the API exposes promoted and rejected candidate evidence, forward metrics, hashes and explicit claim boundaries.
- Added an explicit raw-action projection penalty for new `port_ops_v3` runs while preserving absent-penalty historical configs exactly.
- Retrained SAC with three seeds and 10,000 optimizer steps per seed. The blind-test action-projection mean fell from 73.61% to 50.28%; the maximum seed stayed below the tightened 60% gate, with zero guardrail violations and zero terminal-SOC error.
- Added per-step correction magnitude and root-cause attribution for grid capacity, SOC bounds, terminal-SOC reachability and power/ramp bounds.
- Preserved the V3.1 advantage report under `evidence/v3/history/` before writing the V3.2 report; failed and non-admitted pilot runs remain in the append-only run ledger.
- Added a clickable safety view exposing the V3.1→V3.2 comparison, correction severity, reason rates, three-seed confidence intervals and admission outcome.
- Added clickable twin-reliability and deployment-readiness views. Software stress coverage remains separate from site fidelity, calibration and hardware-in-the-loop acceptance.
- Hardened production APIs with HTTPS-only CORS, per-key sliding-window rate limits, request-body limits, no-store/security headers and explicit operator/admin separation.
- Replaced path-string readiness checks with parsed, hash-recorded graph/calibration/shadow evidence; approvals and one shared `site_id` must validate or production admission stays closed.
- Closed the remaining request-to-DOM and desktop-launch injection paths: Xiaoyi route labels now render through `textContent`, while Godot scenes and presets are code-owned allowlists with bounded timeouts and sanitized linkage metadata.
- Made dataset identifiers fail closed instead of normalizing unsafe input into possible name collisions, added traversal/launch-injection regression tests, and documented every root-contained path sink that CodeQL cannot infer through the project validators.

## 3.1.0 - 2026-08-12

- Added one dense reward-convergence panel to each of the five V3.1 asset tracks. A reproducible exporter reloads every saved checkpoint, performs deterministic replay on a fixed validation episode, and records the real environment reward in non-overlapping ten-step blocks (2,832 samples total). The UI shows raw three-seed fluctuation plus the per-checkpoint mean while explicitly marking `training_time_log=false`, `retrained_model=false`, `blind_test_access=false`, and no interpolation/noise.
- Added two non-interpolated training-process charts to all five asset tracks: per-seed optimizer imitation loss and fixed-validation reward. The UI reads 114 append-only checkpoints directly from `seed_*/metrics.jsonl`, draws three raw seed traces plus a same-epoch mean, exposes every file SHA-256, and records that no display retraining or random jitter occurred.
- Smoothed the Shore BESS and Site BESS evidence charts with shape-preserving interpolation over existing checkpoints only; original checkpoints remain visibly marked and no training artifact or metric is altered.
- Rebuilt HVAC as a V3.1 append-only evidence track with 5,760 chronological engineering-replay rows, a 30-state/three-action/12-constraint contract, three-seed convergence, sealed blind windows, hash-reloaded inference and visible cost/energy/peak/carbon/service gates; all 4,003 legacy records remain preserved.
- Rebuilt Yard Crane as a V3.1 append-only evidence track over 92,160 device rows, 8,559 TOS jobs and 69,120 queue forecasts: a 36-state/two-action/16-constraint contract, three-seed convergence, sealed blind testing, hash-reloaded inference and explicit work/SLA non-degradation. The old 0-byte policy and its 1,001 records remain preserved, as does the first V3.1 run that failed the peak gate.
- Rebuilt Yard Lighting as a public-signal-enriched V3.1 evidence track over 96 zones, 267,168 lighting records and 953,856 activity forecasts, aligned with the 17,544-hour Shanghai/Yangshan public replay. The 42-state/three-action/17-constraint policy now has three-seed convergence, sealed blind tests, hash-reloaded inference and zero under-lux zone-steps; the 498 legacy IQL records and OOD-blocked original policy remain unchanged.
- Rebuilt Shore+BESS as an append-only V3.1 evidence track: 34-state/two-action contract, eight reward terms, twelve hard constraints, 17,544-hour public timeline, three-seed fixed-validation convergence and 20-window blind testing. Legacy SAC artifacts remain unchanged; the UI now shows real selected-model inference, cost/peak gains and the carbon-gate failure separately.
- Rebuilt Site BESS as an append-only V3.1 evidence track: 40-state/two-action contract, nine reward terms, fifteen hard constraints, explicit engineering-only DR/reserve events, three-seed validation selection and chronological blind testing. The first peak-regression run remains recorded as failed; the passing successor reports real model inference, business metrics and site/settlement boundaries while preserving all legacy artifacts.
- Added a Shanghai target-domain dataset with 22 Ministry of Transport throughput anchors and 17,544 hourly Yangshan public reanalysis observations, retaining explicit measured/reanalysis/derived availability labels.
- Added chronological 70/10/20 train/validation/blind-test isolation without changing historical 80/20 run semantics.
- Expanded the executable controller registry to ten RL methods plus MPC and a neutral FCFS comparator.
- Added a version-pinned multi-seed RL-vs-FCFS advantage gate: validation selects the algorithm and the untouched blind test only reports the final result.
- Added the V3 decision center, multi-port data-readiness matrix, twelve-domain operations map and fail-closed public live feed.
- Added an explicit site-data replacement contract and kept production dispatch disabled until mapping, calibration, shadow operation and approval gates pass.
- Added `port_ops_v3` causal service/allocation-to-energy coupling so throughput gains cannot be treated as electrically free, while preserving all v1/v2 manifests.
- Added clickable per-algorithm KPI/CI evidence, per-domain state/action/constraint detail, and a separate paired-window business-value scenario with explicit non-site annualization boundaries.
- Added six clickable policy-value views covering relative advantage, absolute business outcomes, safety/robustness, equivalent-throughput value, MPC paired-window impact and the evidence protocol.
- Added persisted validation artifacts, optimizer traces, model hashes, three-port public-source lineage buttons, repository code pointers and detailed fail-closed gate acceptance drawers.
- Added a clone-safe portable V3 benchmark bundle and API fallback so formal metrics, optimizer checkpoints, validation evidence and historical V1/V2 indices remain visible without ignored local run state.
- Added a continuous, explicitly non-measured Shanghai public-data replay adapter with an 11-asset factor-conditioned, mass-conserving decomposition for the home and 3D views.
- Exported the validation-selected SAC policy as a hash-pinned clone-safe runtime artifact and connected 3D strategy curves to deterministic model inference, control projection and software safety checks.
- Bundled ECharts and added a zero-CDN perspective twin renderer, so first-clone offline reviews still show the 11-asset scene and model-backed charts.
- Connected the V3 hero to the public Yangshan model feed with a hash-pinned calibrated-replay fallback that remains explicitly non-measured.
- Replaced locally-shaped group curves and fallback devices with sums of backend asset series; current, forecast and strategy charts now trace to telemetry, Ridge or SAC output.
- Added a ten-class port-scenario coverage matrix, including fail-closed telemetry drift and a visible contract-only boundary for cyber/actuator faults.
- Added window-specific open-loop throughput, delay, cost/TEU and carbon/TEU projections using the same declared service and queue equations as `port_ops_v3`.
- Preserved ARS callback evidence without fabricating reward convergence when the optimizer backend only exposes progress and step checkpoints.

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
