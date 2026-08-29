# V3 technical evidence map

Version 3.2.0 extends the existing evidence chain; it does not delete or reinterpret V1/V2/V3.1 training metrics. The V3.1 advantage bundle is archived before V3.2 selection evidence is written.

The strong-baseline artifact `evidence/v3/strong_baseline_evidence_v3.*` performs paired blind-window comparisons against FCFS, a fixed engineering operations-rule proxy and MPC. Failure to beat a stronger comparator is displayed as a failed claim gate; the engineering rule is explicitly not a measured incumbent policy.

## What is executable

- Ten RL implementations: SAC, PPO, TD3, DQN, A2C, TQC, QR-DQN, TRPO, Recurrent PPO and ARS.
- Two non-learning comparators: receding-horizon MPC and a neutral FCFS policy.
- MPC emits terminal-SOC-feasible BESS commands before the shared safety projector; the UI still reports projection dependence instead of hiding it.
- `port_ops_v3`: the 37-observation/five-action contract from v2 plus explicit causal coupling from service intensity, berth priority and yard flow to operational electric load. V1/V2 remain loadable for historical reproduction.
- Chronological 70/10/20 train/validation/blind-test isolation for V3 datasets; historical 80/20 manifests remain readable.
- Backend-owned optimizer history, model hash, dataset hash, seed, runtime and test-only trajectory artifacts.
- A clone-safe runtime policy bundle loads the validation-selected SAC model only after model/config/dataset SHA-256 verification, then consumes canonical states from the telemetry adapter with deterministic inference.
- The default 3D chain uses continuous calibrated public replay for **Now**, fitted Ridge P10/P50/P90 for **Forecast**, and the hash-verified SAC plus shared projection/safety checks for **Strategy**. No asset or group curve is synthesized as a frontend fallback.
- A version-pinned weighted RL-vs-FCFS gate: validation rows select the algorithm, while three seeds, at least 10,000 optimizer steps and ten untouched blind-test windows report the final result. Safety requires zero hard-guardrail violations, terminal-SOC restoration and a bounded projection rate.
- A dedicated Shore+BESS V3.1 track preserves the rejected legacy SAC evidence, trains a hash-addressed constrained actor with 34 states, two advisory actions, eight reward terms and twelve hard constraints, and selects each of three seeds without reading the 3,509-row blind split.
- A separate Site BESS V3.1 track preserves the 2,000-step/8,927-transition saturated SAC history, then trains a 40-state/two-action event-aware CMDP actor with nine reward components and fifteen hard constraints. DR/reserve coverage is an explicit engineering calendar with zero observed settlement rows; it cannot be described as live market evidence.
- The HVAC V3.1 track preserves 4,003 legacy records and replaces the UI-only KPI path with 5,760 chronological 15-minute engineering rows, a 30-state/three-action/eight-reward/twelve-constraint actor, three validation-selected seeds, eight sealed blind windows and hash-reloaded model inference.
- The Yard Crane V3.1 track retains 1,001 legacy records, the original empty policy diagnosis and the first V3.1 peak-gate failure. Its passing successor uses 92,160 device rows, 8,559 jobs and 69,120 queue forecasts with a 36-state/two-action/nine-reward/sixteen-constraint contract and explicit moves/SLA non-degradation gates.
- The Yard Lighting V3.1 track retains 498 legacy IQL records and the legacy OOD block. It aligns 267,168 zone records and 953,856 activity forecasts with checked-in public weather/occupancy/load/tariff/carbon signals, then evaluates a 42-state/three-action/ten-reward/seventeen-constraint actor with minimum and critical lux hard gates.
- All five tracks expose a separate dense checkpoint-reward replay artifact. Every saved checkpoint is hash-addressed and replayed deterministically on the same fixed validation episode; real environment rewards are aggregated every ten steps and compared with the same seed's epoch-1 block. This gives 2,832 inspectable reward samples without retraining or opening blind-test rows, and is explicitly not represented as a historical optimizer-step reward log.

## Multi-port strategy

| Package | Evidence density | V3 role |
|---|---|---|
| `public_us_la_6min_v1` | 87,459 six-minute steps; 262,347 independent public observations | High-frequency reference training and cross-port comparison; no weight-transfer claim |
| `public_port_ops_v1` | 52,608 hourly steps; 144 official aggregate anchors | Long-horizon scenario robustness and historical business benchmark |
| `public_cn_sha_hourly_v3` | 17,544 hourly public reanalysis observations + 22 official throughput anchors | Shanghai adaptation and untouched blind test |

More rows do not imply more independent information. Each dataset reports independent source observations separately from expanded driver rows.

## V3.2 value-improvement admission

V3.2 applies the same rule to every non-yard-crane module: a candidate is promoted only when its persisted model passes the predeclared business and safety gates. A higher cost or carbon number cannot conceal a peak, service or safety regression. Rejected candidates remain append-only evidence and the V3.1 champion pointer is unchanged.

| Module | V3.2 action | Admission result | Business conclusion |
|---|---|---|---|
| Yard Lighting | Audited the selected actor against the fixed-validation safe teacher | Retain V3.1 | The validation cost gap is only 0.000027 percentage points; further dimming would increase lux-projection dependence rather than create defensible value |
| HVAC | Trained three 650 Pa low-load-static-pressure candidates, 6,912 samples and 2,160 updates per seed | Reject candidate | Cost/energy/carbon improved, but peak reduction was 1.9728% and missed the predeclared 2.0% gate by 0.0272 percentage points |
| Shore+BESS | Warm-started and ran balanced PPO for 30,000 environment steps on each of three seeds | Reject candidate | No seed made cost, carbon and peak simultaneously non-regressing; the economic champion remains available but its carbon profile stays blocked |
| Site BESS | Trained a no-DR/no-reserve-revenue grid-only profile and evaluated it after selection on a 2026 forward challenge | Admit public-offline profile | Three of three seeds reduced cost, carbon and peak with zero market revenue, zero guardrail violations and terminal-SOC restoration; annual values remain engineering extrapolations |

The fresh challenge package `public_cn_sha_forward_2026m05_v1` contains 3,624 hourly rows from January through May 2026. Its four independent official Shanghai throughput reporting periods are combined with 3,624 public reanalysis hours. The dataset explicitly forbids candidate selection and contains no columns labelled as measured terminal telemetry.

## Evidence paths

| Evidence | Path |
|---|---|
| Historical RL benchmark registry | `data/rl/benchmarks.json` |
| Historical portable bundles | `evidence/rl/` |
| V3 version-pinned selection/evaluation protocol | `config/v3_advantage_benchmark.json` |
| V3 advantage JSON/Markdown/checksum | `evidence/v3/shanghai_public_advantage_v3.*` |
| Clone-safe V3 run/optimizer/validation bundle | `evidence/v3/public_cn_sha_hourly_v3_benchmark.*` |
| Paired MPC-vs-FCFS value scenario | `evidence/v3/shanghai_public_business_impact_v3.*` |
| Per-run configuration, optimizer history, manifest and evaluation | `data/rl/runs/<job_id>/` |
| Per-run validation-only selection artifact | `data/rl/runs/<job_id>/validation_evaluation.json` |
| Current capabilities and evidence inventory | `GET /api/v3/overview` |
| Per-controller training metrics | `GET /api/v3/algorithms/{algorithm_id}/evidence` |
| Per-domain state/action/constraint depth | `GET /api/v3/capabilities/{capability_id}` |
| Dataset readiness and replacement requirements | `GET /api/v3/data-readiness` |
| Runtime model, source and authority state | `GET /api/v3/runtime/status` |
| Current or stress-scenario policy inference | `GET /api/v3/runtime/series` |
| Executable scenario coverage and site-only gaps | `GET /api/v3/runtime/coverage` |
| Twin software-stress/site-fidelity separation | `GET /api/v3/twin/reliability` |
| Open-source and production-site readiness gates | `GET /health/ready` |
| Runtime model/config/checksum bundle | `evidence/v3/runtime/` |
| Shore+BESS append-only run index and selected models | `evidence/v3/shore_bess/` |
| Shore+BESS convergence, blind business metrics and real model inference | `GET /api/v3/modules/shore-bess/evidence` |
| Site BESS append-only failed/passing runs and selected models | `evidence/v3/bess_energy/` |
| Site BESS convergence, event coverage, blind metrics and real model inference | `GET /api/v3/modules/bess-energy/evidence` |
| HVAC append-only runs, selected models, manifests and checksums | `evidence/v3/hvac/` |
| HVAC convergence, blind business metrics, contract and model inference | `GET /api/v3/modules/hvac/evidence` |
| Yard Crane append-only failed/passing runs, models and checksums | `evidence/v3/yard_crane/` |
| Yard Crane convergence, workload/SLA metrics and model inference | `GET /api/v3/modules/yard-crane/evidence` |
| Yard Lighting append-only runs, public-signal linkage and models | `evidence/v3/yard_lighting/` |
| Yard Lighting convergence, lux gates, business metrics and inference | `GET /api/v3/modules/yard-lighting/evidence` |
| V3.2 cross-module admission decisions and exact deltas | `evidence/v3/value_improvement_v32.json` |
| 2026 forward challenge metadata and source boundary | `data/rl/datasets/public_cn_sha_forward_2026m05_v1.meta.json` |
| Site BESS grid-only selected profile and forward evidence | `evidence/v3/bess_energy/latest_grid_only.json` |

## Claim ladder

1. `RL_SMOKE_WIRING_ONLY`: confirms the optimizer and environment connect; never used for performance claims.
2. `RL_HELD_OUT_EVALUATION`: at least 10,000 optimizer steps and evaluation on a time-separated holdout.
3. `POINT_ESTIMATE_ADVANTAGE_NOT_95CI_CONFIRMED`: the version-pinned composite mean is positive but its 95% bootstrap interval is not strictly above zero.
4. `STRICT_ADVANTAGE_95CI`: the version-pinned composite 95% interval is strictly above zero with all safety and comparability gates passing.
5. `TEST_SAFETY_ADMISSION_FAILED`: validation selected the policy but the untouched blind test crossed a safety gate, so no advantage claim is admitted and no alternate policy is chosen from the blind test.
6. Field KPI: unavailable until authorized site data, calibration, shadow operation and port acceptance exist.

The V3 UI intentionally displays the exact claim status returned by evidence files. It cannot promote a point estimate into a strict or field claim.

Local `data/rl/runs/` and `data/rl/benchmarks.json` remain ignored runtime state. The release UI falls back to checked-in portable bundles containing formal metrics, uncertainty, optimizer checkpoints, validation artifacts and model hashes, so a fresh clone does not become an evidence-empty demo.

The clickable evidence workbench separates learned-policy efficiency, deterministic-controller cost/carbon scenarios and evaluation protocol. A composite winner never hides a negative component metric. Cost and carbon annualization is explicitly mechanical and remains ineligible for a group or site claim while tariff, load and emissions inputs are engineering assumptions.

The current Shore+BESS economic profile is a concrete example of that rule: it passes convergence, cost, peak and safety gates, but fails the carbon-reduction gate because modeled emissions increase. The checked-in report labels the economic and carbon profile admissions separately; production authority remains false until shore meters, BMS, PCC demand, tariffs, emissions, interlocks and gateway acknowledgements are replaced with authorized site inputs.

The same rule is applied to Site BESS: the first formal run was retained as failed because its saved actors produced a small blind-test peak regression. A hard no-new-demand-peak projection was then added and a second append-only three-seed run passed convergence, cost, peak, carbon, event-compliance and safety gates. The scenario still has no production authority, because public Shanghai inputs do not contain authorized DR/reserve settlement, PCS/BMS telemetry or write acknowledgements.

HVAC, Yard Crane and Yard Lighting follow the same boundary. Their selected reports pass their predeclared convergence, business and safety gates, and the frontend reads those reports plus actual selected-model inference. HVAC reports 2.698% cost reduction with 100% cooling satisfaction; Yard Crane reports 3.148% cost reduction with 100% moves retention and SLA non-degradation; Yard Lighting reports 1.770% cost reduction with 100% minimum/critical lux compliance. These are chronological blind-test results inside checked-in public-signal or engineering-replay environments—not measurements at a deployed terminal.

No module silently converts missing field data into a production claim. HVAC still requires authorized BMS/BA and plant/terminal sensors; Yard Crane requires TOS jobs, PLC power/thermal data and acknowledgements; Yard Lighting requires zone lux meters, gateway state and actuation receipts. Until those adapters, calibration, shadow-mode acceptance and rollback drills exist, all three remain recommendation-only with `claim_eligible=false` and `production_authority=false`.

## Training-process visibility

Every asset-specific evidence endpoint exposes `training_process`, sourced directly from the selected run's append-only `seed_*/metrics.jsonl` files. The contract carries the epoch, optimizer-update count, actor imitation loss, fixed-validation mean reward, timestamp, source path and SHA-256 for each seed. Across the five modules there are 114 persisted checkpoints. The first two UI charts draw the three raw seed traces and an arithmetic same-epoch mean without interpolation or injected noise; the response explicitly sets `retrained_for_display=false`, `interpolated_points=false` and `frontend_random_noise=false`. Aggregate validation-cost, variability, service and peak charts remain separate from these process traces.
