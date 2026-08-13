# V3 technical-depth and business-coverage audit

This audit is the reviewer index for V3. It links claims to executable code, persisted evidence and explicit production boundaries. It does not convert public-data offline results into a terminal or group KPI.

## Review matrix

| Dimension | V3 implementation | Inspectable evidence | Remaining site gate |
|---|---|---|---|
| Public-data provenance | Shanghai official throughput anchors, Yangshan public weather/marine reanalysis, Los Angeles high-frequency public reference and Singapore long-horizon official aggregates | Dataset cards, source snapshots, SHA-256 metadata, `/api/v3/data-readiness` | Authorized TOS/VTS/EMS/PLC data |
| RL breadth | SAC, PPO, TD3, DQN, A2C, TQC, QR-DQN, TRPO, Recurrent PPO and ARS | Per-run config/model/manifest/optimizer history, clickable algorithm cards | Hyperparameter freeze and site replay |
| Control baselines | Receding-horizon MPC and neutral FCFS | Same blind-test windows and deterministic controller manifests | Operator-approved baseline definition |
| Causal twin | 37 observations, explicit availability masks and five bounded actions; service/allocation actions change operational electric load | `PortOperationsEnv`, `port_ops_v3` contract and causal regression test | Equipment-level calibration |
| Runtime decision chain | Continuous calibrated replay → fitted Ridge forecast → hash-verified saved SAC → shared projection/safety envelope | `/api/v3/runtime/status`, `/series`, runtime bundle checksums and clickable 3D source/model chips | Authorized measured telemetry and shadow-operation acceptance |
| Scenario coverage | Normal, high-density berthing, congestion, equipment degradation, heatwave/reefer, typhoon closure, island grid, tariff/carbon spike and telemetry loss/drift | `/api/v3/runtime/coverage`; clickable **场景覆盖** chip in the 3D console | Cyber/actuator fault coverage needs site gateway and hardware-in-the-loop |
| Twin reliability | Hash-bound policy inference, seven bounded stress replays and fail-closed telemetry-loss behavior; software evidence is explicitly separate from field truth | Clickable **孪生可靠性** view and `/api/v3/twin/reliability` | Authorized graph, measured outcomes, interval calibration, error decomposition and HIL |
| Evaluation | 70/10/20 chronological isolation, train-only normalization, validation-only algorithm selection, deterministic blind test, multi-seed bootstrap intervals | Advantage config/report, per-run validation artifacts, per-algorithm evidence API | Site backtest and shadow A/B protocol |
| Business value | Throughput, delay, cost, carbon, peak, unit intensity, completion, queues, storage cycling and projection dependence | Clickable value workbench and per-controller metric tables | Finance/carbon-ledger settlement inputs |
| Operations breadth | Twelve domains from arrival/berth through quay, yard, transport, gates, energy, reefer, maintenance, met-ocean, safety, carbon and multi-port drift | Clickable domain cards expose state, action, constraints and acceptance KPIs | Domain adapters and SOP acceptance |
| Safety | No production authority, action projection, SOC/power/ramp/terminal constraints, weather blocks, mapping/quality/calibration/shadow/approval gates | Clickable deployment gates, audit/model registry and rollback | Dual approval, execution receipts and drills |
| Deployment security | Role-separated long API keys, HTTPS-only CORS, per-key rate limit, request cap, request IDs, no-store and security headers | Clickable **部署自检**, `/health/ready`, middleware tests | TLS/secret-manager operations, gateway tests and credential exercises |
| Historical integrity | V1/V2 runs, portable bundles and fixed business benchmarks stay append-only; V3 exports optimizer/validation/model-hash evidence for clone-safe display | local registry plus `evidence/rl`, `evidence/v3`, V3 report assertions | None; release check prevents silent replacement |
| V3.2 value admission | Four non-yard-crane modules are re-audited under explicit cost/carbon/peak/service/safety gates; failed candidates remain visible instead of being promoted | `evidence/v3/value_improvement_v32.json`, candidate reports, 2026 forward challenge and per-module **查看V3.2增训结论** buttons | Public-offline value only; field commissioning still requires authorized data and acceptance |

## Reviewer click path

1. Open `/v3` and select all policy-value tabs, especially **安全稳健性**, **强基线对照**, **孪生可靠性** and **部署自检**.
2. Open any algorithm card with **查看训练指标** to inspect runs, seeds, optimizer steps, 95% intervals, job IDs and reward weights.
3. Open any port's **查看数据血缘** to inspect dataset hashes, evidence tiers, measured/derived/unavailable fields and original public links.
4. Open any operation card with **查看技术链路** to inspect inputs, decisions, hard constraints, site replacements, acceptance KPIs and repository code paths.
5. Open **验收规则** in the deployment flow to inspect required evidence, pass criteria and fail-closed action.
6. Open `/api/v3/overview` or a per-algorithm evidence endpoint to verify the same facts without the presentation layer.
7. On `/`, open the 3D twin and click **现在 / 未来六小时 / 策略作用后**. Source chips identify calibrated replay, Ridge and SAC; **场景覆盖** opens the executable/contract boundary matrix.
8. In **部署自检**, confirm that an open-source clone is runnable while production remains closed. Merely setting file paths cannot pass: graph, measured calibration and shadow acceptance are parsed, SHA-256 recorded and bound to one `site_id`.
9. On `/`, open Yard Lighting, HVAC, Shore+BESS and Site BESS in turn, then click **查看V3.2增训结论**. The panel shows the admitted or rejected decision, exact business metrics, training volume, evidence paths and claim boundary returned by the backend.

## Honest limits

- The five asset-specific admitted candidates are constraint-projected teacher-actor distillation policies. Their imitation loss and checkpoint validation reward are inspectable, but they are not PPO/SAC policy-gradient reward logs; environment-reward fine-tunes remain admission-gated. The ten `port_ops_v3` RL methods are a separate interactive-training evidence track.
- Shanghai throughput is official aggregate data; it is not terminal event telemetry.
- Yangshan environmental drivers are public reanalysis; terminal loads, tariffs, carbon factors, equipment and occupancy fields remain documented engineering assumptions.
- Los Angeles and Shanghai runs are independent reference/target training. V3 does not claim cross-port weight transfer.
- Annual values are mechanical scenario extrapolations. They are not Shanghai International Port Group savings, verified emission reductions, production A/B outcomes or financial audit results.
- Production control remains disabled until mapping, calibration, shadow operation, human approval and rollback acceptance pass.
- The continuous homepage feed is a time-warped, mass-conserving public-data replay. It is not Shanghai terminal telemetry, even though values update every second.
- The current-window business projection is model-driven and executable, but formal multi-seed blind-test evidence remains the KPI release basis.
