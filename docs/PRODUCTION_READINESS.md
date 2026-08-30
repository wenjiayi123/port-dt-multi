# Production site readiness checklist

The default repository is open-source research software. `/health/ready` intentionally reports `production_site_ready: false` until the full technical chain, thirty-day continuity evidence and four-week operating-model evidence are configured, semantically reproduced and bound to the same `site_id`.

- [ ] Authorized port dataset mapped, quality-gated, time-synchronized and privacy/licence reviewed.
- [ ] All eight `port-snapshot.v1` adapters bind the authorized `site_id`, owner and source system; HMAC, payload digest, freshness, units, sequence and replay checks pass without storing secrets or silently substituting public data.
- [ ] The replacement training dataset passes `/api/rl/datasets/{dataset_id}/site-readiness`: 720+ gap-free hours, 99% coverage of every V6 input, source-manifest hash and per-field measured/authorized-derived lineage.
- [ ] DTDL-compatible entity graph configured with source timestamps and no generated assets.
- [ ] Twin calibration evidence passes site thresholds on a separate validation window.
- [ ] At least 3 seeds and 5+ held-out episodes per policy; confidence intervals and guardrail rate reviewed.
- [ ] A measured current-operations incumbent is bound to at least 35 read-only shadow cycles across 7+ consecutive operational days; each cycle records the candidate receipt, data quality, feasibility, latency, guardrail outcome and operator disposition.
- [ ] Shadow acceptance records 0 guardrail violations, fixed non-degradation and service thresholds, a completed rollback-drill reference, and distinct operations/safety reviewers who are independent of the data owner.
- [ ] Model artifact hash verified; model card reviewed; champion and rollback approved.
- [ ] Site-specific SoC, power, ramp, demand, equipment and operational constraints calibrated.
- [ ] Independent PLC/BMS interlocks and manual override tested; recommendations fail closed.
- [ ] Actuator whitelist, route and per-action parameter constraints reviewed; separate requester/confirmer and second-channel secret tested.
- [ ] The actuator configuration passes `site_actuator_config.v2`, uses an exact asset route, sets a 15–300 second command expiry, and requires verified readback, independent interlock and rollback.
- [ ] Duplicate command, partial failure, lost acknowledgement and failed-then-retried rollback drills completed against the site gateway.
- [ ] All eight fixed execution-commissioning scenarios pass; unsafe-command execution count is zero; readback and interlock coverage are complete; operations, maritime-safety and controls-engineering reviewers are independent.
- [ ] Shipping line, vessel agent, terminal, pilotage, towage and port authority identities are authorized and bind the same collaboration revision to the standard port-call event digest.
- [ ] Measured delays propagate through the arrival-to-departure dependency chain; berth, pilot and tug conflicts are recorded before and after the recommendation-only replan; post-replan conflicts and unresolved objections are zero.
- [ ] Every required participant acknowledges the proposed revision, and terminal-operations and port-authority reviewers are independent of each other and of the source owner. The evidence does not itself mutate a shared plan or grant authority to change an estimated arrival time.
- [ ] Vessel arrival, berth duration, quay-crane productivity, yard congestion, equipment failure, weather stoppage, regulatory delay and energy-load forecasts all carry pre-issue feature snapshots and later measured outcomes.
- [ ] Training, interval calibration and final test windows are chronological and non-overlapping; all eight targets pass fixed error, interval coverage, interval width and drift gates; binary risk targets additionally pass probability calibration gates.
- [ ] Operations-planning, model-risk and maritime-safety reviewers are independent of each other and of the source owner. Forecasts remain advisory and cannot automatically commit resources or dispatch equipment.
- [ ] Every coordination chain covers arrival, channel, berthing, cargo, yard transfer, yard operation, gate, rail and maintenance, and the combined plan exercises all eleven fixed channel, berth, marine-service, terminal, hinterland, energy and maintenance resource types.
- [ ] The thirty-minute freeze horizon is unchanged; every dependency retains a fifteen-minute handoff; all tasks are scheduled before deadline; candidate capacity conflicts, safety incidents and unresolved exceptions are zero; no task moves more than one hundred eighty minutes.
- [ ] At least ten authorized site chains bind current-plan receipts, forecast receipts and the five upstream evidence digests. Integrated-planning, marine-services, terminal-operations and equipment-energy reviewers are mutually independent and bind the same change ticket.
- [ ] Eight core components have gap-free hourly records for at least 720 hours and independently pass availability, error-rate, latency, freshness and audit-delivery objectives.
- [ ] Every daily backup is immutable, encrypted and restore-tested; process restart, zone failover, database restore, queue recovery, credential rotation and safe-degradation drills pass fixed recovery-time and recovery-point limits.
- [ ] Every incident binds ordered detection-through-closure timestamps, commander, work order, root cause and postmortem; every change binds approval, canary, health and rollback receipts.
- [ ] Twelve responsibility domains and ten critical workflows bind named directory identities. Requester, approver, executor and verifier are distinct, and every workflow has at least two reviewers.
- [ ] Duty manager, terminal operations, maritime safety, site reliability and cybersecurity have distinct primary/backup staff, current competencies and handover receipts across three shifts for at least 28 days; all four escalation levels bind fixed response targets and three distinct escalation people.
- [ ] Production API keys, separate administrator key, HTTPS-only CORS, TLS reverse proxy, secret manager and least privilege enabled.
- [ ] Per-key rate limit, request-body limit, no-store API responses and security headers verified at the ingress and application layers.
- [ ] Telemetry freshness, drift, latency, errors and safety blocks monitored with alert ownership.
- [ ] Backup, restore, rollback, credential rotation and incident exercises completed.
- [ ] TOS/AIS/weather/tariff contracts, quotas, paging, time zones and failure behavior tested.
- [ ] Cybersecurity, electrical, operational, legal and data-governance owners sign off.

Passing the software checks is necessary but not sufficient for port deployment.

The read-only adapter and historical replacement contracts are documented in [SITE_INTEGRATION_GATEWAY.md](SITE_INTEGRATION_GATEWAY.md). A valid signed snapshot or a structurally complete site dataset is an input-admission result only; neither can enable dispatch or production control.

The read-only shadow contract is documented in [SITE_SHADOW_ACCEPTANCE.md](SITE_SHADOW_ACCEPTANCE.md). Candidate energy, throughput and delay during this stage are counterfactual projections, not measured production benefits.

The execution commissioning and interlock contract is documented in [SITE_EXECUTION_ACCEPTANCE.md](SITE_EXECUTION_ACCEPTANCE.md). Passing the browser contract never creates a work order, reads a control credential, dispatches a command or grants site authority.

The shared port-call timeline and delay-propagation contract is documented in [PORT_CALL_COLLABORATION.md](PORT_CALL_COLLABORATION.md). It is an internal interoperability and assurance profile, not a claim of external standards certification. Browser calculations never write to a terminal operating system, vessel-traffic service, pilot or tug booking system.

The standards mapping and bound external-report gate is documented in [MARITIME_INTEROPERABILITY.md](MARITIME_INTEROPERABILITY.md). Internal mapping covers the fixed DCSA Port Call, IMO Maritime Single Window and IHO S-100 profiles, while authority submission, navigational use, official certification claims and dispatch remain disabled.

The eight-target forecast and uncertainty gate is documented in [FORECAST_UNCERTAINTY.md](FORECAST_UNCERTAINTY.md). It requires pre-issued features, later measured outcomes, three chronological windows and empirical interval coverage. A browser contract result is not a live forecast service-level claim, and forecasts remain advisory even after evidence acceptance.

The realized business-benefit attribution gate is documented in [BUSINESS_BENEFIT_ATTRIBUTION.md](BUSINESS_BENEFIT_ATTRIBUTION.md). It requires a pre-registered design, complete candidate/incumbent pre/post pairs, actual execution and human-approval receipts, later metered outcomes, a concurrent comparator, fixed uncertainty and safety gates, and three independent reviewers. Existing offline KPI benchmarks remain offline counterfactuals and are never relabelled as field benefits.

The cross-resource rolling-plan gate is documented in [END_TO_END_COORDINATION.md](END_TO_END_COORDINATION.md). It requires one authorized current plan covering eleven resource types and nine operation stages, fixed freeze and handoff rules, deterministic semantic reproduction, at least ten chains and four independent reviewers. Passing evidence remains recommendation-only and never mutates the shared plan, commits a resource or grants production authority.

The continuous-operations gate is documented in [PRODUCTION_CONTINUITY.md](PRODUCTION_CONTINUITY.md). It requires 720 gap-free hourly observations across eight components, fixed service objectives, daily restore tests, complete incident closure, canary/rollback receipts and six resilience drills. It never grants automatic failover or production-control authority.

The named operating-model gate is documented in [OPERATING_MODEL_GOVERNANCE.md](OPERATING_MODEL_GOVERNANCE.md). It requires twelve responsibility domains, ten segregated workflows, current competencies, 28 days of three-shift primary/backup coverage and four-level escalation. It validates external appointments and authority evidence but cannot appoint people, create accounts or self-grant control.

Required private evidence variables include the full technical evidence chain plus `PORT_DT_PRODUCTION_CONTINUITY_PATH` and `PORT_DT_OPERATING_MODEL_GOVERNANCE_PATH`. Deployment configuration also requires `PORT_DT_ENV=production`, `PORT_DT_TLS_TERMINATION_ATTESTED=true`, `PORT_DT_SECRET_MANAGER_ATTESTED=true`, operator/admin keys and an HTTPS CORS allowlist. The repository does not ship a fabricated passing site file.
