# Production site readiness checklist

The default repository is open-source research software. `/health/ready` intentionally reports `production_site_ready: false` until site evidence is configured and verified. Merely setting a path is not enough: the graph, measured calibration and shadow-acceptance JSON files are parsed, SHA-256 recorded, approval fields checked and all three records must bind to the same `site_id`.

- [ ] Authorized port dataset mapped, quality-gated, time-synchronized and privacy/licence reviewed.
- [ ] DTDL-compatible entity graph configured with source timestamps and no generated assets.
- [ ] Twin calibration evidence passes site thresholds on a separate validation window.
- [ ] At least 3 seeds and 5+ held-out episodes per policy; confidence intervals and guardrail rate reviewed.
- [ ] A measured current-operations incumbent is replayed on the same shadow windows; acceptance evidence records 0 guardrail violations and named approval.
- [ ] Model artifact hash verified; model card reviewed; champion and rollback approved.
- [ ] Site-specific SoC, power, ramp, demand, equipment and operational constraints calibrated.
- [ ] Independent PLC/BMS interlocks and manual override tested; recommendations fail closed.
- [ ] Actuator whitelist, route and per-action parameter constraints reviewed; separate requester/confirmer and second-channel secret tested.
- [ ] Duplicate command, partial failure, lost acknowledgement and failed-then-retried rollback drills completed against the site gateway.
- [ ] Production API keys, separate administrator key, HTTPS-only CORS, TLS reverse proxy, secret manager and least privilege enabled.
- [ ] Per-key rate limit, request-body limit, no-store API responses and security headers verified at the ingress and application layers.
- [ ] Telemetry freshness, drift, latency, errors and safety blocks monitored with alert ownership.
- [ ] Backup, restore, rollback, credential rotation and incident exercises completed.
- [ ] TOS/AIS/weather/tariff contracts, quotas, paging, time zones and failure behavior tested.
- [ ] Cybersecurity, electrical, operational, legal and data-governance owners sign off.

Passing the software checks is necessary but not sufficient for port deployment.

Required private evidence variables are `PORT_DT_TWIN_GRAPH_PATH`, `PORT_DT_TWIN_CALIBRATION_PATH` and `PORT_DT_SHADOW_ACCEPTANCE_PATH`. Deployment configuration also requires `PORT_DT_ENV=production`, `PORT_DT_TLS_TERMINATION_ATTESTED=true`, `PORT_DT_SECRET_MANAGER_ATTESTED=true`, operator/admin keys and an HTTPS CORS allowlist. The repository does not ship a fabricated passing site file.
