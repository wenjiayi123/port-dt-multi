# Production site readiness checklist

The default repository is open-source research software. `/health/ready` intentionally reports `production_site_ready: false` until site evidence is configured.

- [ ] Authorized port dataset mapped, quality-gated, time-synchronized and privacy/licence reviewed.
- [ ] DTDL-compatible entity graph configured with source timestamps and no generated assets.
- [ ] Twin calibration evidence passes site thresholds on a separate validation window.
- [ ] At least 3 seeds and 5+ held-out episodes per policy; confidence intervals and guardrail rate reviewed.
- [ ] Model artifact hash verified; model card reviewed; champion and rollback approved.
- [ ] Site-specific SoC, power, ramp, demand, equipment and operational constraints calibrated.
- [ ] Independent PLC/BMS interlocks and manual override tested; recommendations fail closed.
- [ ] Actuator whitelist, route and per-action parameter constraints reviewed; separate requester/confirmer and second-channel secret tested.
- [ ] Duplicate command, partial failure, lost acknowledgement and failed-then-retried rollback drills completed against the site gateway.
- [ ] Production API keys, restricted CORS, TLS reverse proxy, secret manager and least privilege enabled.
- [ ] Telemetry freshness, drift, latency, errors and safety blocks monitored with alert ownership.
- [ ] Backup, restore, rollback, credential rotation and incident exercises completed.
- [ ] TOS/AIS/weather/tariff contracts, quotas, paging, time zones and failure behavior tested.
- [ ] Cybersecurity, electrical, operational, legal and data-governance owners sign off.

Passing the software checks is necessary but not sufficient for port deployment.
