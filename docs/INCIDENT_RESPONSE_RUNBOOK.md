# Incident response runbook

## Trigger conditions

Treat authentication bypass, leaked credentials, corrupted data, artifact hash mismatch, unsafe recommendation, unexplained drift, actuator disagreement or unavailable rollback as an incident.

## Immediate containment

1. Disable southbound execution and model promotion. Preserve operator control and the approved non-RL procedure.
2. Remove the affected API key or adapter credential from the secret manager; do not paste it into issues or logs.
3. Record UTC start time, request IDs, model job ID/alias, dataset hash, site adapter version and affected assets.
4. If a model is implicated, move serving to the last reviewed rollback artifact only after checking its hash and site approval. If uncertain, stop recommendation serving.
5. Preserve logs, registry audit JSONL, configuration and relevant telemetry read-only. Do not overwrite evidence.

## Triage

- Check `/health/ready`, `/metrics`, request IDs and adapter provenance.
- Re-run dataset quality and verify model/dataset SHA-256.
- Reproduce with the exact configuration on the chronological holdout; never use production actuation for reproduction.
- Determine whether the event is data, model, software, infrastructure, credential or operating-procedure related.

## Recovery

Recovery requires an identified root cause, tested fix, validated rollback path, rotated secrets when relevant, site-owner approval and a monitored restart. Re-enable one boundary at a time; keep automatic actuation disabled unless a separate site safety case permits it.

## Post-incident

Within the organization’s required window, document impact, timeline, evidence, cause, corrective action, regression test and owner. For security issues, follow the private disclosure process in `SECURITY.md` before public release.
