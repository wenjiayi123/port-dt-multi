# V3 site-data replacement contract

V3 uses a stable canonical state/action/safety contract so a port deployment replaces data adapters and calibration parameters instead of rewriting the learning and evidence chain.

## Required production domains

| Domain | Required site fields | Public V3 status | Admission condition |
|---|---|---|---|
| Vessel / berth | vessel and voyage IDs, ETA/ATA/ETB/ATB/ETD/ATD, berth plan, pilot/tug state | Aggregate/derived | TOS + VTS/AIS reconciliation and clock audit |
| Quay crane | QC ID, task, move timestamps, productivity, outage and fault state | Derived availability | PLC/TOS event mapping and missing-event analysis |
| Yard | container position, yard block, dwell, rehandle, YC task and availability | Derived occupancy | Container genealogy and position consistency gate |
| Horizontal transport | AGV/IGV/truck position, mission, queue, battery/energy, road state | Unavailable | Fleet adapter, map version and command/receipt correlation |
| Gate / rail / barge | appointment, arrival, service, departure and capacity | Unavailable | Schedule/actual reconciliation and identity mapping |
| Energy | meter, tariff, BMS, transformer, shore power, reefer and DER state | Derived | Calibrated meter hierarchy and settlement boundary |
| Weather / marine | port weather stations, tide/current/wave, VTS closure and reopening events | Public reanalysis | Site observations, time alignment and safety-owner approval |
| Maintenance / safety | alarms, work orders, inspections, incidents, overrides and interlocks | Unavailable | Taxonomy mapping, severity ownership and audit retention |

Every field is mapped to:

```text
source_system + source_field + canonical_field + unit + timezone
+ entity_key + event_time + ingest_time + quality_flag + availability_mask
+ licence/authority + retention + owner + SHA-256 manifest
```

Unknown mandatory factors fail closed. The adapter must not silently fall back to an engineering derivative when running in site mode.

## Execution-depth disclosure

Each V3 business-domain card exposes an execution class rather than treating every covered domain as an optimizer:

- `model_backed_*`: a persisted model/controller currently produces offline recommendations.
- `executable_sandbox`: code and runtime endpoints execute, but inputs are engineering replay rather than site telemetry.
- `executable_safety_guard`, `executable_governance_workflow`, `executable_evidence_calculator`, `executable_transfer_guard`: deterministic safety, workflow, evidence or transfer logic executes, but is not a learned business optimizer.
- `simulation_contract_only`, `coupled_factor_contract_only`, `monitoring_only`: the domain is represented in state/simulation/monitoring contracts and explicitly has no independent optimization output.

Every card includes callable endpoints, decision source, current data mode, repository artifact SHA-256, missing site fields and fail-closed fallback. `production_ready=false` remains mandatory for all open-source public-data cards.

## Deployment sequence

1. Freeze source manifests, schemas, owners, units and time semantics.
2. Run historical backfill quality gates and compare site distributions with public reference-training data.
3. Calibrate twin parameters only on the training window; keep validation and acceptance windows untouched.
4. Replay decisions offline with policy, FCFS, MPC and operational baselines on identical windows.
5. Run shadow mode without dispatch authority; monitor drift, missingness, constraint violations and operator disagreement.
6. Conduct limited-scope assisted operation with four-eyes approval, allowlists, interlocks and rollback drills.
7. Grant production authority only through the port's safety and change-management process.

## Minimum acceptance gates

- No unresolved identity or timestamp collision on safety-critical events.
- Declared completeness, latency and drift thresholds pass per source.
- Zero software guardrail violations on the acceptance benchmark.
- Policy advantage is evaluated against a version-pinned neutral baseline and operational incumbent; algorithm selection uses validation rows, never the final acceptance rows.
- The recommendation can be rejected, overridden and rolled back with complete receipts.
- Public-data metrics remain labelled offline evidence and are never relabelled as field KPIs.
