# V3 runtime data and model contract

V3 deliberately separates three dashboard states while keeping the same asset
IDs and timestamps:

| UI state | Backend source | What it proves | What it does not prove |
|---|---|---|---|
| Now | `CalibratedReplayTelemetry` | A continuous, mass-conserving replay pipeline over the checked-in Shanghai public benchmark | It is not measured terminal telemetry |
| Forecast | fitted Ridge autoregression | The displayed P10/P50/P90 values are produced from telemetry history by executable model code | It is not a port-certified forecasting SLA |
| Strategy | exported SAC seed 42 | Canonical state enters a hash-verified saved policy, then control projection and software-envelope checks | It has no actuator authority and is not a field KPI |

The runtime bundle is under `evidence/v3/runtime/`. `runtime_model.sha256`
binds the model, training configuration and metadata. The metadata also binds
the dataset SHA-256 and the validation-only selection protocol. A clone fails
closed if any of these hashes differ.

## Temporary continuous source

The public dataset has hourly aggregate rows. The default adapter advances one
public-data minute per wall-clock second and linearly interpolates adjacent
rows. It decomposes aggregate power across 11 declared port asset classes with
factor-conditioned weights and then renormalizes them, so every timestamp
satisfies:

`sum(asset kW) == public aggregate base_load_kw`

The response always carries `measured=false`,
`mode=calibrated_public_replay_simulator`, the artifact hash, and the adapter
replacement contract. No local fallback devices or timer-only business curves
are created if the adapter fails.

The V3 hero first requests the current public Shanghai/Yangshan model feed. If
that external request is unavailable, it reads `public_conditions` from
`GET /api/v3/runtime/frame` and labels the values **calibrated continuous replay,
not measured**. The fallback therefore remains tied to the checked-in public
snapshot and its SHA-256; it never substitutes random weather or a timer curve.

Dashboard charts use the repository-bundled ECharts runtime. The port scene
also includes a repository-native perspective Canvas renderer whose asset
positions and heat colours consume the same `/api/assets` and curve responses.
This removes the visual layer's public-CDN dependency while keeping the existing
WebGL path compatible if a host supplies it.

## Online decision projection

The six-hour 3D strategy view calls the saved SAC policy with deterministic
inference. The same declared service-capacity, allocation, resource,
weather-block and queue equations used by `PortOperationsEnv-v3` produce an
open-loop business projection for the current calibrated window. It reports
throughput, delay, cost/TEU and carbon/TEU alongside total power. This is a
window-specific diagnostic; formal multi-seed blind-test evidence under
`evidence/v3/` remains the release KPI evidence.

## Scenario coverage and site boundary

`GET /api/v3/runtime/coverage` exposes ten operational classes: normal work,
high-density berthing, channel congestion, equipment degradation, heatwave and
reefer pressure, typhoon closure, island grid, tariff/carbon spike, telemetry
loss/drift, and cyber/actuator faults. Nine are executable offline or fail
closed in software. Cyber and actuator faults remain contract-only until an
authorized site adapter and hardware-in-the-loop acceptance are available.

Therefore the matrix is coverage evidence, not a claim that every real event
has been observed. Production replacement requires authorized TOS/EMS/PLC/BMS
fields, port-specific calibration, shadow operation, operator approval,
rollback drills and independent hardware interlocks.
