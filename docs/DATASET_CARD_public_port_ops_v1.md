# Dataset card: public_port_ops_v1

## Summary

`public_port_ops_v1` contains 52,608 contiguous hourly driver rows from 2020-01-01
through 2025-12-31 UTC. It exercises the port adapter, RL environment, chronological
split and reproducible business benchmark. It is not terminal telemetry.

Checked-in CSV SHA-256:
`14ad4422ecea6fae33cb8d715c5e7cbff0ff7863b9d0d24d3c5459e8df4f65b6`.

## Public anchors and transformation

- MPA Singapore monthly container throughput, 2020–2025.
- MPA Singapore monthly container-vessel arrivals, 2020–2025.
- Both official inputs use the same port geography and Singapore Open Data Licence.
- Hourly throughput and arrivals are deterministic normalized allocations of monthly totals.
- Base load, time-of-use price, carbon factor, temperature and harmonic tide stress are
  deterministic engineering derivatives. Tide has zero coefficient in the three business KPIs.

The resulting file is geographically coherent in its official anchors but is not a measured
single-port hourly series. No publisher endorses this project.

## Fields

| Field | Unit | Origin |
|---|---:|---|
| `timestamp` | UTC ISO-8601 | continuous hourly index |
| `base_load_kw` | kW | engineering derivative |
| `throughput_teu` | TEU/hour | allocation of official monthly throughput |
| `vessel_arrivals` | count/hour | allocation of official monthly container-vessel arrivals |
| `tide_m` | m | declared harmonic stress; excluded from KPI calculation |
| `price_per_kwh` | currency/kWh | deterministic scenario tariff |
| `carbon_kg_per_kwh` | kgCO2e/kWh | deterministic scenario factor |
| `ambient_c` | °C | deterministic scenario temperature |

## Split and appropriate use

The business benchmark uses 2020–2023 train, 2024 validation and 2025 test. RL runs use a
chronological train/test split recorded in each manifest and fit normalizers on train only.
Appropriate uses are integration tests, algorithm execution, provenance/split checks and adapter
development. Do not use it to claim measured port savings, safety, emissions or autonomous
equipment control.

Regenerate with `python -m scripts.fetch_public_port_dataset`; the script pins source dataset
IDs and transformation formulas.
