# Dataset card: public_port_ops_v1

## Summary

`public_port_ops_v1` is a deterministic integration dataset with 2,137 hourly rows from 2026-01-01 through 2026-03-31 UTC. It exists to exercise the data adapter, chronological split, RL environment and reproducible tests. It is not a production telemetry export and is not a scientifically coherent single-port benchmark.

The checked-in CSV SHA-256 is `ddc3d2b2e2a091864da02673f30e411bcced561e6fe0b9ff44e55e28f09bff92`.

## Source and geographic limitation

- Monthly container throughput comes from the Maritime and Port Authority of Singapore dataset published through data.gov.sg.
- Hourly tide prediction comes from NOAA CO-OPS station 9414290 in San Francisco.
- These geographically different inputs are deliberately combined only as an integration fixture. Results must not be described as Singapore, San Francisco, or any other port's measured operating performance.

The Singapore source is reused under the [Singapore Open Data Licence](https://data.gov.sg/open-data-licence). Contains information from the MPA container-throughput dataset, accessed 2026-07-20 from data.gov.sg, made available under that licence. The NOAA input is public information; NOAA asks for attribution and disclaims fitness and accuracy. See the [NOAA Tides & Currents disclaimer](https://tidesandcurrents.noaa.gov/disclaimers.html).

No source agency endorses this project.

## Fields

| Field | Unit | Origin |
|---|---:|---|
| `timestamp` | UTC ISO-8601 | Fixed hourly window |
| `base_load_kw` | kW | Deterministic engineering derivative |
| `throughput_teu` | TEU/hour | Deterministic disaggregation of public monthly throughput |
| `vessel_arrivals` | count/hour | Deterministic engineering derivative |
| `tide_m` | m, MLLW | NOAA hourly prediction |
| `price_per_kwh` | currency/kWh | Deterministic time-of-use fixture; not a tariff |
| `carbon_kg_per_kwh` | kgCO2e/kWh | Deterministic fixture; not an official emissions factor |
| `ambient_c` | degree Celsius | Deterministic fixture; not observed weather |

## Quality and split

The API exposes completeness, cadence, physical bounds, outliers, constants, units and governance metadata at `GET /api/rl/datasets/public_port_ops_v1/quality`. Training uses the first 80% by time; evaluation uses the final 20%, without shuffle. The exact split is recorded in each run manifest.

## Appropriate uses

- software integration and contract tests;
- reproducible algorithm execution checks;
- demonstrations of provenance, quality gates and train/test isolation;
- developing a real port CSV adapter.

## Inappropriate uses

- claims about a real port's safety, savings, emissions or throughput;
- autonomous equipment control;
- geographic or causal conclusions;
- public leaderboard claims without a suitable benchmark dataset and multiple seeds.

## Regeneration

Run `python -m scripts.fetch_public_port_dataset`. The source window and transformation formulas are fixed in the script, while `created_at` records the regeneration time. Review upstream licence terms before redistributing a regenerated version.
