# Dataset card: `public_us_la_6min_v1`

## Intended use

This dataset is a reproducible, high-frequency public benchmark for offline
reinforcement-learning wiring, factor-mask validation, chronological evaluation,
and port-adapter development. It is not a terminal telemetry export and cannot
support claims about measured terminal savings or autonomous control readiness.

## Scope and size

- Region: Port of Los Angeles and the nearby NOAA coastal observation network.
- Period: calendar year 2021, the latest complete twelve-month series in the
  BTS source table used by the builder.
- Cadence: six minutes.
- Rows after strict timestamp alignment and gap filtering: 87,459.
- Environment: `port_ops_v2`.
- Port profile: `us_la_public_benchmark_v2`.
- Independent raw public observations: 262,347.
- Short-gap interpolations: 21 air-temperature and 21 wind values; zero
  water-level values.

## Public measured and official inputs

| Field | Publisher | Dataset/station | Treatment |
|---|---|---|---|
| Monthly TEU | U.S. DOT/BTS | `rd72-aq8r` | Official monthly aggregate; deterministically distributed to six-minute driver rows |
| Water level | NOAA CO-OPS | Los Angeles `9410660` | Verified six-minute observation, MLLW datum |
| Air temperature | NOAA CO-OPS | Santa Monica `9410840` | Six-minute coastal observation |
| Wind speed | NOAA CO-OPS | Santa Monica `9410840` | Six-minute coastal observation; retained as an optional factor |

The metadata records exact raw-observation counts, short-gap interpolation
counts, source URLs, access date, row hash, measured columns, derived columns,
and unavailable factors. Interpolated values are not counted as independent
source observations.

## Engineering-derived fields

Base electrical load, sub-monthly throughput allocation, vessel arrivals,
tariff, grid carbon intensity, berth occupancy, yard occupancy, channel
congestion, and reefer load are deterministic engineering derivatives. They are
inputs for reproducibility, not observations of a terminal.

Visibility, waves, currents, crane availability, equipment availability,
pilot/tug availability, and closure state remain unavailable. `port_ops_v2`
encodes those factors as neutral values plus explicit zero availability masks;
it does not silently manufacture values.

## Rebuild

```bash
source .venv/bin/activate
python -m scripts.fetch_public_la_benchmark
```

The builder fails if the BTS monthly series is incomplete or NOAA water/air
coverage falls below the declared thresholds. Linear interpolation is limited
to ten consecutive six-minute steps and is disclosed per source field.

## Limitations and field deployment

Replacing this benchmark with a production scene requires an approved port
profile, terminal-level TOS/AIS/equipment/energy exports, explicit field
mapping, time alignment, source permissions, quality approval, site safety
limits, shadow operation, operator acceptance, and hardware interlocks.
