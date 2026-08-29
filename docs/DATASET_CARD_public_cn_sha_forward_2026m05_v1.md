# Dataset card: `public_cn_sha_forward_2026m05_v1`

## Purpose

This is a separately pinned temporal forward challenge for V3.2 candidate admission. Candidate architectures, objectives and checkpoints must be selected on the 2024–2025 training/validation data before this artifact is opened. It is not Shanghai terminal telemetry and is not eligible for production or group-savings claims.

## Time and integrity

| Item | Value |
|---|---:|
| Period | 2026-01-01 00:00 UTC — 2026-05-31 23:00 UTC |
| Cadence | 1 hour |
| Rows | 3,624 |
| Official reporting periods | 4 |
| Public reanalysis/model hours | 3,624 |
| Independent source observations | 3,628 |
| Dataset SHA-256 | `616fe7cde24695f0d19118c64d1e5c534f9adee47a886b33b6003e7e372bb06a` |
| Reanalysis snapshot SHA-256 | `472cfc1d70ac8c4e41a7d434919928d1407c417bcbffee25ece17fb80ef13883` |
| Role | Forward challenge only; candidate selection prohibited |

The four official Shanghai cumulative container-throughput values are 941, 1,411, 1,896 and 2,375 ten-thousand TEU for January-February through January-May. The deterministic hourly allocation conserves the final cumulative 23,750,000 TEU exactly. January and February remain one official period; no separate January observation is invented.

## Sources

- Shanghai container throughput: official Ministry of Transport workbooks. Publication URLs, attachment URLs and attachment SHA-256 values are pinned in [`data/public_sources/shanghai_port_mot_2026_forward.json`](../data/public_sources/shanghai_port_mot_2026_forward.json).
- Weather: [Open-Meteo historical weather API](https://open-meteo.com/en/docs/historical-weather-api), ERA5 hourly reanalysis near Yangshan.
- Marine context: [Open-Meteo marine API](https://open-meteo.com/en/docs/marine-weather-api), public model aggregation near Yangshan.

## Provenance boundary

| Class | Columns | Interpretation |
|---|---|---|
| Official aggregate | Reporting-period `throughput_teu` totals | Shanghai port-level aggregate |
| Public reanalysis/model | `ambient_c`, wind, wave, tide, current | Gridded environmental context, not a terminal sensor |
| Engineering derivative | Hourly throughput allocation, load, tariff, carbon, arrivals, occupancy, equipment availability, congestion and closure | Reproducible scenario only |
| Unavailable | Visibility, terminal meters, control actions, site tariff and site carbon intensity | Must be replaced before site validation |

## Admission use

- One-time forward comparison after validation-only checkpoint selection.
- Multi-seed non-regression gates for cost, carbon, peak and safety.
- Public offline engineering evidence only; all monetary annualization keeps `claim_eligible=false`.

## Prohibited use

- Repeatedly tuning candidates against this window and still calling it untouched.
- Claiming measured Shanghai terminal savings, audited financial value or verified carbon reduction.
- Production dispatch, navigation or equipment safety certification.

Regenerate and pin the upstream snapshot:

```bash
PYTHONPATH=. .venv312/bin/python scripts/fetch_shanghai_forward_2026.py --refresh
```
