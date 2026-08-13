# Dataset card: `public_cn_sha_hourly_v3`

## Purpose

This dataset is the V3 Shanghai target-domain adaptation and chronological blind-test package. It is designed for public-data offline research and adapter validation. It is not Shanghai terminal telemetry and cannot support a measured site-efficiency claim.

## Time and integrity

| Item | Value |
|---|---:|
| Period | 2024-01-01 00:00 UTC — 2025-12-31 23:00 UTC |
| Cadence | 1 hour |
| Rows | 17,544 |
| Official reporting anchors | 22 |
| Public reanalysis hours | 17,544 |
| Dataset SHA-256 | `803214ea0202abde241f75a28d7bf46b9c7ad801d40605a0916ec14ef7906a01` |
| Reanalysis snapshot SHA-256 | `3bfecc268f830be4f69fab85a4c27a09adec14ad15a6e83bd12c52963ceb14a7` |
| Split | 12,280 train / 1,755 validation / 3,509 blind test |

The official 2024–2025 aggregate allocation is conserved exactly at 106,570,000 TEU. January–February remains a two-month official period; the generator does not invent separate official monthly values.

## Sources

- Container throughput: 22 cumulative reports published by the [Ministry of Transport of the People's Republic of China](https://xxgk.mot.gov.cn/2020/jigou/zhghs/202006/t20200623_3313013.html). Exact report URLs are pinned in [`data/public_sources/shanghai_port_mot_2024_2025.json`](../data/public_sources/shanghai_port_mot_2024_2025.json).
- Weather: [Open-Meteo historical weather API](https://open-meteo.com/en/docs/historical-weather-api), using hourly ERA5 reanalysis near Yangshan.
- Marine factors: [Open-Meteo marine API](https://open-meteo.com/en/docs/marine-weather-api), using public model aggregation near Yangshan.

Reanalysis is observation-informed model output, not a direct terminal sensor. The marine source explicitly warns that coastal accuracy is limited and that the product is not suitable for navigation.

The pinned snapshot contains 96 missing ocean-current values and 432 missing sea-level values. They are linearly interpolated only within their respective public reanalysis series; the counts remain recorded in metadata and no terminal measurement is substituted.

## Column provenance

| Class | Columns | Claim allowed |
|---|---|---|
| Official aggregate anchor | `throughput_teu` period totals | Official port-level aggregate after source verification |
| Public reanalysis | `ambient_c`, `wind_speed_mps`, `wave_height_m`, `tide_m`, `current_speed_mps` | Public gridded environmental context |
| Engineering derivative | hourly throughput allocation, arrivals, load, tariff, carbon, berth/yard occupancy, equipment/crane/tug availability, congestion, reefer load, closure flag | Reproducible scenario feature only |
| Unavailable | `visibility_km` | Missing; the explicit availability mask remains zero |

## Intended use

- Independent Shanghai target-domain training compared with a high-frequency public reference; current evidence does not claim cross-port weight transfer.
- Three-way, no-shuffle time isolation.
- Algorithm wiring, robustness comparison, offline policy selection and site-adapter testing.
- Replacing engineering fields with authorized TOS/VTS/EMS/PLC data without changing the canonical environment contract.

## Prohibited use

- Claiming measured Shanghai terminal savings, throughput uplift or delay reduction.
- Navigation, collision avoidance or equipment safety certification.
- Production dispatch before mapping, calibration, shadow-mode and human-authority gates pass.

Regenerate from the pinned public sources:

```bash
python -m scripts.fetch_shanghai_public_dataset --refresh
```
