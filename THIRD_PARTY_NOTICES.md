# Third-party notices / 第三方声明

Port DT Multi is distributed under the MIT License. Python dependencies retain their own licences; pinned versions are listed in `requirements.txt` and should be reviewed during every release.

The RL engine uses Stable-Baselines3 and SB3-Contrib under their MIT licences.
TQC, QR-DQN, TRPO, Recurrent PPO and ARS are loaded from SB3-Contrib; SAC,
PPO, TD3, DQN and A2C use Stable-Baselines3. MPC uses SciPy and FCFS is a
repository-native rule comparator, as declared by the capability response.

The Web dashboard bundles Apache ECharts 6.1.0 under the Apache License 2.0 in
`app/static/vendor/echarts`. This checked-in runtime removes the dashboard's
CDN dependency so charts remain available in an offline first-clone review.
The port perspective renderer itself is repository-native Canvas 2D code and
does not download a 3D engine at runtime.

The bundled `public_port_ops_v1` integration dataset combines:

- container-throughput input published by the Maritime and Port Authority of
  Singapore through data.gov.sg under the Singapore Open Data Licence; and
- container-vessel arrival input published by the same authority and licence.

All hourly allocations, electrical, environmental, tariff, carbon, and
tide-stress fields in that example are deterministic engineering derivatives
documented in the dataset card. The file is not a measured terminal record.

The bundled `public_us_la_6min_v1` public benchmark combines:

- monthly Port of Los Angeles TEU totals published by the U.S. Department of
  Transportation Bureau of Transportation Statistics;
- verified six-minute water-level observations from NOAA CO-OPS station
  `9410660`; and
- six-minute air-temperature and wind observations from NOAA CO-OPS station
  `9410840`.

U.S. federal government data are generally not subject to domestic copyright;
the source agencies' terms, disclaimers, and attribution guidance still apply.
All terminal electrical, operational, occupancy, tariff, carbon, and
sub-monthly allocation fields are documented engineering derivatives. The
dataset is a public benchmark, not a measured terminal telemetry export.

The bundled `public_cn_sha_hourly_v3` target-domain benchmark combines:

- cumulative Shanghai Port container-throughput tables published by the
  Ministry of Transport of the People's Republic of China; and
- hourly weather and marine reanalysis/model values accessed through
  Open-Meteo near the Yangshan public grid point.

Open-Meteo attribution and upstream model/data terms remain applicable; the
exact endpoints, access date, grid coordinates and snapshot hash are retained
in dataset metadata. Reanalysis is not site telemetry and the marine product
is not used for navigation. Hourly throughput allocation and every internal
terminal-operations field are declared engineering derivatives pending
authorized site replacement.

The separately bundled `public_cn_sha_forward_2026m05_v1` challenge uses four
later Ministry of Transport cumulative Shanghai throughput workbooks covering
January-May 2026 and 3,624 hourly weather/marine model observations accessed
through the same public Open-Meteo endpoints. Exact publication URLs, workbook
SHA-256 values, access date, grid points and the pinned reanalysis snapshot are
retained. It is a temporal forward challenge, not a new training source and not
terminal telemetry; candidate selection on this artifact is prohibited.

The bundled Xiaoyi Q-style maritime-officer PNG is a project-owned character
asset supplied by the repository author and shared across the author's port-AI
projects. It and the repository-native diagrams are released under the
repository MIT License. No private screenshots, proprietary models or
software-copyright submission materials are included.
