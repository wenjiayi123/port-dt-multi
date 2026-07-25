# Third-party notices / 第三方声明

Port DT Multi is distributed under the MIT License. Python dependencies retain their own licences; pinned versions are listed in `requirements.txt` and should be reviewed during every release.

The RL engine uses Stable-Baselines3 and SB3-Contrib under their MIT licences.
TQC is loaded from SB3-Contrib; the remaining six learning/control entries use
Stable-Baselines3 or SciPy as declared in the algorithm capability response.

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

The bundled Xiaoyi Q-style maritime-officer PNG is a project-owned character
asset supplied by the repository author and shared across the author's port-AI
projects. It and the repository-native diagrams are released under the
repository MIT License. No private screenshots, proprietary models or
software-copyright submission materials are included.
