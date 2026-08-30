# Site integration gateway and direct dataset replacement

The repository now exposes two separate fail-closed paths for a future port connection. Neither path grants dispatch or production authority.

## 1. Read-only current snapshot gateway

`GET /api/v3/site-integration/readiness` inventories eight required adapters:

1. terminal operating system (TOS);
2. automatic identification system / vessel traffic service (AIS/VTS);
3. weather and hydrography;
4. energy supervisory control and data acquisition (SCADA);
5. equipment programmable logic controllers (PLC);
6. gate, rail and barge systems;
7. reefer monitoring;
8. computerized maintenance management and workforce systems.

`POST /api/v3/site-integration/ingest` accepts `port-snapshot.v1`. Every envelope is checked for:

- configured site, adapter, owner and source-system identity;
- HMAC-SHA256 signature and canonical payload SHA-256;
- timezone-aware observation time and a fixed freshness limit;
- strictly increasing sequence and snapshot replay protection;
- required fields, exact units, quality labels, finite values and semantic ranges.

Accepted state stores lineage digests and replay identifiers only, not the raw payload. The endpoint is read-only: `dispatch_allowed=false` and `production_authority=false` are invariant.

Configuration is private and intentionally absent from the public repository:

```text
PORT_DT_SNAPSHOT_SITE_ID=SITE.AUTHORIZED.PORT
PORT_DT_SNAPSHOT_ADAPTER_CONFIG=/private/site-adapters.json
PORT_DT_SNAPSHOT_LIVE_ATTESTED=true
```

The adapter configuration contains owner and source-system identities plus either a secret or a private environment-variable name. Secrets must not be committed.

## 2. Historical dataset replacement

Upload a chronological CSV with `POST /api/rl/datasets/upload`, an explicit canonical field mapping, and governance metadata. Public/engineering values are never silently used to fill missing site fields.

Run:

```text
GET /api/rl/datasets/{dataset_id}/quality
GET /api/rl/datasets/{dataset_id}/site-readiness
```

The V6 site-readiness gate requires:

- at least 720 continuous hours;
- at least 99% coverage for all canonical, environmental, regulatory and port-wide V6 inputs;
- no physical-bound or chronology failure;
- `authorized_site_data=true`, a `site_id`, an authorized site-export provenance type and a 64-character source-manifest SHA-256;
- per-field `source_system`, `source_field`, `owner` and `evidence_class` (`measured` or `authorized_derived`).

Passing this gate means only that the dataset can replace the public training scenario. It does not prove live connectivity, calibration, benefit, safe actuation or production acceptance. The replacement data must still be split chronologically, retrained with multiple seeds, compared with measured current operations, challenged on a sealed later period, calibrated in the twin and run in read-only shadow mode.

## Authority boundary

The following remain outside reinforcement learning and outside this gateway:

- customs, maritime, dangerous-goods and navigation release;
- under-keel-clearance acceptance and vessel-traffic instructions;
- berth/resource schedule commitment;
- equipment interlocks and emergency stops;
- work-order creation, execution approval and human overrides;
- identity, access control, audit retention and change approval.

Those functions must be supplied by authorized port systems and independently accepted under the production-readiness chain.
