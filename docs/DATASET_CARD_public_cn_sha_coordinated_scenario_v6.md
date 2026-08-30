# Dataset card: public_cn_sha_coordinated_scenario_v6

## Intended use

This card describes how `public_cn_sha_integrated_scenario_v5` and the sealed `public_cn_sha_integrated_forward_2026m05_v5` challenge are used by the V6 coordinated port environment. V6 does not relabel the files as a new measured dataset. It adds a versioned, replaceable resource-chain contract for real offline optimizer training.

## Evidence boundary

- Training rows: 17,544 continuous hourly rows covering 2024–2025.
- Independent source observations: 17,566, consisting of 22 official aggregate reporting anchors plus 17,544 aligned public reanalysis hours.
- Forward challenge: 3,624 hourly rows from 2026, sealed until validation-based model selection.
- External observations: PRC Ministry of Transport Shanghai container-throughput aggregates and public weather/marine reanalysis near Yangshan.
- Not measured terminal telemetry: gate, rail, barge, reefer, shore-power, maintenance, pilotage, towage, dangerous-goods, vessel-design, equipment-chain and capacity variables.
- Production authority: false.

## V6 engineering completion

The V6 profile declares separate capacities for rail, barge, pilotage, towage, quay-crane moves, horizontal-transport moves and yard-crane moves. These are deterministic engineering parameters selected before the V6 retraining run. They are not estimates of a named Shanghai terminal. Their sole purpose is to make the offline resource-chain contract executable and replaceable.

Minimum service commitments are feasible action parameterizations outside RL. Channel closure, under-keel clearance, dangerous-goods inflow, battery reachability, reefer reserve and maintenance reserve remain hard constraints. The learner cannot trade these away for reward.

## Chronological protocol

- first 70%: training and normalization fit;
- next 10%: validation and model selection;
- final 20%: blind test within the 2024–2025 package;
- separate 2026 package: independent forward challenge after selection;
- shuffle: false;
- training rendering: disabled;
- selection: safety/business validation gates first, then lower confidence bound versus the fixed rule and FCFS;
- forward data may not participate in training, normalization or selection.

## Site replacement contract

A site replacement must provide at least 720 gap-free hours, at least 99% coverage for every V6 input, an authorized export statement, one site identity, a source-manifest SHA-256 and per-field measured or authorized-derived lineage. It must replace public and engineering fields through the stable dataset schema; it must not silently mix public replay with site telemetry.

Passing the dataset structure gate is only input admission. Site calibration, shadow acceptance, interlocks, human authority and independent acceptance remain mandatory.
