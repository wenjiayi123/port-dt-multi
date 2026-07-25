# RL benchmark evidence: `public_port_ops_v1`

- Dataset SHA-256: `926cc138af1859b6d525e24fd4b8594b2d8573b16fe1e8ed1875b6f7854eb71e`
- Rows: 52,608
- Split: 42,086 train / 10,522 chronological holdout
- Training rendering: disabled and verified per run
- Comparative gate: at least three seeds and 10,000 optimizer steps per RL method

| Controller | Formal runs | Seeds | Gate | Reward | Energy cost | Carbon kg | Throughput TEU | Peak kW | Delay index | Violations |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|
| SAC | 0 | — | PENDING | — | — | — | — | — | — | — |
| PPO | 0 | — | PENDING | — | — | — | — | — | — | — |
| TD3 | 0 | — | PENDING | — | — | — | — | — | — | — |
| DQN | 0 | — | PENDING | — | — | — | — | — | — | — |
| A2C | 0 | — | PENDING | — | — | — | — | — | — | — |
| TQC | 0 | — | PENDING | — | — | — | — | — | — | — |
| MPC | 0 | — | PENDING | — | — | — | — | — | — | — |

`RL_SMOKE_WIRING_ONLY` runs are retained in the JSON bundle but excluded from this performance table.
The figures are deterministic-policy results on the chronological public-data holdout, not measured terminal KPIs or production savings.
