# RL benchmark evidence: `public_us_la_6min_v1`

- Dataset SHA-256: `9455a32251c521f567887f0205b0b5db4556801924d443ae1856db9ab4262897`
- Rows: 87,459
- Split: 69,967 train / 17,492 chronological holdout
- Training rendering: disabled and verified per run
- Comparative gate: at least three seeds and 10,000 optimizer steps per RL method

| Controller | Formal runs | Seeds | Gate | Reward | Energy cost | Carbon kg | Throughput TEU | Peak kW | Delay index | Violations |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|
| SAC | 3 | 42, 142, 242 | PASS | -2,124.5845 | 17,248.5737 | 26,009.9970 | 48,260.8338 | 2,776.9048 | 26.0062 | 0.0000 |
| PPO | 3 | 42, 142, 242 | PASS | -3,586.1429 | 17,292.8926 | 25,922.3093 | 44,083.9425 | 2,529.1466 | 43.9178 | 0.0000 |
| TD3 | 3 | 42, 142, 242 | PASS | -1,143.2595 | 17,692.5936 | 26,517.9337 | 52,149.7334 | 2,964.5872 | 13.9792 | 0.0000 |
| DQN | 3 | 42, 142, 242 | PASS | -301.8840 | 16,999.3253 | 25,478.6727 | 52,971.6015 | 2,790.6106 | 3.6695 | 0.0001 |
| A2C | 3 | 42, 142, 242 | PASS | -3,014.6426 | 17,409.4319 | 26,103.2486 | 45,617.9400 | 2,556.9195 | 36.9139 | 0.0000 |
| TQC | 3 | 42, 142, 242 | PASS | -137.8413 | 17,356.8677 | 26,077.4242 | 53,389.9698 | 2,777.5157 | 1.6588 | 0.0000 |
| MPC | 1 | 42 | PASS | -161.0858 | 17,039.3012 | 25,523.7903 | 53,340.9107 | 2,473.7651 | 1.9444 | 0.0000 |

`RL_SMOKE_WIRING_ONLY` runs are retained in the JSON bundle but excluded from this performance table.
The figures are deterministic-policy results on the chronological public-data holdout, not measured terminal KPIs or production savings.
