# RL benchmark evidence: `public_cn_sha_hourly_v3`

- Dataset SHA-256: `803214ea0202abde241f75a28d7bf46b9c7ad801d40605a0916ec14ef7906a01`
- Rows: 17,544
- Split: 12,280 train / 1,755 validation / 3,509 chronological blind holdout
- Training rendering: disabled and verified per run
- Comparative gate: at least three seeds and 10,000 optimizer steps per RL method

| Controller | Formal runs | Seeds | Gate | Reward | Energy cost | Carbon kg | Throughput TEU | Peak kW | Delay index | Violations |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|
| SAC | 6 | 42, 142, 242 | PASS | -122.2546 | 834,870.7848 | 661,099.0654 | 190,005.2113 | 28,517.5628 | 8.3305 | 0.0000 |
| PPO | 3 | 42, 142, 242 | PASS | -148.3103 | 814,388.0545 | 641,708.3384 | 170,626.2982 | 26,303.9880 | 10.4997 | 0.0000 |
| TD3 | 3 | 42, 142, 242 | PASS | -98.3046 | 914,373.6906 | 723,012.6973 | 218,907.1210 | 30,819.0864 | 6.6791 | 0.0007 |
| DQN | 3 | 42, 142, 242 | PASS | -118.7667 | 865,337.1279 | 685,851.3964 | 203,387.8701 | 30,754.3358 | 8.2347 | 0.0007 |
| A2C | 3 | 42, 142, 242 | PASS | -138.9163 | 808,561.1552 | 639,497.5825 | 178,637.0274 | 25,698.9370 | 9.8148 | 0.0000 |
| TQC | 3 | 42, 142, 242 | PASS | -107.1150 | 876,754.3495 | 692,322.2270 | 208,581.7072 | 29,389.7880 | 7.3822 | 0.0021 |
| QR-DQN | 3 | 42, 142, 242 | PASS | -126.5482 | 845,648.8126 | 669,293.4997 | 190,992.8375 | 27,113.0683 | 8.8587 | 0.0000 |
| TRPO | 3 | 42, 142, 242 | PASS | -146.6161 | 779,455.2927 | 619,365.3088 | 168,488.7025 | 27,171.8082 | 10.3821 | 0.0021 |
| Recurrent PPO | 3 | 42, 142, 242 | PASS | -124.5353 | 831,284.4793 | 658,791.0590 | 192,909.2741 | 26,707.4817 | 8.7168 | 0.0007 |
| ARS | 3 | 42, 142, 242 | PASS | -134.4444 | 821,868.7471 | 649,347.6194 | 182,566.7951 | 26,343.9271 | 9.4842 | 0.0000 |
| MPC | 2 | 42 | PASS | -105.3926 | 870,804.3533 | 689,939.8285 | 212,000.5533 | 28,240.3081 | 7.2650 | 0.0000 |
| FCFS neutral | 2 | 42 | PASS | -143.0606 | 800,229.7267 | 633,200.9129 | 174,293.2229 | 25,504.6502 | 10.1635 | 0.0000 |

`RL_SMOKE_WIRING_ONLY` runs are retained in the JSON bundle but excluded from this performance table.
The figures are deterministic-policy results on the chronological public-data holdout, not measured terminal KPIs or production savings.
