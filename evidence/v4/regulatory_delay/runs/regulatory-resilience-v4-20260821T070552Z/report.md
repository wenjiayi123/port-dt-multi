# V4 regulatory resilience evidence

- Status: `ADMITTED_OFFLINE_SCENARIO_CANDIDATE`
- Dataset: `public_cn_sha_regulatory_scenario_v4` / `2325d935d8e752940e00b8c5227a07d1fe6da9c74da0a0c254fcc349b7988c08`
- Selected: SAC seed `185` / `rl-20260821T065745532Z`
- Training: 3 seeds x 20,000 steps; rendering disabled
- Blind test: 20 paired 48-hour windows
- Boundary: predeclared engineering regulatory stress scenario, not field KPI

## Versus regulator-unaware V3 engineering SOP adapter

- Regulatory delay TEU-hours improvement: 64.36% (95% CI 59.34% to 69.32%)
- Service completion improvement: 13.73%
- Cost/TEU improvement: 10.04%
- Guardrail violation rate: 0.000000

The policy reserves inspection readiness and prioritizes post-release recovery only. It has no authority to change a maritime/customs decision or execute production dispatch.
