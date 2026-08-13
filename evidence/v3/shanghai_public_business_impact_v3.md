# V3 Shanghai public-data business-impact scenario

- Status: `DESCRIPTIVE_PAIRED_PUBLIC_DATA_SCENARIO`
- Comparison: **MPC vs FCFS**, 10 identical blind-test windows
- Window length: **48 hours**
- Safety guardrail violation rate: **0.00%**

| Metric | Improvement | Relative improvement | Paired-window 95% CI |
|---|---:|---:|---:|
| throughput_teu | 37707.33 TEU | 21.63% | [35001.45, 40279.11] TEU |
| delay_index_mean | 2.90 index | 28.52% | [2.76, 3.04] index |
| energy_cost | -70574.63 CNY | -8.82% | [-71684.09, -69433.92] CNY |
| carbon_kg | -56738.92 kg | -8.96% | [-58418.75, -55161.83] kg |
| peak_kw | -2735.66 kW | -10.73% | [-2799.15, -2676.67] kW |

- Absolute total-cost difference, FCFS minus MPC: **CNY -12,879,869/year** (may be negative when MPC handles more work)
- Absolute total-carbon difference, FCFS minus MPC: **-10,354.85 tCO2/year** (may be negative when MPC handles more work)
- MPC equivalent-throughput avoided cost: **CNY 18,826,863/year**
- MPC equivalent-throughput avoided carbon: **14,774.50 tCO2/year**

## Learned-policy equivalent-throughput value

- Policy: **SAC**
- Status: `EQUIVALENT_THROUGHPUT_VALUE_95CI`
- Unit cost improvement: **4.11%**
- Unit carbon improvement: **3.90%**
- Mechanical annualized avoided cost at equivalent throughput: **CNY 6,632,753/year**
- Mechanical annualized avoided carbon at equivalent throughput: **4,998.58 tCO2/year**

Equivalent-throughput avoided value is not an absolute electricity-bill reduction: it prices the learned policy's throughput at the FCFS unit intensity.

MPC is reported as the named deterministic safety controller; this descriptive business-impact export was generated after the controller benchmark and is not a preregistered superiority test.
Public Shanghai aggregate throughput and public Yangshan reanalysis are used. Electricity tariff, carbon factor, terminal load, equipment and operating fields are engineering assumptions. Values are not Shanghai International Port Group savings, audited carbon reductions or production KPIs; replace them with authorized EMS/TOS/finance/carbon-ledger data and pass shadow-operation acceptance before any site claim.
