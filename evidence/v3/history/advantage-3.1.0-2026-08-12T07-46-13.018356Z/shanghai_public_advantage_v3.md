# V3 Shanghai public-data advantage evidence

- Status: `STRICT_ADVANTAGE_95CI`
- Selected policy family: **SAC**
- Dataset SHA-256: `803214ea0202abde241f75a28d7bf46b9c7ad801d40605a0916ec14ef7906a01`
- Weighted relative improvement vs FCFS: **3.98%**
- 95% bootstrap CI: **[2.71%, 4.91%]**

| Metric | Relative improvement | 95% CI | Direction |
|---|---:|---:|---|
| throughput_teu | 9.63% | [4.63%, 13.72%] | higher is better |
| delay_index_mean | 19.30% | [15.64%, 21.32%] | lower is better |
| energy_cost | -4.63% | [-6.09%, -2.56%] | lower is better |
| carbon_kg | -4.58% | [-5.90%, -2.26%] | lower is better |
| peak_kw | -10.93% | [-11.75%, -10.05%] | lower is better |

The algorithm was selected on chronological validation rows. This final deterministic-policy comparison uses the untouched chronological blind test only; blind-test scores did not select the winner.
Advantages are offline deterministic-policy results on a public Shanghai aggregate plus public reanalysis scenario. They are not measured Shanghai terminal KPIs and cannot authorize production dispatch.
