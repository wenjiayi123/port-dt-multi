# V3 Shanghai public-data advantage evidence

- Status: `STRICT_ADVANTAGE_95CI`
- Selected policy family: **SAC**
- Dataset SHA-256: `803214ea0202abde241f75a28d7bf46b9c7ad801d40605a0916ec14ef7906a01`
- Weighted relative improvement vs FCFS: **2.95%**
- 95% bootstrap CI: **[1.68%, 4.00%]**

| Metric | Relative improvement | 95% CI | Direction |
|---|---:|---:|---|
| throughput_teu | 8.40% | [3.56%, 11.47%] | higher is better |
| delay_index_mean | 16.77% | [9.78%, 21.38%] | lower is better |
| energy_cost | -4.02% | [-5.58%, -1.47%] | lower is better |
| carbon_kg | -4.23% | [-5.43%, -2.34%] | lower is better |
| peak_kw | -12.70% | [-19.32%, -6.74%] | lower is better |

The algorithm was selected on chronological validation rows. This final deterministic-policy comparison uses the untouched chronological blind test only; blind-test scores did not select the winner.
Advantages are offline deterministic-policy results on a public Shanghai aggregate plus public reanalysis scenario. They are not measured Shanghai terminal KPIs and cannot authorize production dispatch.
