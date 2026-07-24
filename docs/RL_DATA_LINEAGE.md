# RL_DATA_LINEAGE

## 1. 数据分层结论
本轮六模块的数据可以先分成四类：

### A. 状态 / 遥测 / 业务事件
- AGV：`vehicle_state.csv`、`tos_jobs.csv`
- Shore BESS：`ship_calls.csv`、`bess_telemetry.csv`、`shore_power_telemetry.csv`、`grid_meter.csv`
- HVAC：`hvac_telemetry.csv`
- Yard Lighting：`lighting_telemetry.csv`、`complaints_events.csv`
- Yard Crane：crane telemetry / queue / wait time 类表
- QC：`qc_telemetry.csv`、`qc_jobs.csv`、`vessel_plan.csv`

### B. 预测 / 仿真输入
- HVAC：`weather_forecast.csv`、`load_forecast.csv`
- Yard Lighting：`activity_forecast.csv`、`weather_astro.csv`
- QC：`tos_forecast.csv`、`meteo_sea.csv`
- AGV / Shore BESS / Yard Crane 里也都出现 forecast / baseline / offline dataset 口径

### C. 价格 / 碳因子 / 需量 / 规则约束
- 通用表：`market_price.csv`、`grid_ef.csv`、`demand_window_config.json`
- 设备主数据：`vehicles_master.csv`、`plant_master.json`、`bess_master.json`、`qc_master.csv`、`zones_master.csv`
- 这些不属于“实时生产事件”，更接近参数、约束、配置、规则输入

### D. 训练 / 评估 / 策略产物
- `policy.bin` / `policy_meta.json`
- `offline_train.json` / `offline_train.jsonl`
- `offline_dataset*.jsonl`
- `policy_evaluate_history.jsonl`
- `shore_bess_outputs.jsonl`
- `data_quality.jsonl`

## 2. 六模块数据来源梳理

### A. AGV Charge
- 业务任务来源：`tos_jobs.csv`
- 车辆实时状态来源：`vehicle_state.csv`
- 设备静态约束来源：`vehicles_master.csv`
- 能源/价格/排放来源：`grid_meter.csv`、`market_price.csv`、`grid_ef.csv`
- 训练/评估产物：`artifacts/policy_evaluate_history.jsonl`、`policy.bin`、`policy_meta.json`
- 公开口径建议：讲成“按港区 AGV 作业任务流与电池状态构造的 simulation/proxy dataset”最稳。

### B. Yard Lighting
- 分区与照明配置：`zones_master.csv`、`config_limits.json`
- 状态与投诉约束：`lighting_telemetry.csv`、`complaints_events.csv`
- 外部环境：`weather_astro.csv`
- 需求预测：`activity_forecast.csv`
- 价格/碳：`market_price.csv`、`grid_ef.csv`
- 训练/评估产物：`lighting_state.json`、`offline_train.json`、`offline_train.jsonl`、`lighting_dataset_report.json`
- 公开口径建议：可讲“规则生成 + 仿真回放 + 历史状态快照混合数据”。

### C. HVAC Cooling
- 预测输入：`weather_forecast.csv`、`load_forecast.csv`
- 设备状态：`hvac_telemetry.csv`
- 主数据/约束：`plant_master.json`、`ahu_zones_master.csv`、`demand_window_config.json`、`plant_efficiency_map.csv`
- 价格/碳：`market_price.csv`、`grid_ef.csv`
- 训练/评估产物：`artifacts/policy_evaluate_history.jsonl`、`hvac_cooling_state.json`、`policy.bin`
- 公开口径建议：可讲“forecast-driven simulation/proxy control dataset”。

### D. Shore BESS
- 船舶靠泊计划：`ship_calls.csv`
- 设备 / 电表 / 岸电状态：`bess_telemetry.csv`、`shore_power_telemetry.csv`、`grid_meter.csv`
- 主数据 / 约束：`berths_master.csv`、`bess_master.json`、`demand_window_config.json`
- 价格/碳：`market_price.csv`、`grid_ef.csv`
- 训练/评估/审计产物：`shore_bess_outputs.jsonl`、`mode_state.json`、`point_write_state.json`、`policy.bin`
- 公开口径建议：可以讲“berth schedule + energy asset proxy data + rule-constrained simulation outputs”。

### F. Yard Crane
- 状态与队列：模块注释明确包含 crane telemetry、queue、wait time、temperature、regen 等字段
- 经济外生变量：price / ef / pcc / DR
- 主数据与运行约束：由模块内 `COLS` 和默认参数驱动
- 训练/评估产物：`policy_evaluate_history.jsonl`、前端兼容 `offline_dataset_crane.jsonl / offline_dataset_crane_aug.jsonl`
- 公开口径建议：若没有能证明来自生产 RTG/RMG 系统的原始采集链，建议统一讲成 simulation/proxy。

### G. QC / Port_G_qc_mvp
- 船舶计划：`vessel_plan.csv`
- 作业与设备状态：`qc_jobs.csv`、`qc_telemetry.csv`
- 主数据：`qc_master.csv`
- 预测与环境：`tos_forecast.csv`、`meteo_sea.csv`
- 价格/碳/电网：`market_price.csv`、`grid_ef.csv`、`grid_meter.csv`
- 训练/评估产物：`offline_dataset_qc.jsonl`、`policy_evaluate_history.jsonl`、`data_quality.jsonl`、`policy.bin`
- 公开口径建议：最稳妥口径是“基于船期计划和岸桥作业特征构造的 proxy dataset + digital twin environment”。

## 3. 哪些能公开讲成 simulation / proxy data

### 可以公开这样讲的
- 含明显泛化 ID 的表：如 `AGV_001`、`VSL_28671`、`SC_00001` 这类匿名化或模板化字段。
- 规则/配置/主数据表：如 `vehicles_master.csv`、`plant_master.json`、`config_limits.json`。
- 预测类表：如 `weather_forecast.csv`、`load_forecast.csv`、`activity_forecast.csv`、`tos_forecast.csv`。
- 离线训练与评估产物：`offline_dataset*.jsonl`、`offline_train.jsonl`、`policy_evaluate_history.jsonl`。
- 通过 baseline / rule / MPC / shield 组合出来的推荐调度输出。

### 不建议直接讲成真实生产数据的
- `vehicle_state.csv`、`ship_calls.csv`、`vessel_plan.csv`、`hvac_telemetry.csv`、`lighting_telemetry.csv`、`qc_telemetry.csv` 这类“长得像生产表”的数据。
- 除非你能补充明确来源链：来自某真实 TOS / EMS / BAS / SCADA 的脱敏导出。
- 当前仅从字段和命名看，更稳妥的公开口径应是：
  - simulation dataset
  - proxy operational data
  - de-identified / synthetic operational schema

## 4. 本轮上传样本的判断

### AGV
- `tos_jobs.csv` 的字段是 `timestamp / vehicle_id / job_id / due_time / priority / yard_block / berth_id`
- `vehicle_state.csv` 的字段是 `timestamp / vehicle_id / soc / available / priority / eta_min / temp`
- `vehicles_master.csv` 的字段是 `vehicle_id / battery_kwh / p_charge_max_kw / soc_min / soc_max / soc_target / can_swap`
- 这些字段非常像“按真实业务语义设计的代理数据”，但没有真实港口专有编码。
- 所以对外最稳妥口径：`simulation/proxy data based on AGV operations schema`

### Shore BESS
- `ship_calls.csv` 的字段是 `ship_call_id / vessel_id / vessel_type / berth_id / eta_utc / etd_utc / nominal_shore_kw`
- 船名、港名、公司名都没有，ID 也是模板化。
- 对外最稳妥口径：`proxy berth call schedule for shore power dispatch simulation`

### QC
- `vessel_plan.csv` 的字段是 `ship_call_id / vessel_id / vessel_class / quay_id / eta_utc / etd_utc / planned_moves / discharge_pct / twinlift_allowed`
- 同样更像模板化计划表，不像真实生产快照。
- 对外最稳妥口径：`proxy vessel and quay plan used by QC digital twin`

## 5. 哪两个模块最适合优先收口成展示重点

### 第一优先：Shore BESS
- 数据链完整：计划、遥测、价格、碳因子、BESS 主数据、输出 JSONL 都齐。
- 控制链完整：dispatch、range、audit、actuate、mode、export 都有。
- 公开叙事自然：岸电接入、削峰降费、BESS 协同。

### 第二优先：HVAC Cooling
- 数据分层最清楚：预测、遥测、配置、效率图、状态、动作日志。
- 控制逻辑最容易讲成工业闭环：plan -> residual -> shield -> write_jobs。
- 不容易被追问“为什么没有真实 TOS 航运数据”。

## 6. 最稳妥的对外总口径
- 可以讲：
  - `simulation-backed RL module`
  - `proxy operational dataset`
  - `de-identified operational schema`
  - `digital-twin-driven control dataset`
- 不建议直接讲：
  - `real production data from Shanghai Port`
  - `live operational data`
  - `online production control`
- 更稳妥的表述：
  - `The current prototype uses a mix of simulation data, proxy operational tables, rule-generated inputs, and configurable engineering parameters.`

