# RL_MODULE_MAP

## 1. 六个 RL 主模块

### A. AGV Charge
- 业务定位：AGV 充电排程 / 车队 SoC 与充电窗口协调。
- 主代码入口：`app/services/rl_model/agv_charge/module.py`
- 服务入口：`app/services/rl_model/agv_charge/api.py`
- 平台治理入口：`app/services/rl_ops_center/api.py`
- 已见数据入口：
  - `app/services/rl_model/agv_charge/data/tos_jobs.csv`
  - `app/services/rl_model/agv_charge/data/vehicle_state.csv`
  - `app/services/rl_model/agv_charge/data/vehicles_master.csv`
  - 同目录还包含 `charge_sessions.csv / chargers_master.csv / grid_meter.csv / market_price.csv / grid_ef.csv / port_grid_config.json`
- artifact 入口：
  - `app/services/rl_model/agv_charge/artifacts/policy_evaluate_history.jsonl`
  - `app/services/rl_model/agv_charge/policy.bin`
  - `app/services/rl_model/agv_charge/policy_meta.json`
- 当前备注：本轮材料里没有拿到 AGV `module.py` 正文，只拿到了目录与数据文件，因此对内部函数级结构只做路径级映射。

### B. Yard Lighting
- 业务定位：堆场照明滚动控制 / RL 调光 / MPC 或规则兜底。
- 主代码入口：`app/services/rl_model/yard_lighting/api.py`
- 平台治理入口：`app/services/rl_ops_center/api.py`
- 已见数据入口：
  - `app/services/rl_model/yard_lighting/data/activity_forecast.csv`
  - `app/services/rl_model/yard_lighting/data/complaints_events.csv`
  - `app/services/rl_model/yard_lighting/data/config_limits.json`
  - `app/services/rl_model/yard_lighting/data/lighting_telemetry.csv`
  - `app/services/rl_model/yard_lighting/data/weather_astro.csv`
  - `app/services/rl_model/yard_lighting/data/market_price.csv`
  - `app/services/rl_model/yard_lighting/data/grid_ef.csv`
  - `app/services/rl_model/yard_lighting/data/zones_master.csv`
- artifact 入口：
  - `app/services/rl_model/yard_lighting/artifacts/lighting_state.json`
  - `app/services/rl_model/yard_lighting/artifacts/offline_train.json`
  - `app/services/rl_model/yard_lighting/artifacts/offline_train.jsonl`
  - `app/services/rl_model/yard_lighting/artifacts/lighting_dataset_report.json`
  - `app/services/rl_model/yard_lighting/policy.bin`
  - `app/services/rl_model/yard_lighting/policy_meta.json`

### C. HVAC Cooling
- 业务定位：冷站 / 末端设定点联动，MPC 参考 + residual 微调 + safety shield + BAS 写点任务编排。
- 主代码入口：`app/services/rl_model/hvac_cooling/module.py`
- 依赖服务入口：`app/services/rl_model/hvac_cooling/api.py`
- 平台治理入口：`app/services/rl_ops_center/api.py`
- 已见数据入口：
  - `app/services/rl_model/hvac_cooling/data/weather_forecast.csv`
  - 同目录还包含 `hvac_telemetry.csv / load_forecast.csv / demand_window_config.json / plant_master.json / plant_efficiency_map.csv / market_price.csv / grid_ef.csv / ahu_zones_master.csv`
- artifact 入口：
  - `app/services/rl_model/hvac_cooling/artifacts/policy_evaluate_history.jsonl`
  - `app/services/rl_model/hvac_cooling/artifacts/hvac_cooling_state.json`
  - `app/services/rl_model/hvac_cooling/policy.bin`
  - 运行日志：`run_*.log / train_*.log`

### D. Shore BESS
- 业务定位：岸电 + 电池储能联合调度，基线 + 策略残差 + 安全投影 + 点表握手。
- 主代码入口：`app/services/rl_model/shore_bess/api.py`
- 依赖逻辑入口：`app/services/rl_model/shore_bess/adapter.py`、`rl_engine.py`
- 平台治理入口：`app/services/rl_ops_center/api.py`
- 已见数据入口：
  - `app/services/rl_model/shore_bess/data/ship_calls.csv`
  - 同目录还包含 `berths_master.csv / bess_master.json / bess_telemetry.csv / demand_window_config.json / grid_ef.csv / grid_meter.csv / market_price.csv / shore_power_telemetry.csv`
- artifact 入口：
  - `app/services/rl_model/shore_bess/artifacts/shore_bess_outputs.jsonl`
  - `app/services/rl_model/shore_bess/artifacts/mode_state.json`
  - `app/services/rl_model/shore_bess/artifacts/point_write_state.json`
  - `app/services/rl_model/shore_bess/policy.bin`
  - `app/services/rl_model/shore_bess/policy_meta.json`

### F. Yard Crane
- 业务定位：RTG/RMG 待机与功率模式，孪生环境 + 规则/MPC 兜底 + residual RL + 动作屏蔽。
- 主代码入口：`app/services/rl_model/yard_crane/module_f.py`
- 平台治理入口：`app/services/rl_ops_center/api.py`
- 已见数据入口：
  - 目录已确认存在 `app/services/rl_model/yard_crane/data/*`
  - 本轮未单独上传 F 模块 csv，但前端与模块注释均表明其读取 crane telemetry / queue / price / ef 等表
- artifact 入口：
  - `app/services/rl_model/yard_crane/policy_evaluate_history.jsonl`
  - `static/api/rl/artifacts/policy_evaluate_history.jsonl`（镜像供前端展示）
  - 前端还兼容 `offline_dataset_crane.jsonl / offline_dataset_crane_aug.jsonl`

### G. QC / Port_G_qc_mvp
- 业务定位：岸桥作业节拍与待机，数字孪生 + 计划层 + Shielding。
- 主代码入口：`app/services/rl_model/port_G_qc_mvp/module_g.py`
- 平台治理入口：`app/services/rl_ops_center/api.py`
- 已见数据入口：
  - `app/services/rl_model/port_G_qc_mvp/data/vessel_plan.csv`
  - 同目录还包含 `qc_jobs.csv / qc_telemetry.csv / qc_master.csv / tos_forecast.csv / market_price.csv / grid_ef.csv / grid_meter.csv / meteo_sea.csv`
- artifact 入口：
  - `app/services/rl_model/port_G_qc_mvp/policy_evaluate_history.jsonl`
  - `app/services/rl_model/port_G_qc_mvp/offline_dataset_qc.jsonl`
  - `app/services/rl_model/port_G_qc_mvp/data_quality.jsonl`
  - `app/services/rl_model/port_G_qc_mvp/policy.bin`
  - `app/services/rl_model/port_G_qc_mvp/policy_meta.json`

## 2. RL Ops Center 的平台角色
- `app/services/rl_ops_center/api.py` 不是六个模型之一，但它是统一治理入口。
- 当前已暴露的能力包括：`/ping`、`/overview`、`/ope/eval`、`/policies`、`/policies/verify`、`/signals`、`/experiments`、`/rollback`、`/causal/estimate`。
- 所以在平台叙事里，六模块更像“模型族”，RL Ops Center 更像“训练/评估/守护栏/上线治理总面板”。

## 3. 哪两个模块最适合先讲

### 第一优先：Shore BESS
原因：
- 不只是离线训练，有完整 HTTP API、推荐调度、模式切换、写点握手、CSV 导出。
- 有较清晰的工程落地链：baseline -> recommendation -> audit -> actuation stub。
- 容易对外讲成“能源成本、岸电负荷、BESS 协同”的可落地模块。

### 第二优先：HVAC Cooling
原因：
- 控制链完整，结构清楚：plan -> decide -> masks -> final_action -> write_jobs。
- 数据、配置、状态、日志分层明确，适合讲“可控、可审计、可回退”的工业控制口径。
- 对外叙事上容易落到“需量管理 + 舒适/工艺约束 + 能效优化”。

## 4. 公开展示时的建议模块分层
- 强展示：Shore BESS、HVAC Cooling
- 次展示：AGV Charge、QC
- 补充展示：Yard Lighting、Yard Crane

