# 模块 G｜岸桥（QC）作业节拍与待机的“可直接落地”的详细设计方案（无代码版）

> 口径与 A–F 完全一致：**CMDP + 动作屏蔽/Shielding + 规则/MPC 兜底**，**强规则屏蔽 + 影子运行周期长**，**以船时/泊位效率为最高约束**。  
> 统一输出 **policy_evaluate_history.jsonl**，同名同字段即可无缝替换现场数据/接口。

---

## 1) 业务目标与范围（原文保留）

* **目标**：在不增加船舶在港时长（Vessel Time in Port, VTP）与泊位时间（BT），不降低岸桥效率（GMPH/BPH）的前提下，降低岸桥空转/待机能耗与峰段功率，抑制启停抖动与设备热负荷，兼顾电价/碳因子/DR联动。
* **对象**：QC（Quay Crane，岸边集装箱起重机），含单小车/双小车、单吊具/双吊具、自动化/半自动化版本；作业活动包含：起升（hoist）/小车（trolley）/大车（gantry 沿岸）/抓放（spreaders）/舱内协同（lashing/hatch）。

**补充与落地细化：**

* **目标优先级**（全模块统一）：  
  `安全/互锁/防撞/风切出 ≥ VTP/BT ≥ GMPH/BPH ≥ 并网/噪声合规 ≥ 需量/DR ≥ 能耗/碳 ≥ 温升/寿命成本`。
* **收益口径**：日历日（本地时区）汇总；能耗/峰罚/DR/回收/退化/切换成本与 A–F 同口径（¥）。
* **并网边界**：RMG 并网参数、PQ（PF/THD）与 PCC/N-1 裕度口径与 E/F 一致；QC 若为柴油-电混动，加入怠速燃耗与启停磨损项。
* **联动接口**：TOS（舱单/贝位/里程碑/窗口）、电价/EF、PCC/需量/DR、海岸气象（风/浪/能见度）。

---

## 2) 控制时标与滚动策略（原文保留）

* **步长**：Δt = 5 分钟（能捕捉舱作业节拍、工序切换与待机段）。
* **滚动窗口**：H = 6–12 小时（覆盖一个/多个装卸舱段与班次）。
* **分层**：
  1. **计划层（规则/MPC）**：根据船舶舱单、贝位（bay）计划、QC 配置数与作业顺序，产出节拍参考曲线（目标 GMPH、贝位推进计划、关键里程碑时刻）。
  2. **执行层（Residual RL）**：在参考曲线附近 ±δ 的安全域内微调：待机门限、速度/功率上限、节拍缓急与唤醒策略；任何动作必须通过强屏蔽（SLA/安全/风/干涉）。

**补充与落地细化：**

* **需量口径**：PCC 按**滚动 15 分钟需量**；Δt=5min 时以 3 点滑动均值对齐 15min。
* **事件前收紧**：lashing/hatch/靠离泊等窗口前 **60–120 min** 自动收紧 RL 残差带、抬升 `β（SLA）/ζ（峰罚）`。
* **影子周期**：≥ 2–4 周；仅在 **VTP/BT 不增、GMPH 不降** 连续满足后转灰度。

---

## 3) 数据与文件（最小可训练/仿真集，≤10｜原文保留）

统一时间戳 UTC；步长 5 min；单位：功率 kW、能量 kWh、温度 °C、速度 %、节拍 moves/h、风速 m/s。

1. **qc_master.csv**  
   `qc_id, model, trolley_type{single/twin}, spreader{single/tandem}, rated_kW, hoist_kW, trolley_kW, gantry_kW, regen_capable{0/1}, eco_levels{L1|L2|L3}, min_on_min, min_off_min, accel_limit, jerk_limit, sway_limit_deg, wind_cutout_mps, temp_redline_C`
2. **vessel_plan.csv（TOS 舱单/计划摘要）**  
   `vessel_id, berth_id, ata_utc, atd_utc_plan, target_gmph, qc_assigned, bays_seq(json), lashing_windows(json), hatch_moves`
3. **qc_telemetry.csv（历史/影子）**  
   `ts_utc, qc_id, state{idle/working/move/hold/fault}, mode{normal/ecoL1/L2/L3}, hoist_speed%, trolley_speed%, gantry_speed%, power_kW, energy_kWh, temp_motor_C, temp_inverter_C, start_stop_event{0/1}, regen_kWh, cycle_time_s_median, moves_5min, bay_id, interference_flag{0/1}, wind_mps, sway_deg`
4. **qc_jobs.csv（实际作业片段/节拍）**  
   `job_id, vessel_id, qc_id, bay_id, start_utc, end_utc, move_type{load/unload/hatch/lashing}, moves, avg_cycle_time_s`
5. **tos_forecast.csv（未来 H 的节拍/队列预测）**  
   `ts_utc, vessel_id, bay_id, gmph_p50, gmph_p90, queue_len_p50, queue_len_p90, next_lashing_utc, next_hatch_utc`
6. **market_price.csv（P50/P90）**  
   `ts_utc, price_yuan_kWh_p50, price_p90`
7. **grid_ef.csv（P50/P90）**  
   `ts_utc, ef_kg_kWh_p50, ef_p90`
8. **meteo_sea.csv（岸边气象/海况）**  
   `ts_utc, wind_mps, gust_mps, temp_C, rain_mm, visibility_km`
9. **grid_meter.csv（可选，峰功/需量联动）**  
   `ts_utc, P_pcc_kW, feeder_id, P_feeder_kW`

**补充与落地细化：**

* **列名候选集（鲁棒摄取）**  
  - `ts_utc`: {`ts_utc`,`timestamp`,`time_utc`,`ts`}；  
  - `qc_id`: {`qc_id`,`crane_id`}；  
  - `power_kW`: {`power_kW`,`p_kw`,`p`}；`moves_5min`: {`moves_5min`,`moves`}；  
  - `wind_mps`: {`wind_mps`,`wind_ms`,`wind`}；`sway_deg`: {`sway_deg`,`sway`}；  
  - 价格/EF：{`price_yuan_kWh_p50`,`price_p50`}/ {`ef_kg_kWh_p50`,`ef_p50`}。
* **时间戳多口径识别**：支持 epoch 秒/毫秒、`YYYY-mm-dd HH:MM:SS`、`Z`/`+08:00`；摄取失败 → 记录到 **数据质量 JSONL** 并采用**经验曲线**兜底。
* **缺测兜底**：  
  - `gmph_p50/p90` 缺失 → 用 `qc_jobs/qc_telemetry` 回归得出；  
  - `wind/gust` 缺失 → 用 `meteo_sea` 或站端默认风玫瑰；  
  - `queue_len_*` 缺失 → 依据 `moves_5min` 与 `cycle_time` 推断；  
  - `grid_meter` 缺失 → PCC 软上限按 `soft_cap_kW=PCC_limit*(1-0.05)`。
* **数据路径（默认）**：`/mnt/data/{qc_master.csv, vessel_plan.csv, qc_telemetry.csv, qc_jobs.csv, tos_forecast.csv, market_price.csv, grid_ef.csv, meteo_sea.csv, grid_meter.csv}`。

---

## 4) CMDP 任务定义（与 A–F 对齐）

### 4.1 状态 S（孪生内部）

* **设备/环境**：`state, mode, temp_motor, temp_inverter, sway_deg, wind_mps/gust_mps, accel_timer, jerk_timer, regen_buffer`
* **功率/能耗**：`power_kW, energy_kWh, regen_kWh`
* **作业/节拍**：`moves_5min, cycle_time_s_median, gmph_recent, bay_id, vessel_id, interference_flag`
* **计划/外生**：`target_gmph（参考）, next_lashing_utc, next_hatch_utc, bays_seq_pos, queue_len, price/EF 分位`
* **约束记忆**：`min_on/min_off, sway_limit_deg, wind_cutout_mps, temp_redline_C, berth_n1_margin_kW, PCC_soft_cap_kW`

**补充与落地细化：**

* **温升裕度**：`thermal_margin = min(temp_redline_C - temp_motor, temp_redline_C - temp_inverter)`；进入低裕度区收紧功率与加速度。
* **干涉/防撞**：`interference_flag=1` 或**邻位距离 < 阈值**时仅允跟随参考，不得降额/待机。

### 4.2 观测 O（策略可见）

* 实时传感 + 计划即时量；预测（`tos_forecast` 的 `gmph/queue` 分位、`market_env` 与 `meteo_sea`）；近窗历史摘要（EMA/分位）。

**补充与落地细化：**

* **跨设备观测**：同泊位其它 QC 的 `mode/功率/干涉距离` 的摘要纳入（MAPPO 时作为共享信息）。

### 4.3 动作 A（参数化混合）

* **离散**：`mode ∈ {normal, ecoL1, ecoL2, ecoL3}`；`standby ∈ {keep_awake, soft_idle, deep_idle}`。
* **连续**：  
  - `hoist/trolley/gantry_power_limit_pct ∈ [L_min, 100%]`（含斜坡/限速）；  
  - `pace_adj ∈ [-Δ, +Δ]`（对 `target_gmph` 的微调，等效于周期增减）；  
  - `wake_ahead_min ∈ [0, 10]`（lashing/hatch 结束前提前唤醒）。
* **动作掩码（强）**：`wind ≥ cutout`、`sway ≥ limit`、`interference=1`、`lashing/hatch 进行中`、`min_on/off 未满足`、`温度逼近 redline`、`SLA 风险` → 禁止降额/深待机，必要时**强制 normal 与唤醒**。

### 4.4 输出（到现场系统/TOS）

* **QC 控制**：模式位、功率/速度上限、空转关断与软启动、唤醒指令、抗摆参数（只在安全域微调）。
* **TOS**：允禁时间窗、贝位推进节拍建议（微调）、并发 QC 数量/间距建议（不触发干涉）。
* **电力侧（可选）**：DR 减载曲线、PCC 峰值保护信号。

### 4.5 约束与屏蔽（硬+软）

* **SLA/船时**：`VTP/BT 不增，GMPH 不降`（按 P50/P90 舱段目标）；  
* **安全/合规**：`wind ≤ cutout, sway ≤ limit, 防撞/互锁, min_on/off, accel/jerk 限`；  
* **工序互锁**：`lashing/hatch` 期间禁止深待机与降额导致复位延迟；  
* **电气**：`PCC/馈线/N-1` 上限；DR/需量窗口峰值约束；  
* **温度**：`temp < redline`；越阈自动限流与冷却。
  
**屏蔽伪码（与 A–F 一致）**：
