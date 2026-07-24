# 模块 F｜场桥（RTG/RMG）待机与功率模式 —— 可直接落地的无代码设计书
（版本：v1.0；与 A–E 的 CMDP + Shielding + 规则/MPC 兜底范式严格一致；字段名即落地口径）

> 数据口径与外部边界：  
> - DR 事件：/mnt/data/dr_events.json（`event_id, start_utc, end_utc, required_reduction_kw, target_blocks, price_adder_yuan_per_kWh`）。:contentReference[oaicite:3]{index=3}  
> - 需量窗口与软限：demand_window_config.json（`soft_cap_kW, pcc_limit_kW, n_minus_1_margin_kW, export_allowed` 等）。:contentReference[oaicite:4]{index=4}  
> - 并网边界对齐：bess_master.json（仅用于统一 `export_allowed` 等站端并网口径）。:contentReference[oaicite:5]{index=5}

---

## 1) 业务目标与范围（原文保留）

* **目标**：在不降低堆场吞吐（SLA）与作业安全的前提下，最小化 RTG/RMG 的空转/待机能耗与峰段功率，并降低温升/老化与启停抖动成本。可与PCC/需量、电价/碳因子、DR/限电联动。
* **对象**：
  * **RTG（橡胶轮胎场桥）**：柴油-电/混动/锂电改造版本；
  * **RMG（轮胎/轨道龙门）**：一般为并网电驱，部分带超级电容/小电池回收能量。
* **作业构成**：起升（Hoist）/小车（Trolley）/大车（Gantry）/抗摆（Anti-sway）/空转等待；队列受堆场任务分布、AGV/集卡到达、TOS 调度影响。

**补充与落地细化：**

* **优先级**：`安全/吞吐 ≥ 并网/噪声合规 > 需量/DR > 能耗/碳 > 温升/寿命成本`。
* **适配差异**：RTG 增加 `idle_fuel_lph, start_stop_wear_cost, dpf_regen_state`；RMG 增加并网 `pf_min, thd_max, export_allowed` 与 `regen_buffer_kWh`。
* **电力侧联动**：对齐 `soft_cap_kW/pcc_limit_kW/n_minus_1_margin_kW/export_allowed`；DR 事件来自 `/mnt/data/dr_events.json`，需按 `required_reduction_kw/target_blocks` 做局部/全域减载。

---

## 2) 控制时标与滚动策略（原文保留）

* **步长**：Δt = **5–10 分钟**（推荐 5 分钟以更好捕捉作业节拍与空转段）。
* **滚动窗口**：H = **6–12 小时**（覆盖班次/夜间与峰段）。
* **模式切换**：RL 为主优化待机与功率模式；异常或 OOD（队列预测失真/传感失联）→ 规则/曲线兜底。
* **耦合接口**：与 TOS（任务分配/优先级/队列）、电力侧（PCC/需量/DR）双向订阅。

**补充与落地细化：**

* **需量口径**：Δt=5 min 时，`rolling_mean_15(power_kW)` 用 3 步滑动平均（线性插值到 15min 口径）；超过 `soft_cap_kW` 触发峰罚保护。:contentReference[oaicite:7]{index=7}
* **班次切换保护**：交接前后 15–30 min 收紧残差带 `δ_res(power)` ×0.5，临时抬高 `β（SLA违约）`。
* **DR/限电影响**：DR 窗口内将 `required_reduction_kw` 转换为 block/全域目标，首选“**合并队列 + 关停低负载机**”，剩余机降档至 `ecoL2/L3`。:contentReference[oaicite:8]{index=8}

---

## 3) 数据与文件（最小可训练/仿真集，≤10 个｜原文保留）

**时间戳 UTC；单位：功率 kW、能量 kWh、速度/频率 %、温度 °C；步长与训练一致（5–10 分钟）。**

最小文件集（建议放置 `/mnt/data`，字段**同名同义**可直接替换）：
1) `crane_telemetry.csv`：`ts, crane_id, power_kW, speed_pct, temp_motor_C, temp_inverter_C, mode, start_stop_event, regen_kWh`
2) `job_events.csv`：`ts, crane_id, boxes_done, cycle_time_s, distance_m, queue_len, wait_time_s`
3) `queue_forecast.csv`：`ts, block_id, queue_len_p50, queue_len_p90, arrival_rate_p50`
4) `cranes_master.csv`：`crane_id, type{RTG|RMG}, block_id, pf_min, thd_max, idle_auto_off_min, min_on_min, min_off_min`
5) `yard_blocks.csv`：`block_id, near_residential{0/1}, quiet_hours_local, night_noise_limit_dBA`
6) `grid_meter.csv`：`ts, pcc_kw`（PCC/feeder）  
7) `market_price.csv`：`ts, price_yuan_per_kWh`  
8) `grid_ef.csv`：`ts, ef_kg_per_kWh`
9) `dr_events.json`：见上（事件清单）:contentReference[oaicite:9]{index=9}
10) `demand_window_config.json`：见上（需量/并网窗口）:contentReference[oaicite:10]{index=10}

> **数据质量**：统一列名候选；多口径时间戳（ISO/epoch/有无时区）自动识别；读失败时采用**经验曲线**兜底（夜间合并、峰段降档）。

---

## 4) 任务建模（CMDP｜原文保留 + 细化）

### 4.1 状态 S（孪生内部｜原文保留）

* **设备态**：`state, mode, temp_motor/inverter, start/stop 计时器, speed_setpoints(历史), regen_buffer`
* **功率/能耗**：`power_kW, energy_kWh, regen_kWh`
* **作业态**：`queue_len, job_arrival_rate, task_eta_min, boxes_per_hr, distance_to_job`
* **环境/外生**：`price, EF, DR 标志, night_noise_limit, near_residential`
* **合规模块**：`min_on/min_off, idle_auto_off_min, accel/jerk_limit, temp_redline, SLA_throughput 目标`

**补充与落地细化：**
* `thermal_margin = temp_redline - max(temp_motor, temp_inverter)`；定义阈值 `θ1/θ0`（预警/红线）。
* `boxes_target_15m, wait_time_max` 由 TOS/班次表下发（block 级），关键 block/时段可×2 权重。
* 跨机特征：同 block `active_cranes, mode_share{normal,ecoL1,L2,L3}`。

### 4.2 观测 O（代理可见｜原文保留）

* 实时传感 + 预测分位（队列/价/EF/DR） + 近窗历史摘要（EMA/分位）

**补充与落地细化：**
* 观测拼接：`[soc_like=None, power_kW_norm, queue_len_p50/p90, price_p50/p90, ef_p50/p90, thermal_margin, active_cranes, mode_share, DR_active, P_roll15_resid]`。
* 噪声指示：`near_residential, in_quiet_hours` 用于屏蔽启停频次与限速。

### 4.3 动作 A（参数化混合｜原文保留）

* **离散**：`mode ∈ {normal, ecoL1, ecoL2, ecoL3}`
* **连续**：
  * `idle_timeout_min ∈ [0, idle_auto_off_min]`
  * `hoist/trolley/gantry_power_limit_pct ∈ [L_min, 100%]`
  * `regen_usage_ratio ∈ [0,1]` 或 `battery_P ∈ [−Pmax,+Pmax]`（如有）

**补充与落地细化：**
* 残差幅度默认：`Δpower_limit ≤ ±10%`，`Δidle_timeout ≤ ±2–5 min`；`near_residential=1` 且 `quiet_hours` → 残差带减半。
* Eco 档映射（附录 A）：速度/功率/加速度/抗摆参数一并调整；`ecoL3` 仅在 `queue_len ≤ q_lo` 与 `thermal_margin ≥ θ1` 时允许。

### 4.4 输出（下发｜原文保留）

* 驱动/PLC 设定 + 抗摆参数微调 + 软启动/斜坡；TOS 协同建议（非强制）。

**补充与落地细化：**
* 点表（最小集）：`mode_cmd, power_limit_cmd{hoist,trolley,gantry}, idle_timeout_cmd, anti_sway_param_cmd, write_enable, nonce, qos_flag`；反馈 `mode_fb, power_kW, speed%, temp_motor/inverter, start_stop_event, regen_kWh`（附录 C）。

### 4.5 约束与屏蔽（硬｜原文保留）

* **吞吐/SLA**：每 15/60 分钟的最小箱量或最大等待；
* **设备保护**：`min_on/min_off、accel/jerk_limit、温度 < redline、软启动/刹车斜坡`；
* **安全/抗摆**：摆幅阈值触发速度/加速度上限；
* **夜间噪声**：指定 block 时段内速度/启停频次受限；
* **电力约束（可选）**：`PCC/feeder` 上限、DR 减载目标；
* **回收/电池**：`SOC ∈ [soc_min, soc_max]`、`|ΔP| ≤ ramp`。

**补充与落地细化（屏蔽伪码）**：
