# 强化学习引擎

## 目标与边界

引擎优化同一个港口运营环境中的储能功率、服务强度、柔性负荷、泊位优先级和堆场流量。当前环境用于离线研究与接港前集成，不直接下发生产设备。

## V3 十二个控制器

| ID | 类型 | 实现 | v1 / v2 动作空间 |
|---|---|---|---|
| `sac` | RL | `stable_baselines3.SAC` | 连续 3 维 / 5 维 |
| `ppo` | RL | `stable_baselines3.PPO` | 连续 3 维 / 5 维 |
| `td3` | RL | `stable_baselines3.TD3` | 连续 3 维 / 5 维 |
| `dqn` | RL | `stable_baselines3.DQN` | 45 / 405 个显式离散动作 |
| `a2c` | RL | `stable_baselines3.A2C` | 连续 3 维 / 5 维 |
| `tqc` | RL | `sb3_contrib.TQC` | 连续 3 维 / 5 维 |
| `qrdqn` | RL | `sb3_contrib.QRDQN` | 45 / 405 个显式离散动作 |
| `trpo` | RL | `sb3_contrib.TRPO` | 连续 3 维 / 5 维 |
| `recurrent_ppo` | RL | `sb3_contrib.RecurrentPPO` | LSTM 连续 3 维 / 5 维 |
| `ars` | RL | `sb3_contrib.ARS` | 连续 3 维 / 5 维 |
| `mpc` | 控制 | `scipy.optimize.minimize` | 3 维约束控制 / 5 维输出且新增调度量保持中性 |
| `fcfs` | 规则 | `port_dt.FCFSNeutralPolicy` | 3 维 / 5 维中性动作；环境按 FCFS 处理队列 |

DQN 与 QR-DQN 使用储能、服务强度和柔性负荷的有限动作格点；其余 RL 算法使用相同控制量的连续版本。新增方法覆盖分位数价值学习、信赖域优化、LSTM 时序记忆和无梯度随机搜索。MPC 使用滚动时域近似模型；FCFS 保持储能、柔性负荷、泊位优先级和堆场流量中性。两者都不训练神经网络。

## 防止数据泄漏

历史数据与运行仍可由 `PortDataset.split()` 按 80/20 时间顺序读取。V3 数据显式调用 `split_three_way()`，按 70/10/20 划分训练、验证和最终盲测；归一化最小值/最大值只从训练段拟合：

```text
V3 完整时间序列 = [ chronological train | validation | blind test ]
```

训练环境只接收 `train_slice`。V3 由独立的 `evaluate_split_evidence(..., split_name="validation")` 在验证段比较算法；选型完成后，最终评测环境才接收 `test_slice`，盲测指标不参与冠军选择。不进行随机全局打乱，也不从验证/盲测段计算训练奖励。旧 manifest 不含 `validation_ratio` 时保持原 80/20 语义，避免破坏历史证据。

## 渲染隔离

- `training=True` 与 `record_trace=True` 同时出现会直接抛错。
- 训练环境 `render()` 总是抛错并记录调用次数。
- 训练产物清单记录 `render_calls_during_training`，正常值必须是 0。
- 测试环境才允许 `record_trace=True`，第一条测试 episode 的轨迹写入 `evaluation_trajectory.json`。

## 环境版本、观测与动作

`port_ops_v1` 保留13维观测和3维动作，用于读取既有模型与证据。`port_ops_v2` 使用37维观测：13个基础状态、12个国际港口因素以及12个因素可用性掩码；动作扩展为BESS功率、服务强度、柔性负荷、泊位优先级和堆场流量5维建议量。`port_ops_v3` 保持相同接口，但进一步把服务强度、泊位优先和堆场流量与作业电负荷显式耦合，避免策略通过“免费增产”制造虚假吞吐优势；环境版本写入每个 manifest，跨版本结果禁止进入同一优势门。

缺失因素以中性值和零掩码进入网络，避免把“缺数据”混同为“观测值为零”。DQN的v2动作空间是显式有限格点，连续算法使用相同控制量的连续版本。场景包中的校准状态不是 `site_calibrated_approved` 时，安全评估不会给出现场声明资格。

## 奖励与约束

每步奖励由以下明确归一化分量构成：

- 电能成本；
- 碳排；
- 超需量峰值；
- SOC 与爬坡安全惩罚；
- 作业队列延迟；
- 储能吞吐退化成本。

权重会归一化。环境同时约束储能 SOC（12%–88%）、功率、电网余量、期末 SOC 可达性和服务强度。服务强度 0.75–1.25 表示可调度资源池相对参考班组的启用水平，不表示设备超频。当前参数是工程初值，接港时必须按设备铭牌、EMS/BMS 限值、合同需量和作业 SLA 校准。

V3 评测除吞吐、延误、成本、碳排、峰值和违规率外，还输出单位吞吐成本、单位吞吐碳强度、单位吞吐能耗、作业完成率、队列峰值/期末队列、储能等效循环、动作安全投影率、气象封锁率、资源可用因子和期末 SOC 误差。前端算法卡片可逐项打开均值、95% bootstrap 区间、种子、步数、作业 ID 和奖励权重。

## 后端进度与产物

进度来自 Stable-Baselines3 `BaseCallback.num_timesteps`，不是定时器。回调将指标追加到 `metrics.jsonl`，状态同步写入 `status.json`。

每个任务目录包含：

- `config.json`：完整输入配置；
- `status.json`：后端状态、进度、指标、日志；
- `metrics.jsonl`：真实优化器回调历史；
- `monitor.csv`：Gymnasium episode 监控；
- `model.zip`：RL 模型；MPC/FCFS 无模型文件；
- `manifest.json`：实现、数据哈希、切分、种子、步数与训练渲染次数；
- `evaluation.json`：留出集聚合指标；
- `evaluation_trajectory.json`：测试回放轨迹。

`data/rl/runs` 是运行时目录，默认不提交 Git。经模型哈希和数据哈希校验的安全摘要由 `scripts.export_rl_evidence` 写入 `evidence/rl`，供全新 clone 复核；模型二进制仍需本地重跑生成。

## 主要接口

| 方法 | 路径 | 用途 |
|---|---|---|
| GET | `/api/rl/engine/capabilities` | 运行时、算法和数据集 |
| GET | `/api/rl/datasets` | 可训练数据集 |
| GET | `/api/rl/port-profiles` | 可换港场景包、目标权重和校准状态 |
| POST | `/api/rl/datasets/upload` | CSV + 字段映射导入 |
| POST | `/api/rl/train/start` | 启动真实训练/创建 MPC 或 FCFS 基线 |
| GET | `/api/rl/train/status` | 后端拥有的当前状态 |
| GET | `/api/rl/train/{job_id}/history` | 优化器真实历史 |
| POST | `/api/rl/train/{job_id}/control` | pause/resume/cancel |
| POST | `/api/rl/train/{job_id}/evaluate` | 留出集测试与轨迹生成 |
| POST | `/api/rl/train/{job_id}/predict` | 已训练策略/确定性基线推理，不渲染、不下发 |
| GET | `/api/rl/train/baselines` | 十二控制器已评测结果登记 |
| GET | `/api/rl/benchmarks/summary?dataset_id=...` | 同一数据集范围内的多种子比较门禁 |
| GET | `/api/v3/overview` | V3 算法、业务、证据和部署闸门事实源 |
| GET | `/api/v3/data-readiness` | 多港数据来源与现场替换清单 |
| GET | `/api/v3/algorithms/{algorithm_id}/evidence` | 单算法训练步数、种子、盲测指标和区间 |
| GET | `/api/v3/capabilities/{capability_id}` | 单业务域状态、动作、硬约束与现场验收指标 |

## 接港前验收

至少需要完成多随机种子对比、超参数冻结、数据漂移阈值、动作可行域联合验收、影子运行、故障注入、人工接管、回滚和生产数据权限审计。短步数 smoke test 只证明链路真实运行，不能证明策略优于基线。
