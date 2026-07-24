# 强化学习引擎

## 目标与边界

引擎优化同一个港口运营环境中的储能功率、服务强度和柔性负荷。当前环境用于离线研究与接港前集成，不直接下发生产设备。

## 五个基线

| ID | 类型 | 实现 | 动作空间 |
|---|---|---|---|
| `sac` | RL | `stable_baselines3.SAC` | 连续 3 维 |
| `ppo` | RL | `stable_baselines3.PPO` | 连续 3 维 |
| `td3` | RL | `stable_baselines3.TD3` | 连续 3 维 |
| `dqn` | RL | `stable_baselines3.DQN` | 45 个显式离散动作 |
| `mpc` | 控制 | `scipy.optimize.minimize` | 连续 3 维 |

DQN 使用储能、服务强度和柔性负荷的有限动作格点；其余 RL 算法使用相同控制量的连续版本。MPC 使用六步滚动时域近似模型，并由与环境相同的动作屏蔽器执行 SOC、电网余量与期末 SOC 可达性投影，不训练神经网络。

## 防止数据泄漏

`PortDataset.split()` 只做时间顺序切分。默认前 80% 是训练集，后 20% 是测试留出集；归一化最小值/最大值只从训练段拟合：

```text
完整时间序列 = [ chronological train | chronological test holdout ]
```

训练环境只接收 `train_slice`。测试环境只在显式调用评测接口后构造，并接收 `test_slice`。不进行随机全局打乱，也不从测试段计算训练奖励。

## 渲染隔离

- `training=True` 与 `record_trace=True` 同时出现会直接抛错。
- 训练环境 `render()` 总是抛错并记录调用次数。
- 训练产物清单记录 `render_calls_during_training`，正常值必须是 0。
- 测试环境才允许 `record_trace=True`，第一条测试 episode 的轨迹写入 `evaluation_trajectory.json`。

## 奖励与约束

每步奖励由以下明确归一化分量构成：

- 电能成本；
- 碳排；
- 超需量峰值；
- SOC 与爬坡安全惩罚；
- 作业队列延迟；
- 储能吞吐退化成本。

权重会归一化。环境同时约束储能 SOC（12%–88%）、功率、电网余量、期末 SOC 可达性和服务强度。服务强度 0.75–1.25 表示可调度资源池相对参考班组的启用水平，不表示设备超频。当前参数是工程初值，接港时必须按设备铭牌、EMS/BMS 限值、合同需量和作业 SLA 校准。

## 后端进度与产物

进度来自 Stable-Baselines3 `BaseCallback.num_timesteps`，不是定时器。回调将指标追加到 `metrics.jsonl`，状态同步写入 `status.json`。

每个任务目录包含：

- `config.json`：完整输入配置；
- `status.json`：后端状态、进度、指标、日志；
- `metrics.jsonl`：真实优化器回调历史；
- `monitor.csv`：Gymnasium episode 监控；
- `model.zip`：RL 模型；MPC 无模型文件；
- `manifest.json`：实现、数据哈希、切分、种子、步数与训练渲染次数；
- `evaluation.json`：留出集聚合指标；
- `evaluation_trajectory.json`：测试回放轨迹。

`data/rl/runs` 是运行时目录，默认不提交 Git。

## 主要接口

| 方法 | 路径 | 用途 |
|---|---|---|
| GET | `/api/rl/engine/capabilities` | 运行时、算法和数据集 |
| GET | `/api/rl/datasets` | 可训练数据集 |
| POST | `/api/rl/datasets/upload` | CSV + 字段映射导入 |
| POST | `/api/rl/train/start` | 启动真实训练/创建 MPC 基线 |
| GET | `/api/rl/train/status` | 后端拥有的当前状态 |
| GET | `/api/rl/train/{job_id}/history` | 优化器真实历史 |
| POST | `/api/rl/train/{job_id}/control` | pause/resume/cancel |
| POST | `/api/rl/train/{job_id}/evaluate` | 留出集测试与轨迹生成 |
| POST | `/api/rl/train/{job_id}/predict` | 已训练策略/MPC 的确定性控制推理，不渲染、不下发 |
| GET | `/api/rl/train/baselines` | 五算法已评测结果登记 |

## 接港前验收

至少需要完成多随机种子对比、超参数冻结、数据漂移阈值、动作可行域联合验收、影子运行、故障注入、人工接管、回滚和生产数据权限审计。短步数 smoke test 只证明链路真实运行，不能证明策略优于基线。
