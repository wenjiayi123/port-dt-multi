# 岸电储能 V8 复现与证据说明

本目录的 V8 工作专门复核旧岸电储能策略增碳问题。根因和物理可行界限见 [根因审计](SHORE_BESS_V8_ROOT_CAUSE.md)。训练中的候选、规则基线和 LP 上界分别保存，不能互相替代。所有策略都仅用于公开工程场景回放，尚无生产调度权限。

## 环境与训练

在仓库根目录运行，使用项目 Python 环境。模块方式运行可保证仓库导入路径正确：

```bash
.venv312/bin/python -m scripts.train_shore_bess_v8 \
  --algorithm sac --steps 60000 --block 10000 --seeds 912 \
  --carbon-price 12 --pilot

.venv312/bin/python -m scripts.train_shore_bess_v8 \
  --algorithm td3 --steps 60000 --block 10000 --seeds 912 \
  --carbon-price 12 --pilot

# 多步回报诊断变体，不能与标准 TD3 混称。
.venv312/bin/python -m scripts.train_shore_bess_v8 \
  --algorithm td3 --n-step 24 --steps 60000 --block 10000 --seeds 912 \
  --carbon-price 12 --pilot
```

默认生成新的 UTC 时间戳运行目录；指定 `--run-id` 时，该目录已存在就拒绝覆盖。`--pilot` 只打开 2025 年 5—7 月的固定验证窗口，不更新正式指针、不加载 2026 前向数据。正式运行必须至少三个独立种子、三个训练检查点；每个种子的 checkpoint 与整体种子选择在加载测试/前向基准前冻结。

准确的数据访问边界是：历史数据文件包含2024—2025整张表，加载与数据质量检查会读取全表；梯度、归一化及参数选择只使用规定的训练/验证切片。pilot 中2025年8—12月不评估、不用于选模；独立的2026前向文件不会加载。该区别保存在新运行的 `manifest.access_protocol_details`，不能把“测试区间未评估”说成“整个历史CSV未读取”。

SAC 与 TD3 共用 31 维状态、2 维连续动作、128×128 ReLU 网络、相同设备与服务限制。环境保存实际执行功率和柔性任务 FIFO 台账。动作坐标只根据当前物理状态确定可行区间，不调用价格/碳规则或经济教师。零动作有精确可达区间，减少无意义的小幅充放电。

碳系数 12 元/kg 是优化目标中的约束乘子，**不是碳交易收入或实际碳价**。奖励逐步累计真实场景电费、退化成本、周期最大需量费与碳差值。库存势函数和训练集阈值需量势函数在 `gamma=1`、周期结束势函数归零时望远镜抵消，不改变整周期目标或报告账单。

V3.2 JSON 仅复用物理设备与场景参数。其历史 `training` 和 `reward_weights` 字段不控制 V8；实际训练配置由每次运行的顶层 `algorithm_parameters`、真实模型反序列化参数及网络结构共同证明。

`--n-step 24` 仅支持本实验的 `gamma=1` TD3。目标使用已观察行为轨迹的多步奖励和端点 bootstrap，没有重要性修正，因此是明确的 off-policy 近似变体。真实终止不 bootstrap，时间截断仍 bootstrap；短尾部不伪装成完整24步。checkpoint记录已观察/已存储/待存储/丢弃的数量守恒、实际抽样跨度直方图、终止/截断端点及各优化器计数。训练结束的补存动作不运行优化器，补存样本不声称都已用于学习。

## 数据与统计

| 分区 | UTC 时间 | 小时数 | 用途 |
|---|---|---:|---|
| 训练 | 2024-01—2025-04 | 11,664 | 梯度、归一化尺度和规则统计 |
| 验证 | 2025-05—07 | 2,208 | 参数探索、checkpoint/种子选择 |
| 重复测试 | 2025-08—12 | 3,672 | 选择冻结后评估 |
| 前向基准 | 2026-01—05 | 3,624 | 选择冻结后评估 |

后两段已被历史版本使用，不能称为全新盲测。小时负荷、电价和碳因子是公开数据校准的工程特征，不是港口电表逐时实测。

评估使用不重叠的 168 小时窗口，各策略配对相同起点，并要求末端 SOC 与未完成任务恢复。报告同时保存周窗口 bootstrap 和官方来源时期的 cluster bootstrap。每年 1、2 月共享官方累计锚，合并为一个时期；跨时期的周按窗口起始时期归组，报告明确这一近似。来源簇数量较少，名义 95% 区间不代表现场置信保障。

## 独立验证与可视化

已完成的三个独立 pilot 可以复用，不需要重新训练其权重：

```bash
.venv312/bin/python -m scripts.admit_shore_bess_v8 --run-ids RUN1,RUN2,RUN3
```

该工具要求三个来源在算法、预算、奖励、动作、数据、源码和运行库版本上严格一致。它重新评估每一个 checkpoint 的全部13个验证窗口、最后三个检查点的稳定性并重新选模；全体通过后才访问重复测试/前向基准。若验证失败，产出 `VALIDATION_REJECTED`，不打开后续基准。正式报告同时记录被引用的真实训练步数和**本次新增训练0步**，避免重复计数；复制模型的字节SHA保持不变。

```bash
.venv312/bin/python -m scripts.verify_shore_bess_v8 --report evidence/v8/shore_bess/runs/RUN_ID/report.json
.venv312/bin/python -m scripts.verify_shore_bess_v8 --report evidence/v8/shore_bess/runs/RUN_ID/report.json --replay
.venv312/bin/python -m scripts.evaluate_shore_bess_v8_comparators --report evidence/v8/shore_bess/runs/RUN_ID/report.json
.venv312/bin/python -m scripts.plot_shore_bess_v8 --reports evidence/v8/shore_bess/runs/RUN_ID/report.json
```

`RUN_ID` 替换为实际运行目录。比较器需要正式报告；重放要求当前源码与该运行冻结源码一致。历史证据仍可进行哈希和统计核验，复现旧策略则需对应源码快照。

完整证据包括数据/源码 SHA-256、配置、种子、初始与最终模型摘要、保存模型、逐 checkpoint 指标与置信区间、实际 PyTorch 优化器更新次数，以及失败门槛。SAC actor/critic 各更新一次会计作两次 optimizer 调用；TD3 actor 延迟更新，因此不能直接用总调用次数宣称公平训练预算。算法对照以同环境步数和 critic 更新预算为主，并分开披露组件次数。

图表来自保存的 checkpoint 数值，不平滑、不补造采样点；曲线和审计文件放在运行目录以外，避免破坏封存产物的完整清单。完整性核验通过不等于业务准入通过。失败候选不能写入 champion。
