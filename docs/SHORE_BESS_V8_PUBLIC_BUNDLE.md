# V8 公开模型与审计副本

公开目录保留 **77 个原始检查点身份、66 份按原 SHA 去重的模型 ZIP**。原训练 ZIP、报告、曲线、账本、旧失败候选和所有原 SHA 均不改写。此发布不新增训练，不改变准入或冠军；SAC/TD3 r5 仍是未达业务门槛的单种子诊断。

## 模型身份与加载

`evidence/v8/shore_bess/public_models/manifest.json` 将 `source_paths + training_model_sha256` 映射到 `public_model_path + public_model_sha256`。原 SHA 是历史训练产物的身份，公开 ZIP 使用自己的真实 SHA，二者不会混称同一文件。

原 SB3 ZIP 的 `data` 中，cloudpickle 缓存的学习率调度器包含本机源码路径。导出器仅删除可由原标量参数重建的学习率、DQN 探索率调度器，并将已验证为常数的 PPO clip 调度器写成相同标量。所有其他元数据的规范化 SHA、网络和优化器成员逐字节保持一致。每份独立原 ZIP 与公开副本均实际通过 SB3 CPU 加载、64 个固定输入的确定动作比较、101 个进度点调度值比较以及环境步数/优化器计数/组件计数/n-step 快照比较；MaskablePPO 另比较 64 组动作掩码。

`app/services/rl_model/shore_bess/v8_public_artifacts.py` 校验映射、公开 SHA、成员清单、非 data 成员 SHA 与受保护元数据；发现原 ZIP 存在但原 SHA 已改变时拒绝加载。证据适配器和独立验证器均通过此映射加载，显示的历史 `model_sha256` 保持原值，并另报实际 `loaded_model_sha256`。干净检出无需本地原 ZIP。

```bash
python scripts/export_public_shore_bess_v8.py --verify
python scripts/verify_shore_bess_v8.py --report evidence/v8/shore_bess/runs/shore-bess-v8-td3-pilot-20260912-seed912-c16-physicalfix-r5/report.json --replay
python scripts/verify_shore_bess_v8.py --report evidence/v8/shore_bess/runs/shore-bess-v8-sac-pilot-20260912-seed912-c16-physicalfix-r5/report.json --replay
```

重放使用训练段拟合的归一化和相符的冻结环境，复核已经记录的 selected 与最终 60k 验证窗口及零动作基线，不打开新的 test/forward 评估。`INTEGRITY:PASS` 证明记录一致，不代表策略准入。

带本地原始 ZIP 的维护工作区可运行 `python scripts/export_public_shore_bess_v8.py` 重做等价导出。导出器遇到同名不同字节的公开副本会停止；不会覆盖历史训练产物。

## 历史审计映射与边界

`evidence/v8/shore_bess/public_audit_sources/manifest.json` 保存旧路径、原 SHA 与内容寻址副本的逐字节映射，覆盖原来被 `.codex_artifacts` 忽略的 13 份脚本/JSON，以及相关历史源码和证据快照。旧审计仍记录当时的原路径；当前验证器解析其映射，不修改旧报告来迁就发布目录。

这些副本是可检验的历史来源材料。不能把将旧脚本放入内容寻址目录等同于其可直接原样执行；旧脚本可能依赖当时的目录结构、原 ZIP 字节和原日志。上面的独立验证命令是公开副本支持的实际验证入口。

两份较早的源码版本未封存且未找到匹配字节，清单明确列为 `unavailable_historical_source_snapshots`：

- `train-feasibility-20260912-v8.json` 使用的旧 `audit_shore_bess_v8.py`，SHA `ef013e46…`；后继 `train-feasibility-20260912-v8-r1.json` 源码完整。
- `td3-demandphi-r2-critic-ranking-20260912.json` 使用的旧 `audit_shore_bess_v8_critic.py`，SHA `08d9bbb6…`；后继 `…-v2.json` 源码完整。

原证据保留，不用相近版本冒充缺失版本。因此发布检查会明确返回 `all_historical_audits_reproducible=false`。这两项不属于最终 r5 训练/业务重放证据链；当前最终报告、完整训练账本与验证源码均可核验。
