# 当前发布树的模型隐私清理

本次处理公开 ZIP 的序列化元数据路径，不修改训练权重、历史报告、原 SHA 或策略准入结论，也不重写 Git 历史。

- V8：77 个原检查点身份映射到 66 份去重的公开 ZIP，详见 [V8 公开包说明](SHORE_BESS_V8_PUBLIC_BUNDLE.md)。
- 已发布旧模型：18 个原路径、17 份不同 ZIP，公开副本与原 SHA 清单位于 `evidence/public_models/legacy_v3_v6_20260912/`。覆盖 V3 runtime、Shore BESS、BESS energy 及相关已发布训练模型。
- 原 ZIP 本地保留并另有备份，当前 Git 树取消跟踪泄露路径的原件；历史提交仍保留原历史，不宣称已清除过去提交中的记录。

两个导出器共用元数据转换与等价验证实现。仅重建原参数决定的学习率/探索率调度器、将常数 PPO clip 调度器写为相同标量。所有非 `data` 成员逐字节相同；其他元数据规范化 SHA 相同；SB3 实际加载、确定性动作、调度值与计数器比较均通过。

旧报告的 `model_sha256` 继续表示原训练 ZIP。loader 按该身份解析公开副本，验证其真实公开 SHA、网络/优化器成员 SHA、不可变元数据与等价证据，另返回实际加载路径和 SHA。不会将公开副本的新 SHA 冒充原 SHA。原件存在却被修改时拒绝通过映射掩盖变化。

V3 runtime、Shore/BESS evidence、MLOps、模型注册表及 HTTP 推断/用户独立评测入口已接映射。冻结 trainer/environment/replay 文件未因发布迁移而修改。旧实验脚本若直接访问原 ZIP，仍属于原始工作区复现入口；公开验证和运行使用映射入口。

```bash
python scripts/export_public_shore_bess_v8.py --verify
python scripts/export_public_legacy_models.py --verify
python scripts/public_privacy_scan.py
python -m scripts.release_check
```

隐私扫描现在会打开所有待发布 ZIP/NPZ 的成员并检查 JSON/base64 元数据，但不执行反序列化代码或提取归档。检查覆盖本机账户路径、私钥与访问令牌。Docker 只允许三处经过验证的公开模型目录中的 ZIP；原始训练目录与 ZIP 仍被排除。V8 账本和快照通过 Git 属性保留原字节，避免换行转换破坏证据 SHA。

验证使用从待发布工作树构造的隔离检出，移除 95 个原 ZIP（77 个 V8、18 个旧模型）和本地审计工作目录。它是提交前的依赖闭包检验；远端 commit、真实干净 clone 和 GitHub CI 的验证在最终发布后另行记录。

提交前验证已通过：隔离树128个ZIP/NPZ深度扫描、release-check、V7的45模型/180万步/469500次更新证据核验、8个旧训练模型加载、6个运行接口，以及V8 SAC/TD3 r5已记录验证窗口重放。公开验证材料位于 `evidence/public_models/verification_20260912/`，主记录为 `report.json`。
