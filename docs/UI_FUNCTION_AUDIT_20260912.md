# 2026-09-12 UI 功能核验

## 自动化验证范围

本节只记录源码、VM/DOM 合同、隔离 API 与本地发布门槛的结果。VM 使用实际页面控制器与可控 DOM/Promise/HTTP fixture，不等同真实浏览器点击。真实 CUA 验收应单独记录页面、控件、前置条件与最终画面/接口回执；不得从本节推导“全部按钮已在浏览器通过”。

| 检查 | 最终结果 | 口径 |
| --- | --- | --- |
| `python -m unittest discover -s tests -v` | 424 tests，139.819 s，OK | 最终代码冻结后完整重跑，包含全部 13 个 JS VM 脚本 |
| `tests/test_javascript_contracts.py` | PASS | 发现 `tests/js/*.cjs` / `*.test.js`；缺 Node 或脚本失败均使测试失败 |
| 静态按钮定义扫描 | 289 / 289 绑定，0 未解析 | 静态定义；动态列表实例与真实交互另验 |
| 页面内联 JavaScript 语法 | 49 / 49 PASS | 主页面 47，联动页 1，副驾页 1；仅语法 |
| `compileall -q app scripts tests` | PASS | Python 语法与导入编译 |
| 固定业务 KPI `--verify` | PASS，8,760 test rows | 保留既有基准定义，不代表本轮新 RL 收益 |
| `scripts.release_check` | PASS | 当前工作树发布检查；不是远端 CI 结果 |
| `scripts/public_privacy_scan.py` | PASS | 可发布文件及 ZIP/NPZ 内容、序列化 JSON/base64 深扫 |
| V8 / legacy 公共模型 bundle `--verify` | 两项 PASS | 安全模型映射、摘要与原身份验证 |
| `scripts.rl_smoke_test --steps 64` | 12 / 12 PASS | 10 RL 算法及 MPC、FCFS，临时确定性数据；不构成业务训练或准入证据 |

本机使用 Python 3.12 与 Node 24；通用入口为 `python -m unittest discover -s tests -v`。Node 不在 PATH 时，应将 `NODE_BINARY` 指向本机 Node 可执行文件。CI 显式要求 Node，并通过同一 unittest 入口运行 VM 合同；隐私扫描与两种公共模型 bundle 校验也已纳入 CI。

当前以下 13 个 VM 脚本自动执行：`compliance_result_scope_contract.cjs`、`external_detail_loading_contract.cjs`、`governance_detail.cjs`、`home_evidence_rules_contract.cjs`、`integration_training_contract.cjs`、`linkage_state_guards.test.js`、`main_module_state_guards.cjs`、`mlops_evidence_contract.cjs`、`module_lifecycle.cjs`、`rl_pages_contract.cjs`、`simulation_export_contract.cjs`、`standalone_navigation_vm.cjs`、`story_availability_contract.cjs`。

既有 `tests/js/standalone_navigation.cjs` 保持原样，属于需要 Playwright 与浏览器的可选测试，本轮未执行，也不计入 13 个 VM 脚本。新增导航 VM 合同验证实际控制器的 35 条菜单路由、query/hash 保留、提前请求、缓存、三次失败回退、离开取消、恢复重试及坏缓存恢复。浏览器导航结果依 CUA 的独立记录。

## 已修复的联动与副驾合同

- 联动页将“所选训练档案”与“全局最新任务”分开。任务必须含匹配的持久化 `config.module_target` / `target` 才能归属于所选档案；不匹配或字段缺失时，不显示该档案训练完成。全局任务另列 job ID、数据集、策略；读取失败清除旧完成状态。
- 高级窗保留 66 项历史档案，仅开放训练器与当前算法实际消费的参数。其余字段只读并标明未传入当前学习器；网络显示真实固定 `64 × 64`，SAC/TQC 熵调节仍由现有训练器自动处理。数据集及随机种子在 RL 面板最终复核；档案目标不等于训练环境，也不会自动启动 V8。
- 修正 `trainMaxRamp` 的默认数值与 HTML step 基准不一致。5 套默认档案的全部数值约束均有效；不受支持的只读字段不阻挡 Apply，支持字段无效仍阻止提交。保存仅表示档案保存，未声称训练已执行。
- 副驾请求失败清除答案、SOP、审计、交接与旧上下文证明；显示明确失败原因和 `Context INVALID`，恢复生成按钮。
- 交接留痕由原生 `window.confirm` 改为可访问 HTML `dialog`，仍需“打开审阅 → 最终确认”两步，显示值班员、班次、上下文 SHA、交接 SHA 与本地边界。取消/Escape 不提交；确认前上下文、操作员或交接包变化均拒绝。VM 验证确认前 0 POST、取消 0 POST、确认 1 POST、重复确认不重复提交。
- 主页面异步合同覆盖 Monitoring 时间/灵敏度参数、OpsX/TwinLab/治理/MLOps 详情加载和错误重试、实时资产切换、3D 图层切换及返回后的单实例定时器。RL 建议读取真实 runtime frame；没有当前可用状态时不得用硬编码状态伪造建议。
- 首页状态规则新增 22 项 VM 合同：未知风险、缺失审批待办或不可用计数不再当作低风险/零待办；存在已知高风险时继续显示该风险并说明缺失证据。
- OpenAPI 地址复制仅在剪贴板 Promise 成功后显示成功。拒绝权限或缺失 clipboard API 时明确说明未写入，并展示可手动复制的完整地址；新增成功、拒绝及 API 缺失的 VM 回归。
- 仿真图导出必须有当前所选策略的有效基线/策略后曲线且绘制成功；未评测、重选、失败或旧请求迟到均不能导出网格空图。图片保留基线/策略图例，增加不透明背景与策略 ID / kW 标题。左侧概览导出读取真实资产下拉值与视图标题，缺预测实绩对齐数据明确待接入，不沿用失效 KPI。

静态扫描器新增有限键集合的 click 绑定识别，修正数组 / `Object.keys` 循环造成的 15 个误报；负例确保“仅出现集合”“非成员按钮”“非 click 赋值”“未定义集合”不会被该规则判为已绑定。原有行为断言未删除。

## 留存证据与边界

最终冻结轮日志位于 `.codex_artifacts/ui_full_audit_20260912/linkage/final_after_all_ui_fixes/`：`unittest.log`、`compileall.log`、`release_check.log`、`public_privacy_scan.log`、`business_verify.log`、`static_button_binding.log`、`inline_script_syntax.json`。最终完整回归为 424 tests / 139.819 s / OK，包含全部 13 个 VM 脚本；没有把可选 Playwright 计入。此前第一轮 422 tests 的失败日志与中间复测均在上级目录保留。

本次最终完整回归已覆盖副驾错误证明栏、剪贴板失败反馈、两类图像导出和首页未知状态规则。此前各阶段定向日志仍保留，最终证据以 `final_after_all_ui_fixes/verification.json` 为准。测试前后 580 个冻结代码、配置和 CI 文件均按 SHA-256 比对，未发生变化。另仅更新文档，并对已封存 Matplotlib SVG 精确追加 `.gitattributes` 行尾空格规则；原 SVG 字节及 SHA 不改，不扩大到其他文件。V8 原模型验证与已封存重放没有重复执行，也没有新增 test/forward 评测。

逐控件浏览器清单位于上级 `linkage/` 目录 `control_inventory_and_click_plan.json`、`priority_click_scenarios.json`、`main_module_click_plan.json`。这些清单仅描述操作步骤，实际完成状态以独立浏览器记录为准。API 只读/动作预览回执与 VM 结果也不冒充真实点击。

公共模型隔离检出验证见 [公共模型验证记录](../evidence/public_models/verification_20260912/report.json)。该记录另证实 SAC/TD3 r5 的 selected 与 final 60k 验证窗口指标可从安全模型重放；未打开 test/forward，不是正式准入。V8 无正式 champion，诊断结果不能成为当前策略或生产权限。

本节仅记录本地验证结果；远端提交、干净克隆、CI 与发布状态见发布验收记录。本节未执行本地容器构建。没有通过副驾发送外部消息或操作真实生产设备。旧策略与失败候选保留；历史用户评测覆盖缺口及实际封存边界见浏览器审计记录，不宣称缺失 raw SHA 已恢复。

## 真实浏览器验收

实际 CUA 点击、最终接口回执、导出文件与已知限制参见 [真实浏览器验收](UI_BROWSER_AUDIT_20260912.md)。该记录由主任务依据浏览器实测更新；本节的自动化结果不替代它，也不推导现场接入或生产准入。
