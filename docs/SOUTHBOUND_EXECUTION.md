# 南向执行安全契约

默认状态为 `fail_closed`。训练、评测和推理都不会自动下发设备命令。原 `/api/rl/dispatch` 和 `/api/exec/*` 生成式演示链路不再默认挂载。

## 现场配置

1. 复制 `config/actuators.example.json` 到不进入仓库的现场路径。
2. 替换资产 ID、白名单、路由、通道、每个资产+动作+参数的上下界。缺少约束的动作会被拒绝。
3. 由密钥管理器分别注入执行通道凭据、`PORT_DT_ADMIN_API_KEYS` 和独立的 `PORT_DT_SECOND_CHANNEL_TOKEN`。
4. 设置 `PORT_DT_ACTUATOR_CONFIG=/private/path/actuators.json`，完成影子演练后才把文件中 `enabled` 改为 `true`。

`GET /api/actuators/capabilities` 只返回脱敏状态，不暴露端点、凭据或本地证据路径。

## 指令生命周期

1. `POST /api/actuators/stage` 暂存人工命令，或使用 `POST /api/actuators/rl-stage` 由服务端重新执行指定模型推理和安全包络检查。
2. 网关检查开关、白名单、幂等键、资产路由和现场参数约束。通过后只返回 `PENDING`。
3. 与申请人不同的人员用管理员 API Key 和第二通道密钥调用 `POST /api/actuators/{command_id}/confirm`。密钥不写入证据包。
4. 证据包以原子写入记录请求、路由、模型/数据哈希、软件安全判定、现场约束、两位人员和执行结果。
5. 回滚同样需要管理员身份、审批人、理由和第二通道密钥；仅对已成功执行且底层支持回滚的通道有效。

## RL 暂存请求

`rl-stage` 需要 `job_id`、完整规范状态、现场资产/动作、`control_field` 以及现场参数名。后端不信任客户端传来的“已安全”标志，而是读取已登记的训练产物重新计算推理和安全包络。分布外状态或越界结果在生成工单前就会被拒绝。

这一软件链路不取代 PLC/BMS 联锁、急停、电气保护、变更工单和现场操作规程。
