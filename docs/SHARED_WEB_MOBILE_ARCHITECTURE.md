# Web / 移动端共享后端架构

Web 与 Flutter 移动端是同一套“数字孪生 AI 港口智能决策系统”的两个前台，
共同连接 `port-dt-multi` FastAPI 服务。移动端不再以自身仓库中的实验服务
充当系统后端。

```text
公开输入与工程派生数据
          |
          v
port-dt-multi FastAPI
  |- 数字孪生与固定业务基准
  |- 10 RL（含分布式/时序/信赖域/无梯度）+ MPC + FCFS
  |- 模型登记与留出集评测
  |- 移动端契约 /api/mobile/*
  |- 两人确认的南向执行边界 /api/actuators/*
          |
          +------------------+
          |                  |
          v                  v
Web 决策与推演前台       Flutter 移动态势与审批前台
```

## 移动端稳定契约

| 接口 | 作用 | 证据边界 |
|---|---|---|
| `GET /api/mobile/status` | 后端身份、算法清单、业务与移动闭环证据 | 业务结果与工作流结果分别标注 |
| `GET /api/mobile/situation` | 共享证据状态与留出集逐日序列 | 默认不是实时港口态势 |
| `GET /api/mobile/alerts` | 系统证据边界、审计异常 | 未接实时源时不生成生产告警 |
| `WS /api/mobile/alerts/ws` | 系统告警与服务心跳 | 心跳不冒充业务告警 |
| `GET /api/mobile/strategy/candidates` | 固定业务候选与已登记模型 | 只读取已验证报告/产物 |
| `POST /api/mobile/strategy/decisions` | 人工表态 | 强制 `Idempotency-Key`，默认只干跑记录 |
| `GET /api/mobile/strategy/decisions/{id}` | 服务端执行回执 | 不由客户端计时器伪造 |
| `POST /api/mobile/strategy/replan` | 登记重规划审阅申请 | 不生产下发 |
| `POST /api/mobile/audit/events` | 上传移动端审计摘要 | 服务器追加 SHA-256 前向链 |
| `GET /api/mobile/audit/verify` | 校验审计链 | 返回首个无效序号 |

生产动作不从 `/api/mobile/strategy/decisions` 直达设备。南向控制只能经过
`/api/actuators/*` 的白名单、约束检查、幂等、异人确认与独立通道门禁。

## 两类指标不能混写

- `泊位 +7.45 个百分点 / 待泊 -16.94% / 成本 -11.80%` 是整套系统在固定数字孪生留出集上的业务情景结果，
  不是移动端单独创造的收益。
- 移动端的 `500` 项固定集成操作衡量幂等、越权阻断、回执和审计完整性，
  不是业务收益、网络 SLA 或现场可用性。
