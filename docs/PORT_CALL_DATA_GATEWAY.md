# 港口靠泊事件数据网关

## 目标

该网关把船公司、代理、码头操作系统和港口管理系统提供的靠泊事件，统一转换为可校验的事件包。事件进入数字孪生或决策上下文之前，必须通过来源治理、船舶身份、时区、重复记录、修订顺序、靠离泊时序和新鲜度门禁。

网关采用“预计、请求、计划、实际”的港口靠泊协同语义，但这里只声明内部互操作配置文件，不宣称已获得任何第三方标准一致性认证。

## 安全边界

- 未配置经授权的现场接口时，实时查询返回不可用，不生成模拟航次。
- `POST /api/v3/port-call/validate` 只校验并规范化输入，不保存、不下发。
- 合同样例通过只证明字段和门禁可运行，不证明已接入真实港口。
- 即使实时数据通过验证，`dispatch_allowed` 和 `production_authority` 仍为 `false`。
- 接口地址和令牌不会出现在来源状态接口中。

## 接口

| 方法 | 路径 | 用途 |
|---|---|---|
| `GET` | `/api/v3/port-call/readiness` | 查看字段合同、质量门和现场配置阻断项 |
| `POST` | `/api/v3/port-call/validate` | 校验一个靠泊事件数据包，不保存 |
| `GET` | `/api/v3/port-call/events` | 从经配置的只读现场网关拉取并校验事件；未配置时失败关闭 |

实时查询窗口最长三十一天，参数为 `start`、`end` 和五位港口位置代码 `port_unlocode`。

## 数据包合同

顶层字段：

```text
schema_version + site_id + source + events
```

来源字段：

```text
source_system + owner + license + timezone + retrieved_at + evidence_class
```

事件字段：

```text
event_id + port_call_id + vessel_name + vessel_imo/vessel_mmsi
+ port_unlocode + terminal_id/berth_id
+ event_type + event_phase + event_side + event_time
+ source_updated_at + revision + source_reference
```

事件类型包括到港、引航、拖轮、系泊、泊位、装卸、加油和离港。事件阶段包括预计、请求、计划和实际。泊位、系泊、装卸和加油事件必须携带码头或泊位位置。

## 现场配置

配置必须放在部署环境或密钥管理器中，不得提交到仓库：

| 环境变量 | 说明 |
|---|---|
| `PORT_DT_PORT_CALL_BASE_URL` | 现场只读网关根地址，必须使用安全传输协议 |
| `PORT_DT_PORT_CALL_PATH` | 靠泊事件路径，默认 `/api/port-calls/events` |
| `PORT_DT_PORT_CALL_AUTH_MODE` | `bearer` 或由上游网关完成鉴权的 `gateway` |
| `PORT_DT_PORT_CALL_TOKEN` | 令牌模式使用的密钥 |
| `PORT_DT_PORT_CALL_SITE_ID` | 经批准的现场标识 |
| `PORT_DT_PORT_CALL_OWNER` | 数据责任主体 |
| `PORT_DT_PORT_CALL_LICENSE` | 数据使用授权标识 |
| `PORT_DT_PORT_CALL_LIVE_ATTESTED` | 现场连接、来源与用途审核完成后才设为 `true` |
| `PORT_DT_PORT_CALL_MAX_AGE_SEC` | 实时数据最大允许陈旧时间，默认三百秒 |

完整实时状态必须同时满足安全地址、鉴权、现场标识、责任主体、数据授权和实时连接证明。少任何一项都不会显示为“已验证实时靠泊源”。

## 进入现场后的验收

1. 用现场历史导出包运行校验，修正身份、时区和重复事件问题。
2. 核对码头操作系统、船舶交通服务系统和船舶自动识别系统的航次关联。
3. 在影子窗口中监控延迟、缺失、晚到、修订和事件顺序异常。
4. 将靠泊事件映射到孪生状态前，冻结字段版本和责任主体。
5. 完成运营人员复核、故障演练和回滚后，仍由独立现场流程决定是否允许有限执行。
