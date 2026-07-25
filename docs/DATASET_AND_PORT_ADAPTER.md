# 数据集与港口适配

## 规范训练表

| 字段 | 单位/口径 | 要求 |
|---|---|---|
| `timestamp` | ISO-8601 | 时间顺序，建议 UTC |
| `base_load_kw` | kW | 未执行策略前的港区/对象基础负荷 |
| `throughput_teu` | TEU/时间步 | 同一时间步作业吞吐 |
| `vessel_arrivals` | 艘/时间步 | 到港或计划到港强度 |
| `tide_m` | m | 统一潮位基准 |
| `price_per_kwh` | 货币/kWh | 与成本口径一致 |
| `carbon_kg_per_kwh` | kgCO2e/kWh | 发布机构与版本需写入元数据 |
| `ambient_c` | °C | 同时间步环境温度 |

至少 48 行，全部数值必须有限。导入不会编造缺失列；来源字段名不一致时必须显式提交 `mapping_json`。

### 可选现实因素（`port_ops_v2`）

`wind_speed_mps`、`visibility_km`、`wave_height_m`、`current_speed_mps`、
`berth_occupancy_ratio`、`yard_occupancy_ratio`、`crane_availability_ratio`、
`equipment_availability_ratio`、`channel_congestion_ratio`、`reefer_load_kw`、
`pilot_tug_availability_ratio` 和 `closure_flag` 可以随规范CSV导入。

缺失因素不会被补造成优秀观测。v2环境为每个因素同时编码一个可用性掩码；港口场景包可把指定因素设为训练必需项或现场声明必需项。

## 港口场景包

`config/ports/*.json` 把换港所需的非时序参数与训练数据分离，包括港口代码、时区、货币、BESS容量、需量上限、SOC/服务/调度动作边界、目标权重、天气阈值、因素要求和校准状态。

开源场景固定为 `recommendation_only`。样例中的设备参数和停工阈值不是港口批准值；真实接入必须新增一个经现场所有者批准的场景包，不能直接修改公开基准来伪装校准。

## 自带公开实例

`public_port_ops_v1` 使用两类同地理口径的公开官方输入：

- 新加坡海事及港务管理局发布的 2020–2025 月度集装箱吞吐量；
- 新加坡海事及港务管理局发布的 2020–2025 月度集装箱船到港量。

为了形成可以运行环境的 52,608 小时表，吞吐与到港量由月度总量确定性分配；
基础负荷、电价、碳因子、气温和潮位压力项是脚本内可复现的工程派生量。潮位不参与
三项业务 KPI。元数据将官方锚点和派生列分开记录，因此该文件只能作为
集成/复现实例，不能称为港口生产遥测。

`public_us_la_6min_v1` 使用美国交通部BTS的洛杉矶港月度TEU以及NOAA CO-OPS六分钟水位、气温和风速，形成87,459步的高频公开对照。其基础负荷、时步吞吐/到港、电价、碳因子、泊位/堆场占用和航道拥堵仍是明确派生列。原始观测与短缺口插值分别计数，详见 [数据卡](DATASET_CARD_public_us_la_6min_v1.md)。

刷新固定公开数据窗口：

```bash
python -m scripts.fetch_public_port_dataset
python -m scripts.fetch_public_la_benchmark
```

## 导入港口数据

```bash
curl -X POST http://127.0.0.1:8000/api/rl/datasets/upload \
  -F 'file=@/path/to/port_export.csv' \
  -F 'dataset_id=my_port_2026q2' \
  -F 'mapping_json={"timestamp":"event_time","base_load_kw":"grid_kw","throughput_teu":"teu","vessel_arrivals":"arrivals","tide_m":"tide","price_per_kwh":"tariff","carbon_kg_per_kwh":"grid_ef","ambient_c":"temperature"}' \
  -F 'metadata_json={"port_code":"MYTPP","timezone":"Asia/Kuala_Lumpur","owner":"operator","license":"internal-approved","intended_use":"offline RL decision-support training"}'
```

上传元数据必须提供 `license`、`owner`、`timezone` 和 `intended_use`。同名数据集不会被静默覆盖；确需替换时须显式增加 `"replace_existing":true`。上传默认限制为 50 MiB。

导入后先检查 `/api/rl/datasets/<dataset_id>/quality` 中的行数、时间范围、哈希、物理边界和治理元数据，再开始训练。

默认遥测适配器会把规范数据集作为只读的 `port-grid-aggregate` 回放源。可通过
`PORT_DT_TELEMETRY_DATASET=/absolute/path/to/mapped_port.csv` 替换；该文件仍须满足上表字段。
短期预测使用遥测拟合的岭回归自回归模型，分位区间来自拟合残差；预测时间粒度不会细于源数据采样粒度。

## 实体画面适配

默认 `PORTVIZ_MODE=dataset` 使用规范数据集驱动确定性视觉投影。车辆和设备位置不是实测轨迹，返回数据会明确标记 `measured_entity_tracks=false`。

生产/回放实体轨迹使用：

```bash
export PORTVIZ_MODE=real
export PORTVIZ_FRAME_PATH=/absolute/path/to/port_frames.jsonl
export PORTVIZ_CONFIG=/absolute/path/to/port_layout.json
```

JSONL 每行是一帧，至少应包含时间戳和所需实体列表：

```json
{"ts":1784515200000,"agv":[{"lane":0,"s":0.42,"alarm":false}],"qc":[{"busy":true,"trolley":0.31}],"yc":[{"busy":false}],"tr":[],"hotspots":[],"vessels":[{"berth":0,"progress":0.68,"len":240}]}
```

`real` 模式缺少文件、文件为空或 JSON 非法时直接失败，不回退到模拟。

## 外部 REST 适配

TOS、市场和 AIS/潮汐配置位于 `data/objects/config`，也可以使用 `.env.example` 中的环境变量覆盖。`fallback_mock=false` 是推荐默认值：

- 未填写 `base_url`：适配器明确报告 `engineering_simulator`；
- 填写 `base_url` 且运行库可用：报告 `live_rest`；
- 真实请求失败：接口返回 502，不生成模拟业务结果。

通过 `/api/system/provenance` 在接港验收脚本中统一检查来源状态。令牌不得进入仓库。

## 非 RL 证据文件

ESG、合规、TwinLab、AI 可信度和多港口汇总不会仅因 JSON 文件存在就展示结论。证据需内联 `_provenance` 或提供同名 `.meta.json`，最少包含：

```json
{
  "provenance_type": "port_export",
  "source_url": "tos://approved-export/2026-q2",
  "owner": "terminal-data-office",
  "generated_at": "2026-07-20T00:00:00Z",
  "license": "internal-approved"
}
```

`provenance_type` 可使用模块允许的 `public`、`port_export`、`audited` 或 `verified_test`。缺少来源时接口返回 `available=false`，不生成替代评分、收益、演练通过率或合规结论。

## A/B 实测适配器

闭环模块不会把预测加噪声当作实测。现场 telemetry 实现可选方法：

```python
def collect_ab_observations(self, job_id: str, strategy: dict, predicted: dict) -> dict:
    return {
        "baseline_kwh": 1280.4,
        "strategy_kwh": 1216.8,
        "window_start": "2026-07-20T01:00:00Z",
        "window_end": "2026-07-20T02:00:00Z",
        "source": "historian://approved-meter-query/abc123",
    }
```

返回值必须来自对齐后的独立基线/策略观测窗口并携带来源。适配器缺失或窗口不完整时，`/api/exec/abtest/{job_id}` 返回 `available=false`，`/api/exec/learn` 不更新模型。
