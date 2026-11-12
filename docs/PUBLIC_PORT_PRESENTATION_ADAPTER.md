# 展示数据源替换

新增源独立于训练模型与生产控制。默认PORTVIZ_MODE=public_demo使用洋山公开参数标定演示；原dataset/mock/real模式保留。

只读接口：

- GET /api/public-port-demo/profile：事实、来源、假设与哈希。
- GET /api/public-port-demo/snapshot?at=2025-12-15T12:00:00%2B08:00：固定时刻快照；省略at使用历史演示时钟。
- GET /api/public-port-demo/assets、/asset-series/{id}：工程机群与近期24点负荷。
- GET /api/public-port-demo/replacement-contract：字段映射与接入边界。

服务台可导出当前快照。生成带11组历史的完整接入样本（不覆盖已有文件）：

```bash
.venv312/bin/python scripts/export_public_port_presentation.py --output /绝对路径/snapshot.json
```

--at可指定含时区的历史时刻。完整样本已生成在.codex_artifacts/module-visuals-20261023/yangshan-presentation.example.json。

对接进程输出同一schema的JSON，然后配置并重启服务：

```bash
export PORT_DT_PRESENTATION_SNAPSHOT=/绝对路径/snapshot.json
.venv312/bin/python -m uvicorn app.server:app --host 127.0.0.1 --port 8000
```

| 字段 | 现场来源 | 单位/规则 |
| --- | --- | --- |
| schema | 固定port-dt-presentation-snapshot.v1 | 不改版本名 |
| source_timestamp | 源系统时间 | ISO8601，必须含时区 |
| profile | 主数据与来源说明 | facts/sources/scenario_parameters/timezone/profile_id |
| operating | TOS/ECS/EMS汇总 | 台、TEU/h、kW、kWh、kgCO₂e；总数与作业数分列 |
| calls | TOS航次/泊位计划 | call_id、berth_index从0起、phase、eta、planned_berth_time、planned_completion、cargo_teu、progress 0–1 |
| series | 同窗小时序列 | timestamp、power_kw、throughput_teu_per_hour及环境量 |
| environment | 授权气象/海洋数据 | ambient_c、wind_speed_mps、wave_height_m、tide_m、current_speed_mps |
| provenance.source_ref | 来源系统/制品/接口引用 | 必填，保留来源时间与核验说明 |
| asset_groups/asset_series | 可选EMS分项计量 | 机群ID、kW；历史含source_ts。缺少时不擅自拆分总表 |

11个映射ID：qc-01、qc-02、yard-01、agv-01、bess-01、hvac-01、reefer-01、shore-01、gate-01、lighting-01、aux-01，表示展示机群类别。现场适配器映射标签和测点，不把机群值当单台电表。建议用临时文件加原子重命名更新快照，避免读取半写入文件。

校验时区、有限值、来源引用、资源边界、航次进度和时序；损坏/缺失配置返回503，不静默回填演示。文件不能自行授予控制权限或声明已完成现场核验。

## 三维布局与轨迹

公开布局是结构示意。实际坐标、路径或设备规模不同，应切换既有严格适配器：PORTVIZ_MODE=real、PORTVIZ_CONFIG=/path/layout.json、PORTVIZ_FRAME_PATH=/path/frames.jsonl。快照负责作业/能碳，空间适配器负责几何/实体帧，二者应由相同源时间与资产映射关联。

公开空间固定为7泊位、29岸桥、121轨道吊、155AGV。替换快照规模不一致时会报错，防止把新数据画到旧示意布局。

这些配置替换新增展示层。原模型输入、盲测、门禁、审批和执行按原契约接入。现场工况仍需校准或重训、独立验证，不能凭替换展示文件宣称生产就绪。
