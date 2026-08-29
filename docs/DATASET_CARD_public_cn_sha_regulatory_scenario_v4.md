# public_cn_sha_regulatory_scenario_v4

## 用途

用于离线训练和盲测“海事检查 / 海关查验 → 待放行 → 放行后追赶”任务策略。基础运行数据继承 `public_cn_sha_hourly_v3`；六个监管字段来自预声明工程压力情景，不是上海港、上海海事局或海关现场统计。

## 监管链路

- 海事侧：检查需求、严重缺陷滞留比例、检查资源可用度、放行率。
- 海关侧：查验需求、二次查验比例、检查资源可用度、放行率。
- 下游传播：被检查作业量进入监管持有队列；放行后进入恢复队列，再与普通作业共同竞争泊位、岸桥、堆场和能耗资源。
- 策略权限：仅建议预留检查窗口和放行后恢复优先级；不得改变检查结果、签发放行、绕过人工确认或触发生产执行。

## 制度依据与参数边界

IMO 港口国监督流程允许在发现缺陷时延误或滞留船舶，整改后才能继续；海关公开办事指南包含进出境船舶申报、登临检查和检疫环节。V4 只据此确定流程状态机。检查比例、时长、资源能力和海关二次查验比例仍是可替换的工程压力参数。IMO 公布的 2024 年全球 PSC 滞留率约 3% 只作为全球压力参考，不代表中国或上海本地比例。

官方流程来源：

- https://www.imo.org/en/ourwork/iiis/pages/port%20state%20control.aspx
- https://www.imo.org/en/mediacentre/meetingsummaries/pages/iii-11th-session.aspx
- https://online.customs.gov.cn/static/pages/guides/000629018001/000629018001.html

## 可复现性

情景配置：`config/regulatory_delay_scenario_v4.json`。构建命令：

```bash
python -m scripts.build_regulatory_delay_scenario_v4
```

配置固定种子，监管压力脉冲由 `seed + timestamp + stream` 的 SHA-256 确定；训练使用前 70%，验证使用后续 10%，最终测试只使用最后 20%，不打乱时间顺序。

2026 独立前向挑战使用相同冻结参数、独立配置和数据集；候选模型在读取前向数据前已锁定，前向时段不得用于选模或调参：

```bash
python -m scripts.build_regulatory_delay_scenario_v4 \
  --config config/regulatory_delay_forward_challenge_v4.json
python -m scripts.evaluate_regulatory_resilience_forward_v4
```

前向证据标注为 `OUT_OF_PERIOD_FORWARD_ENGINEERING_STRESS_CHALLENGE_NOT_FIELD_KPI`，只能用于说明跨时段软件和策略表现，不能表述为现场 KPI。

## 禁止外推

所有结果必须标注 `PREDECLARED_ENGINEERING_STRESS_SCENARIO_NOT_FIELD_KPI`。若要接入真实港口，必须用经授权的船舶/航次、检查开始与结束、缺陷/滞留、复查/放行、货物查验、泊位与设备占用事件替换六个监管字段，并完成映射、校准、影子运行、人工验收和回滚演练。
