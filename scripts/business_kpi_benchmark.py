"""Generate the fixed, public-input-driven business KPI benchmark."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from app.services.business_benchmark import (
    DEFAULT_CONFIG,
    DEFAULT_REPORT,
    ROOT,
    build_report,
    load_verified_report,
)


MARKDOWN = ROOT / "docs/BUSINESS_KPI_BENCHMARK.md"
DAILY_CSV = ROOT / "data/rl/business_kpi_benchmark_v1_daily.csv"


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def render_markdown(report: dict) -> str:
    test = report["test"]
    improvements = test["improvements"]
    validation = report["validation"]["improvements"]
    uncertainty = test["uncertainty"]
    baseline = test["baseline"]
    policy = test["coordinated_policy"]
    complete_days = uncertainty[
        "berth_utilization_relative_improvement_percent"
    ]["n"]
    berth_point_gain = (
        policy["berth_utilization"] - baseline["berth_utilization"]
    ) * 100.0
    sensitivity = report["sensitivity"]
    ranges = sensitivity["ranges"]
    return f"""# Web端跨资源协同调度业务KPI基准 v1

## 结论

- 固定输入：`{report["dataset"]["dataset_id"]}`，共 {report["dataset"]["rows"]:,} 条小时记录。
- 时间切分：{report["dataset"]["split_sizes"]["train"]} train / {report["dataset"]["split_sizes"]["validation"]} validation / {report["dataset"]["split_sizes"]["test"]} test，不打乱。
- 测试区间：{report["dataset"]["test_period"]["start"]} 至 {report["dataset"]["test_period"]["end"]}。
- 对照：静态 FCFS + 固定能源时刻表。
- 候选：泊位、岸桥/AGV服务强度与岸电储能协同策略。
- 验证集相对变化：泊位 +{validation["berth_utilization_relative_improvement_percent"]:.2f}% / 待泊 -{validation["average_waiting_time_reduction_percent"]:.2f}% / 成本 -{validation["scenario_energy_cost_reduction_percent"]:.2f}%，与最终测试方向和量级一致。

| 测试指标 | 静态对照 | 协同策略 | 相对变化 |
|---|---:|---:|---:|
| 泊位利用率 | {baseline["berth_utilization"]:.2%} | {policy["berth_utilization"]:.2%} | +{berth_point_gain:.2f} 个百分点（相对 +{improvements["berth_utilization_relative_improvement_percent"]:.2f}%） |
| 平均待泊时间 | {baseline["average_waiting_hours"]:.3f} h | {policy["average_waiting_hours"]:.3f} h | -{improvements["average_waiting_time_reduction_percent"]:.2f}% |
| 情景用电成本 | {baseline["scenario_energy_cost"]:,.2f} | {policy["scenario_energy_cost"]:,.2f} | -{improvements["scenario_energy_cost_reduction_percent"]:.2f}% |

简历可写为：**在参数预声明的数字孪生情景对照中，泊位有效利用率提升 {berth_point_gain:.2f} 个百分点（相对 +{report["resume_claims_rounded_percent"]["berth_utilization_relative_improvement_percent"]:.0f}%）、平均待泊时间缩短 {report["resume_claims_rounded_percent"]["average_waiting_time_reduction_percent"]:.0f}%、情景用电成本降低 {report["resume_claims_rounded_percent"]["scenario_energy_cost_reduction_percent"]:.0f}%**。

## 稳定性

测试窗包含 {complete_days} 个完整 UTC 日。对完整日的 2,000 次成对 bootstrap 均值 95% 区间为：泊位利用率相对提升 {uncertainty["berth_utilization_relative_improvement_percent"]["ci_low"]:.2f}%–{uncertainty["berth_utilization_relative_improvement_percent"]["ci_high"]:.2f}%、平均待泊时间缩短 {uncertainty["average_waiting_time_reduction_percent"]["ci_low"]:.2f}%–{uncertainty["average_waiting_time_reduction_percent"]["ci_high"]:.2f}%、情景用电成本降低 {uncertainty["scenario_energy_cost_reduction_percent"]["ci_low"]:.2f}%–{uncertainty["scenario_energy_cost_reduction_percent"]["ci_high"]:.2f}%。

预声明 {sensitivity["predeclared_scenarios"]} 组参数敏感性复算区间为：泊位利用率相对提升 {ranges["berth_utilization_relative_improvement_percent"]["min"]:.2f}%–{ranges["berth_utilization_relative_improvement_percent"]["max"]:.2f}%、平均待泊时间缩短 {ranges["average_waiting_time_reduction_percent"]["min"]:.2f}%–{ranges["average_waiting_time_reduction_percent"]["max"]:.2f}%、情景用电成本降低 {ranges["scenario_energy_cost_reduction_percent"]["min"]:.2f}%–{ranges["scenario_energy_cost_reduction_percent"]["max"]:.2f}%。

## 计算口径

- 泊位利用率：生产性泊位小时 /（生产性泊位小时 + 预留不确定性占位小时）。
- 平均待泊时间：按到港量加权的拥堵等待 + 一半预留缓冲。
- 情景用电成本：小时净负荷 × 分时电价；每个UTC日独立初始化BESS，并把日末未恢复柔性负荷和BESS电量按配置参考价结算，避免跨日借能或通过欠供制造收益。
- 吞吐量在对照和候选中保持一致。

## 数据与算法边界

公开输入为 MPA 新加坡 2020–2025 月度集装箱吞吐量和集装箱船到港量，业务地理口径一致；小时负荷、小时到港分配、分时电价、碳因子、气温与潮位扰动仍是确定性工程派生量，其中潮位不参与三项业务 KPI 计算。50 MWh / 4 MW BESS、泊位缓冲与设备生产率是冻结的情景假设，不是港口标定值。候选结果来自数字孪生反事实仿真，不是港口实测运营 KPI。Web 端实现多智能体业务编排与联合策略控制，不应写成已完成现场部署的分布式 MARL。

## 收益归因

- 泊位利用率与待泊时间的差异由预声明的 4 小时/船与 2 小时/船不确定性缓冲情景机械产生；缓冲量不是由历史结果训练或现场标定，不能解释为因果改善。
- 情景用电成本差异来自公开实现的确定性 BESS 与柔性负荷时刻表，并包含每日终端能量结算；这项指标不是“RL 已训练并优于基线”的证据。
- 对照与候选使用完全相同的吞吐量，未通过减少作业量制造收益，也不声明新增吞吐能力。

## 复现

```bash
.venv312/bin/python -m scripts.business_kpi_benchmark
.venv312/bin/python -m scripts.business_kpi_benchmark --verify
```
"""


def write_daily_csv(report: dict) -> None:
    rows = report["test"]["daily_paired_metrics"]
    temporary = DAILY_CSV.with_suffix(".csv.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(DAILY_CSV)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--verify", action="store_true")
    arguments = parser.parse_args()
    if arguments.verify:
        report = load_verified_report(arguments.output)
        print(
            "business KPI benchmark verify: PASS "
            f"({report['dataset']['split_sizes']['test']} test rows)"
        )
        return 0
    report = build_report(arguments.config)
    write_json(arguments.output, report)
    MARKDOWN.write_text(render_markdown(report), encoding="utf-8")
    write_daily_csv(report)
    print(json.dumps(report["resume_claims_rounded_percent"], ensure_ascii=False))
    return 0 if report["release_gate"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
