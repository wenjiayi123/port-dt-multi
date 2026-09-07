"""Export an additive, source-linked RL audit and actual validation curves."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import numpy as np

from app.services.rl_training.datasets import file_sha256
from app.services.rl_training.model_artifacts import resolve_model_artifact
from scripts.train_business_rl_v7 import ROOT, write, relative


def main():
    base = ROOT / "evidence/v7"
    reports = []
    for path in sorted(base.glob("*/runs/*/report.json")):
        value = json.loads(path.read_text())
        models = value.get("results", [])
        for row in models:
            resolve_model_artifact(ROOT, row["model_path"], row["model_sha256"])
        reports.append({"run_id": value["run_id"], "module": path.parents[2].name, "status": value["status"], "report_path": relative(path), "report_sha256": file_sha256(path), "seed_count": len(models), "environment_steps": sum(row.get("steps", row.get("real_additional_environment_steps", 0)) for row in models), "optimizer_updates": sum(row.get("optimizer_updates", row.get("real_additional_optimizer_updates", row.get("optimizer_steps", 0))) for row in models)})
    receipt = json.loads((base / "runtime_verification.json").read_text())
    previous = {r["module"]: r["comparison_vs_previous_distilled_actor_same_windows"] for r in receipt["specialist_receipts"]}
    modules = []
    names = {"yard_lighting": "堆场照明", "hvac": "空调与冷站", "yard_crane": "场桥作业"}
    font_path = Path("/System/Library/Fonts/PingFang.ttc")
    font = FontProperties(fname=font_path) if font_path.exists() else FontProperties()
    fig, axes = plt.subplots(1,3,figsize=(15,4.7))
    colors = ["#007c91", "#ed8b23", "#6c54a4"]
    for ax, (name, title) in zip(axes, names.items()):
        pointer = json.loads((base / name / "offline_champion.json").read_text())
        report = json.loads((ROOT / pointer["report_path"]).read_text())
        selected = next(r for r in report["results"] if r["seed"] == report["selected_seed"])
        for color, result in zip(colors, report["results"]):
            curve = json.loads((ROOT / result["model_path"]).with_name("curve.json").read_text())
            xs = [0] + [r["step"] / 1000 for r in curve]
            ys = [result["initial_comparison"]["total_cost_cny"]["mean"]] + [r["comparison"]["total_cost_cny"]["mean"] for r in curve]
            ax.plot(xs, ys, color=color, marker="o", markersize=3, linewidth=1.8, label=f"seed {result['seed']}")
            best = result["selected"]
            ax.scatter([best["step"] / 1000], [best["comparison"]["total_cost_cny"]["mean"]], marker="*", color=color, s=110, zorder=5)
        ax.set_title(title + " · SAC", fontproperties=font, fontsize=14)
        ax.set_xlabel("真实环境交互步数（千步）", fontproperties=font)
        ax.set_ylabel("固定验证窗口成本改善（%）", fontproperties=font)
        ax.set_ylim(bottom=0)
        ax.grid(alpha=0.18)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(fontsize=8, loc="lower right")
        summary = {"module": name, "name_cn": title, "run_id": report["run_id"], "seed": selected["seed"], "converged_seeds": sum(r["convergence"]["passed"] for r in report["results"]), "test_window_count": len(selected["test"]["evaluation"]["rows"]), "window_hours": report["config"]["episode_steps"] * (1/12 if name == "yard_lighting" else .25), "cost_reduction_percent": selected["test"]["comparison"]["total_cost_cny"], "energy_reduction_percent": selected["test"]["comparison"]["energy_kwh"], "carbon_reduction_percent": selected["test"]["comparison"]["carbon_kg"], "peak_reduction_percent": selected["test"]["comparison"]["peak_kw"], "comparison_vs_previous_actor": previous[name], "model_path": pointer["model_path"], "model_sha256": pointer["model_sha256"]}
        modules.append(summary)
    fig.suptitle("V7 实际强化学习验证曲线 · 星号为验证集选中的检查点", fontproperties=font, fontsize=16)
    fig.text(.5,.015,"历史工程回放；保留原始检查点，无插值、无展示噪声；不代表现场测得收益。", ha="center", fontproperties=font, fontsize=10, color="#555555")
    fig.tight_layout(rect=(0,.055,1,.93))
    fig.savefig(base / "convergence_curves.png", dpi=160)
    plt.close(fig)
    result = {"schema": "port-rl-convergence-audit.v7", "scope": "RL only; no control algorithm training", "completed_runs": reports, "completed_seed_runs": sum(r["seed_count"] for r in reports), "new_environment_steps": sum(r["environment_steps"] for r in reports), "new_optimizer_updates": sum(r["optimizer_updates"] for r in reports), "admitted_specialists": modules, "joint_strategy": "retained_hash_verified_V6_SAC", "still_not_admitted": ["shore_bess_balanced", "bess_energy_balanced", "agv_charge_business_value"], "legacy_qc": "linear_IQL_lite_preserved; corrected_KL_metric; core_QC_allocation_is_in_retained_V6_SAC", "production_authority": False, "dispatch_allowed": False, "live_data_verified": False, "evaluation_boundary": "previously used chronological/public engineering benchmarks; not a fresh blind dataset or measured field benefit"}
    write(base / "audit_summary.json", result)
    print(json.dumps({"runs": len(reports), "seed_runs": result["completed_seed_runs"], "environment_steps": result["new_environment_steps"], "optimizer_updates": result["new_optimizer_updates"], "all_model_hashes_verified": True}))


if __name__ == "__main__":
    main()
