"""Plot saved validation checkpoints, without smoothing or invented samples."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports", nargs="+", required=True)
    parser.add_argument("--output")
    args = parser.parse_args()
    output = Path(args.output) if args.output else ROOT / "evidence/v8/shore_bess/figures" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output.mkdir(parents=True, exist_ok=False)
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    metrics = [("total_cost_cny", "Cost reduction (%)"), ("carbon_kg", "Carbon reduction (%)"),
               ("peak_kw", "Peak reduction (%)"), ("projection_rate", "Executed-action correction rate (%)")]
    rows = []
    inputs = []
    for index, report_path in enumerate(args.reports):
        path = Path(report_path)
        report = json.loads(path.read_text())
        inputs.append({"report": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
        cfg = report["config"]
        algorithm = cfg["algorithm"].split(".")[-1]
        if cfg.get("n_step", 1) > 1:
            algorithm += f" (n-step {cfg['n_step']} variant)"
        for seed in report["results"]:
            curve_path = path.parent / f"seed_{seed['seed']}" / "curve.json"
            curve_relative = str(curve_path.resolve().relative_to(ROOT))
            expected = report.get("evidence_files_sha256", {}).get(curve_relative)
            actual = hashlib.sha256(curve_path.read_bytes()).hexdigest()
            if expected is not None and actual != expected:
                raise ValueError("curve bytes differ from sealed report: " + curve_relative)
            inputs[-1].setdefault("curves", []).append({"path": curve_relative, "sha256": actual,
                "bound_to_report_hash_registry": expected is not None})
            curve = json.loads(curve_path.read_text())
            label = f"{algorithm} / seed {seed['seed']} / run {index + 1}"
            for ax, (metric, title) in zip(axes.flat, metrics):
                steps = [point["step"] for point in curve]
                if metric == "projection_rate":
                    values = [100 * point["evaluation"]["mean"][metric] for point in curve]
                else:
                    values = [point["comparison"][metric]["mean"] for point in curve]
                line, = ax.plot(steps, values, marker="o", markersize=3, linewidth=1.3, label=label)
                if metric != "projection_rate":
                    ax.fill_between(steps, [p["comparison"][metric]["ci_low"] for p in curve],
                                    [p["comparison"][metric]["ci_high"] for p in curve],
                                    color=line.get_color(), alpha=.07)
                ax.set_title(title, loc="left", fontsize=11)
                ax.set_xlabel("Actual environment steps")
                ax.grid(alpha=.2)
                ax.axhline(0, color="#6b7280", linewidth=.7, linestyle="--")
            for point in curve:
                rows.append({"run_id": report["run_id"], "algorithm": algorithm, "seed": seed["seed"],
                    "step": point["step"], "optimizer_updates": point["optimizer_updates"],
                    **{metric: point["comparison"][metric]["mean"] for metric, _ in metrics[:3]},
                    "projection_rate": point["evaluation"]["mean"]["projection_rate"],
                    "all_business_gates_passed": all(point["gates"].values()),
                    "model_sha256": point["model_sha256"]})
    axes[0, 0].legend(fontsize=7)
    fig.suptitle("Shore + BESS: measured validation checkpoints\nPublic engineering replay; bands are paired-window bootstrap intervals", fontsize=14)
    fig.savefig(output / "validation_curves.png", dpi=180)
    fig.savefig(output / "validation_curves.svg")
    plt.close(fig)
    with (output / "checkpoint_values.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output / "inputs.json").write_text(json.dumps({"reports": inputs,
        "plot_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "point_source": "persisted per-seed curve.json", "smoothing": False,
        "lines": "straight connectors between observed checkpoints, not extra observations"}, indent=2) + "\n")
    print(output)


if __name__ == "__main__":
    main()
