"""Generate trajectory analysis visualization plots."""

import argparse
import csv
import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def load_per_trajectory_metrics(csv_path):
    """Load per-trajectory metrics from CSV."""
    rows = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            parsed = {}
            for k, v in row.items():
                try:
                    parsed[k] = float(v)
                except (ValueError, TypeError):
                    parsed[k] = v
            rows.append(parsed)
    return rows


def plot_metric_comparison(metrics, metric_name, output_path):
    """Bar chart comparing a metric across models."""
    by_model = {}
    for row in metrics:
        model = row.get("model_name", "unknown")
        val = row.get(metric_name)
        if val is not None and isinstance(val, (int, float)):
            by_model.setdefault(model, []).append(val)

    models = sorted(by_model.keys())
    means = [np.mean(by_model[m]) for m in models]
    stds = [np.std(by_model[m]) for m in models]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(models, means, yerr=stds, capsize=5)
    ax.set_ylabel(metric_name)
    ax.set_title(f"Model Comparison: {metric_name}")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_feasibility_violations(metrics, output_path):
    """Bar chart of feasibility violation rates per model."""
    feas_keys = ["feas_joint_violation", "feas_velocity_violation", "feas_acceleration_violation"]
    by_model = {}
    for row in metrics:
        model = row.get("model_name", "unknown")
        for k in feas_keys:
            val = row.get(k)
            if val is not None:
                by_model.setdefault(model, {}).setdefault(k, []).append(float(val))

    models = sorted(by_model.keys())
    x = np.arange(len(models))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 6))
    for i, k in enumerate(feas_keys):
        means = [np.mean(by_model[m].get(k, [0])) for m in models]
        ax.bar(x + i * width, means, width, label=k.replace("feas_", ""))

    ax.set_xticks(x + width)
    ax.set_xticklabels(models, rotation=45, ha="right")
    ax.set_ylabel("Violation Rate")
    ax.set_title("Feasibility Violations by Model")
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Plot trajectory analysis results")
    parser.add_argument("--metrics-dir", default="results/metrics/trajectory_analysis")
    parser.add_argument("--figures-dir", default="figures/trajectory_analysis")
    args = parser.parse_args()

    os.makedirs(args.figures_dir, exist_ok=True)

    csv_path = os.path.join(args.metrics_dir, "per_trajectory_metrics.csv")
    if not os.path.exists(csv_path):
        print("Run analyze_trajectories.py first.")
        return

    metrics = load_per_trajectory_metrics(csv_path)

    key_metrics = [
        "joint_angle_rmse", "path_rmse_taskspace", "max_path_deviation",
        "velocity_squared_cost", "energy_per_trajectory", "diversity",
    ]
    for m in key_metrics:
        plot_metric_comparison(metrics, m, os.path.join(args.figures_dir, f"{m}_comparison.png"))

    plot_feasibility_violations(metrics, os.path.join(args.figures_dir, "feasibility_violation_barchart.png"))

    # Accuracy vs energy scatter
    by_model = {}
    for row in metrics:
        model = row.get("model_name", "unknown")
        by_model.setdefault(model, {"acc": [], "energy": []})
        if isinstance(row.get("joint_angle_rmse"), (int, float)):
            by_model[model]["acc"].append(row["joint_angle_rmse"])
        if isinstance(row.get("energy_per_trajectory"), (int, float)):
            by_model[model]["energy"].append(row["energy_per_trajectory"])

    fig, ax = plt.subplots(figsize=(10, 6))
    for model, vals in sorted(by_model.items()):
        if vals["acc"] and vals["energy"]:
            ax.scatter(np.mean(vals["acc"]), np.mean(vals["energy"]), label=model, s=100)
    ax.set_xlabel("Joint Angle RMSE")
    ax.set_ylabel("Energy per Trajectory")
    ax.set_title("Accuracy vs Energy Trade-off")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.figures_dir, "accuracy_vs_energy_scatter.png"), dpi=300)
    plt.close()

    print(f"Trajectory analysis figures saved to {args.figures_dir}")


if __name__ == "__main__":
    main()
