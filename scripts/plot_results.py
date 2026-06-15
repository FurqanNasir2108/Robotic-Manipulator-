"""Generate publication-style plots from the evaluation summary."""

from __future__ import annotations

import argparse
import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.utils.config import load_config


def load_summary(summary_path: str) -> dict:
    """Load the evaluation summary JSON."""
    with open(summary_path, "r") as f:
        return json.load(f)


def _overall_metric(summary: dict, metric_name: str) -> tuple[list[str], list[float]]:
    models = list(summary["overall"].keys())
    values = [float(summary["overall"][model].get(metric_name, 0.0)) for model in models]
    return models, values


def _per_shape_matrix(summary: dict, metric_name: str) -> tuple[list[str], list[str], np.ndarray]:
    models = list(summary["per_shape"].keys())
    shapes = sorted({shape for per_model in summary["per_shape"].values() for shape in per_model.keys()})
    matrix = np.full((len(models), len(shapes)), np.nan, dtype=float)

    for row, model in enumerate(models):
        for col, shape in enumerate(shapes):
            metrics = summary["per_shape"].get(model, {}).get(shape)
            if metrics is not None:
                matrix[row, col] = float(metrics.get(metric_name, np.nan))

    return models, shapes, matrix


def plot_metric_grid(summary: dict, output_path: str):
    """Plot a compact 2x2 grid for the most important evaluation metrics."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    metrics = [
        ("joint_rmse", "Joint RMSE", "#b03a2e", False),
        ("ee_position_error", "EE Position Error", "#2874a6", False),
        ("smoothness_jerk", "Smoothness Jerk", "#af7ac5", True),
        ("inference_time_ms", "Inference Time (ms)", "#117864", True),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    axes = axes.ravel()
    for ax, (metric_name, title, color, log_scale) in zip(axes, metrics):
        models, values = _overall_metric(summary, metric_name)
        ax.bar(models, values, color=color, alpha=0.88)
        if log_scale and any(value > 0 for value in values):
            ax.set_yscale("log")
        ax.set_title(title, fontweight="bold")
        ax.tick_params(axis="x", rotation=30)
        ax.grid(True, alpha=0.25, axis="y")

    dataset_meta = summary.get("dataset", {})
    sample_count = summary.get("num_test_samples", 0)
    subset_strategy = dataset_meta.get("subset_strategy", "full")
    plt.suptitle(
        f"Evaluation Overview ({sample_count} samples, {subset_strategy} subset)",
        fontsize=15,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_accuracy_latency_tradeoff(summary: dict, output_path: str):
    """Plot the accuracy-latency Pareto-style view."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    models = list(summary["overall"].keys())
    latencies = [float(summary["overall"][model]["inference_time_ms"]) for model in models]
    rmse = [float(summary["overall"][model]["joint_rmse"]) for model in models]
    diversity = [float(summary["overall"][model].get("diversity_score", 0.0)) for model in models]

    sizes = [120 + 380 * value for value in diversity]
    colors = plt.cm.plasma(np.linspace(0.15, 0.9, len(models)))

    fig, ax = plt.subplots(figsize=(10, 7))
    for color, model, latency, score, size in zip(colors, models, latencies, rmse, sizes):
        ax.scatter(latency, score, s=size, color=color, alpha=0.8, edgecolors="black", linewidths=0.6)
        ax.annotate(model, (latency, score), textcoords="offset points", xytext=(6, 6), fontsize=9)

    if any(latency > 0 for latency in latencies):
        ax.set_xscale("log")
    ax.set_xlabel("Inference Time (ms, log scale)")
    ax.set_ylabel("Joint RMSE")
    ax.set_title("Accuracy vs Latency Trade-off", fontweight="bold")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_per_shape_heatmap(summary: dict, metric_name: str, output_path: str):
    """Plot a model-by-shape heatmap for a single metric."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    models, shapes, matrix = _per_shape_matrix(summary, metric_name)
    if not shapes:
        return

    fig, ax = plt.subplots(figsize=(max(7, len(shapes) * 1.4), max(5, len(models) * 0.75)))
    masked = np.ma.masked_invalid(matrix)
    image = ax.imshow(masked, cmap="YlGnBu", aspect="auto")
    ax.set_xticks(np.arange(len(shapes)))
    ax.set_xticklabels(shapes, rotation=30, ha="right")
    ax.set_yticks(np.arange(len(models)))
    ax.set_yticklabels(models)
    ax.set_title(f"Per-Shape {metric_name}", fontweight="bold")

    for row in range(masked.shape[0]):
        for col in range(masked.shape[1]):
            value = masked[row, col]
            if not np.ma.is_masked(value):
                ax.text(col, row, f"{float(value):.3f}", ha="center", va="center", fontsize=8)

    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_generative_oracle_gap(summary: dict, output_path: str):
    """Compare oracle vs mean joint RMSE for stochastic models."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    models = []
    oracle = []
    mean = []
    for model_name, metrics in summary["overall"].items():
        oracle_rmse = float(metrics.get("oracle_joint_rmse", metrics.get("joint_rmse", 0.0)))
        mean_rmse = float(metrics.get("mean_joint_rmse", oracle_rmse))
        if mean_rmse > oracle_rmse or metrics.get("diversity_score", 0.0) > 0:
            models.append(model_name)
            oracle.append(oracle_rmse)
            mean.append(mean_rmse)

    if not models:
        return

    x = np.arange(len(models))
    width = 0.38
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.bar(x - width / 2, oracle, width, label="Oracle", color="#1f618d")
    ax.bar(x + width / 2, mean, width, label="Mean Sample", color="#cb4335")
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=25, ha="right")
    ax.set_ylabel("Joint RMSE")
    ax.set_title("Generative Oracle vs Mean RMSE", fontweight="bold")
    ax.grid(True, alpha=0.25, axis="y")
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def generate_result_plots(summary: dict, output_dir: str):
    """Generate the evaluation plot set from a loaded summary."""
    os.makedirs(output_dir, exist_ok=True)
    plot_metric_grid(summary, os.path.join(output_dir, "metric_grid.png"))
    plot_accuracy_latency_tradeoff(summary, os.path.join(output_dir, "accuracy_latency_tradeoff.png"))
    plot_per_shape_heatmap(summary, "joint_rmse", os.path.join(output_dir, "per_shape_joint_rmse_heatmap.png"))
    plot_generative_oracle_gap(summary, os.path.join(output_dir, "generative_oracle_gap.png"))


def main():
    parser = argparse.ArgumentParser(description="Plot evaluation results")
    parser.add_argument("--config", default="configs/evaluation.yaml")
    parser.add_argument("--summary-json", default=None)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    summary_json = args.summary_json or cfg["paths"]["summary_json"]
    output_dir = args.output_dir or cfg["paths"]["plots_dir"]

    summary = load_summary(summary_json)
    generate_result_plots(summary, output_dir)
    print(f"Saved evaluation plots to {output_dir}")


if __name__ == "__main__":
    main()
