"""Build the final full-test comparison from cached trajectories."""

from __future__ import annotations

import json
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scripts.plot_results import generate_result_plots
from src.evaluate.comparisons import (
    _benchmark_model_latency,
    load_model_specs,
    load_test_dataset,
    plot_main_comparison,
    plot_trajectory_overlay,
    write_summary_files,
)
from src.evaluate.metrics import (
    constraint_violation_rate,
    diversity_score,
    ee_orientation_error,
    ee_position_error,
    energy_proxy,
    joint_rmse,
    path_tracking_error,
    smoothness_jerk,
)
from src.simulation.manipulator import ThreeLinkManipulator
from src.utils.config import load_config


MODEL_ORDER = [
    "analytical_ik",
    "cnn",
    "cnn_lstm",
    "cnn_gru",
    "cvae",
    "diffusion_ddim",
]

CACHE_FILES = {
    "analytical_ik": "results/generated_trajectories/analytical_ik.npz",
    "cnn": "results/generated_trajectories/cnn.npz",
    "cnn_lstm": "results/generated_trajectories/cnn_lstm.npz",
    "cnn_gru": "results/generated_trajectories/cnn_gru.npz",
    "cvae": "results/generated_trajectories/cvae_full_4500.npz",
    "diffusion_ddim": "results/generated_trajectories/diffusion_ddim.npz",
}

MODEL_KIND = {
    "analytical_ik": "deterministic",
    "cnn": "deterministic",
    "cnn_lstm": "deterministic",
    "cnn_gru": "deterministic",
    "cvae": "generative",
    "diffusion_ddim": "generative",
}


def metric_row(prediction, target, fk_fn, joint_limits):
    return {
        "joint_rmse": joint_rmse(prediction, target),
        "ee_position_error": ee_position_error(prediction, target, fk_fn),
        "ee_orientation_error": ee_orientation_error(prediction, target, fk_fn),
        "path_tracking_error": path_tracking_error(prediction, target, method="dtw"),
        "energy_proxy": energy_proxy(prediction, method="velocity"),
        "smoothness_jerk": smoothness_jerk(prediction, dt=1.0),
        "constraint_violation_rate": constraint_violation_rate(prediction, joint_limits),
    }


def aggregate_rows(rows):
    metric_names = rows[0].keys()
    return {name: float(np.mean([row[name] for row in rows])) for name in metric_names}


def validate_cache_lengths():
    expected = 4500
    errors = []
    for model_name, cache_path in CACHE_FILES.items():
        if not os.path.exists(cache_path):
            errors.append(f"{model_name}: missing cache {cache_path}")
            continue
        data = np.load(cache_path, allow_pickle=True)
        n_samples = int(len(data["q_reference"]))
        if n_samples != expected:
            errors.append(f"{model_name}: expected {expected} samples, found {n_samples} in {cache_path}")
    if errors:
        raise RuntimeError("Full-evaluation caches are incomplete:\n" + "\n".join(errors))


def build_summary_from_caches():
    robot = ThreeLinkManipulator()
    overall_summary = {}
    per_shape_summary = {}
    overlay_payload = {}
    model_metadata = {}

    cfg = load_config("configs/evaluation.yaml")
    dataset, normalizer = load_test_dataset(
        cfg["paths"]["data_dir"],
        cfg["paths"]["metadata_dir"],
        max_samples=None,
        seed=cfg["evaluation"].get("seed", 42),
        subset_strategy="stratified",
    )
    model_specs = load_model_specs(
        device="cpu",
        include=MODEL_ORDER,
        include_diffusion_ddpm=False,
    )

    for model_name in MODEL_ORDER:
        cache_path = CACHE_FILES[model_name]
        payload = np.load(cache_path, allow_pickle=True)
        q_predicted = payload["q_predicted"]
        q_oracle_predicted = payload["q_oracle_predicted"]
        q_reference = payload["q_reference"]
        shape_type = payload["shape_type"]
        waypoints = payload["waypoints"]

        latency_ms, latency_metadata = _benchmark_model_latency(
            model_specs[model_name],
            dataset,
            normalizer,
            seed=cfg["evaluation"].get("seed", 42),
            subset_strategy=cfg["evaluation"].get("latency_subset_strategy", "stratified"),
            n_conditions=cfg["evaluation"].get("latency_num_conditions", 3),
            n_runs=cfg["evaluation"].get("inference_runs", 10),
        )

        rows = []
        shape_rows = defaultdict(list)
        for index in range(len(q_reference)):
            target = np.asarray(q_reference[index], dtype=np.float32)
            shape = str(shape_type[index])

            if MODEL_KIND[model_name] == "generative":
                predictions = np.asarray(q_predicted[index], dtype=np.float32)
                sample_rows = [
                    metric_row(prediction, target, robot.forward_kinematics, robot.joint_limits)
                    for prediction in predictions
                ]
                oracle_idx = int(np.argmin([row["joint_rmse"] for row in sample_rows]))
                oracle_row = dict(sample_rows[oracle_idx])
                mean_row = aggregate_rows(sample_rows)
                oracle_row["diversity_score"] = diversity_score(predictions, method="dtw")
                oracle_row["inference_time_ms"] = latency_ms
                oracle_row["oracle_joint_rmse"] = oracle_row["joint_rmse"]
                oracle_row["mean_joint_rmse"] = mean_row["joint_rmse"]
                row = oracle_row
                best_prediction = np.asarray(q_oracle_predicted[index], dtype=np.float32)
            else:
                prediction = np.asarray(q_predicted[index], dtype=np.float32)
                row = metric_row(prediction, target, robot.forward_kinematics, robot.joint_limits)
                row["diversity_score"] = 0.0
                row["inference_time_ms"] = latency_ms
                row["oracle_joint_rmse"] = row["joint_rmse"]
                row["mean_joint_rmse"] = row["joint_rmse"]
                best_prediction = prediction

            rows.append(row)
            shape_rows[shape].append(row)

            if index == 0:
                overlay_payload[model_name] = {
                    "condition": np.asarray(waypoints[index], dtype=np.float32).tolist(),
                    "target": target.tolist(),
                    "prediction": best_prediction.tolist(),
                }

        overall_summary[model_name] = aggregate_rows(rows)
        per_shape_summary[model_name] = {
            shape: aggregate_rows(shape_metric_rows)
            for shape, shape_metric_rows in shape_rows.items()
        }
        model_metadata[model_name] = {
            "kind": MODEL_KIND[model_name],
            "latency_benchmark": latency_metadata,
            "source_cache": cache_path,
        }

    summary = {
        "overall": overall_summary,
        "per_shape": per_shape_summary,
        "overlay": overlay_payload,
        "num_test_samples": int(dataset.original_size),
        "dataset": {
            "original_size": int(dataset.original_size),
            "selected_size": int(dataset.original_size),
            "subset_strategy": "full",
            "shape_counts": dataset.original_shape_counts,
            "original_shape_counts": dataset.original_shape_counts,
        },
        "model_metadata": model_metadata,
    }
    return summary


def main():
    validate_cache_lengths()
    summary = build_summary_from_caches()

    write_summary_files(summary, "results/metrics/summary.json", "results/metrics/summary.md")
    plot_main_comparison(summary, "figures/comparison/main_comparison.png")
    plot_trajectory_overlay(summary, "figures/comparison/trajectory_overlay.png")
    generate_result_plots(summary, "results/plots")

    os.makedirs("results/metrics/final_full_test", exist_ok=True)
    with open("results/metrics/final_full_test/full_cross_model_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    write_summary_files(
        summary,
        "results/metrics/final_full_test/full_cross_model_summary.json",
        "results/metrics/final_full_test/full_cross_model_summary.md",
    )

    print(f"Final full-test summary written for {summary['num_test_samples']} samples")
    print("Primary summary: results/metrics/summary.json")


if __name__ == "__main__":
    main()
