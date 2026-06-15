"""Fast full-test summary assembly using caches plus the existing cVAE summary."""

from __future__ import annotations

import json
import math
import os
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor

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

KIND = {
    "analytical_ik": "deterministic",
    "cnn": "deterministic",
    "cnn_lstm": "deterministic",
    "cnn_gru": "deterministic",
    "diffusion_ddim": "generative",
}

N_WORKERS = 8


def wrap_angle(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi


def fk_batch(q):
    q = np.asarray(q, dtype=np.float32)
    l1, l2, l3 = 1.0, 1.0, 0.5
    q1 = q[..., 0]
    q2 = q[..., 1]
    q3 = q[..., 2]
    x = l1 * np.cos(q1) + l2 * np.cos(q1 + q2) + l3 * np.cos(q1 + q2 + q3)
    y = l1 * np.sin(q1) + l2 * np.sin(q1 + q2) + l3 * np.sin(q1 + q2 + q3)
    theta = q1 + q2 + q3
    return np.stack([x, y, theta], axis=-1).astype(np.float32)


def dtw_distance(path_a, path_b):
    path_a = np.asarray(path_a, dtype=np.float32)
    path_b = np.asarray(path_b, dtype=np.float32)
    n_steps_a = len(path_a)
    n_steps_b = len(path_b)
    cost = np.full((n_steps_a + 1, n_steps_b + 1), np.inf, dtype=np.float64)
    cost[0, 0] = 0.0
    for i in range(1, n_steps_a + 1):
        for j in range(1, n_steps_b + 1):
            dist = float(np.linalg.norm(path_a[i - 1] - path_b[j - 1]))
            cost[i, j] = dist + min(cost[i - 1, j], cost[i, j - 1], cost[i - 1, j - 1])
    return float(cost[n_steps_a, n_steps_b] / max(n_steps_a + n_steps_b, 1))


def energy_proxy(trajectory):
    velocity = np.diff(trajectory, axis=0)
    return float(np.mean(np.sum(velocity**2, axis=-1)))


def smoothness_jerk(trajectory, dt=1.0):
    velocity = np.diff(trajectory, axis=0) / dt
    acceleration = np.diff(velocity, axis=0) / dt
    jerk = np.diff(acceleration, axis=0) / dt
    if jerk.size == 0:
        return 0.0
    value = float(np.mean(np.sum(jerk**2, axis=-1)))
    return 0.0 if abs(value) < 1e-12 else value


def constraint_violation_rate(trajectory):
    lower = np.array([-math.pi, -math.pi, -math.pi], dtype=np.float32)
    upper = np.array([math.pi, math.pi, math.pi], dtype=np.float32)
    violations = (trajectory < lower) | (trajectory > upper)
    return float(np.mean(violations))


def diversity_score(samples):
    n_samples = len(samples)
    if n_samples < 2:
        return 0.0
    distances = []
    for i in range(n_samples):
        for j in range(i + 1, n_samples):
            distances.append(dtw_distance(samples[i], samples[j]))
    return float(np.mean(distances))


def metric_row(prediction, target):
    prediction = np.asarray(prediction, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32)
    pred_pose = fk_batch(prediction)
    target_pose = fk_batch(target)
    return {
        "joint_rmse": float(np.sqrt(np.mean((prediction - target) ** 2))),
        "ee_position_error": float(np.mean(np.linalg.norm(pred_pose[:, :2] - target_pose[:, :2], axis=1))),
        "ee_orientation_error": float(np.mean(np.abs(wrap_angle(pred_pose[:, 2] - target_pose[:, 2])))),
        "path_tracking_error": dtw_distance(prediction, target),
        "energy_proxy": energy_proxy(prediction),
        "smoothness_jerk": smoothness_jerk(prediction, dt=1.0),
        "constraint_violation_rate": constraint_violation_rate(prediction),
    }


def aggregate_rows(rows):
    keys = rows[0].keys()
    return {key: float(np.mean([row[key] for row in rows])) for key in keys}


def split_ranges(n_items, n_splits):
    step = int(math.ceil(n_items / n_splits))
    ranges = []
    for start in range(0, n_items, step):
        end = min(start + step, n_items)
        ranges.append((start, end))
    return ranges


def worker_compute(cache_path, kind, start, end):
    payload = np.load(cache_path, allow_pickle=True)
    q_predicted = payload["q_predicted"][start:end]
    q_reference = payload["q_reference"][start:end]
    shape_type = payload["shape_type"][start:end]

    rows = []
    shape_rows = defaultdict(list)
    for index in range(len(q_reference)):
        target = np.asarray(q_reference[index], dtype=np.float32)
        shape = str(shape_type[index])

        if kind == "generative":
            predictions = np.asarray(q_predicted[index], dtype=np.float32)
            sample_rows = [metric_row(prediction, target) for prediction in predictions]
            oracle_idx = int(np.argmin([row["joint_rmse"] for row in sample_rows]))
            oracle_row = dict(sample_rows[oracle_idx])
            mean_row = aggregate_rows(sample_rows)
            oracle_row["diversity_score"] = diversity_score(predictions)
            oracle_row["oracle_joint_rmse"] = oracle_row["joint_rmse"]
            oracle_row["mean_joint_rmse"] = mean_row["joint_rmse"]
            row = oracle_row
        else:
            prediction = np.asarray(q_predicted[index], dtype=np.float32)
            row = metric_row(prediction, target)
            row["diversity_score"] = 0.0
            row["oracle_joint_rmse"] = row["joint_rmse"]
            row["mean_joint_rmse"] = row["joint_rmse"]

        rows.append(row)
        shape_rows[shape].append(row)

    return rows, dict(shape_rows)


def summarize_cache_parallel(model_name):
    cache_path = CACHE_FILES[model_name]
    payload = np.load(cache_path, allow_pickle=True)
    n_samples = int(len(payload["q_reference"]))
    ranges = split_ranges(n_samples, N_WORKERS)

    rows = []
    shape_rows = defaultdict(list)
    with ProcessPoolExecutor(max_workers=N_WORKERS) as executor:
        futures = [
            executor.submit(worker_compute, cache_path, KIND[model_name], start, end)
            for start, end in ranges
        ]
        for future in futures:
            chunk_rows, chunk_shape_rows = future.result()
            rows.extend(chunk_rows)
            for shape, shape_row_list in chunk_shape_rows.items():
                shape_rows[shape].extend(shape_row_list)

    overall = aggregate_rows(rows)
    per_shape = {
        shape: aggregate_rows(shape_row_list)
        for shape, shape_row_list in shape_rows.items()
    }
    first_waypoints = np.asarray(payload["waypoints"][0], dtype=np.float32)
    first_target = np.asarray(payload["q_reference"][0], dtype=np.float32)
    first_pred = np.asarray(payload["q_oracle_predicted"][0], dtype=np.float32)
    overlay = {
        "condition": first_waypoints.tolist(),
        "target": first_target.tolist(),
        "prediction": first_pred.tolist(),
    }
    return overall, per_shape, overlay


def validate_cache(path, expected=4500):
    payload = np.load(path, allow_pickle=True)
    n_samples = int(len(payload["q_reference"]))
    if n_samples != expected:
        raise RuntimeError(f"Expected {expected} samples in {path}, found {n_samples}")


def main():
    for path in CACHE_FILES.values():
        validate_cache(path)

    with open("results/metrics/cvae/cvae_full_4500_evaluation_summary.json", "r", encoding="utf-8") as f:
        cvae_summary = json.load(f)

    dataset, normalizer = load_test_dataset(
        "data/processed",
        "data/metadata",
        max_samples=None,
        seed=42,
        subset_strategy="stratified",
    )
    model_specs = load_model_specs(
        device="cpu",
        include=MODEL_ORDER,
        include_diffusion_ddpm=False,
    )

    overall = {}
    per_shape = {}
    overlay = {}
    model_metadata = {}

    for model_name in ["analytical_ik", "cnn", "cnn_lstm", "cnn_gru", "diffusion_ddim"]:
        latency_ms, latency_metadata = _benchmark_model_latency(
            model_specs[model_name],
            dataset,
            normalizer,
            seed=42,
            subset_strategy="stratified",
            n_conditions=3,
            n_runs=10,
        )
        model_overall, model_per_shape, model_overlay = summarize_cache_parallel(model_name)
        model_overall["inference_time_ms"] = latency_ms
        for shape_name in model_per_shape:
            model_per_shape[shape_name]["inference_time_ms"] = latency_ms
        overall[model_name] = model_overall
        per_shape[model_name] = model_per_shape
        overlay[model_name] = model_overlay
        model_metadata[model_name] = {
            "kind": KIND[model_name],
            "latency_benchmark": latency_metadata,
            "source_cache": CACHE_FILES[model_name],
        }

    cvae_latency_ms, cvae_latency_metadata = _benchmark_model_latency(
        model_specs["cvae"],
        dataset,
        normalizer,
        seed=42,
        subset_strategy="stratified",
        n_conditions=3,
        n_runs=10,
    )
    overall["cvae"] = dict(cvae_summary["overall"])
    overall["cvae"]["inference_time_ms"] = cvae_latency_ms
    per_shape["cvae"] = dict(cvae_summary["per_shape"])
    for shape_name in per_shape["cvae"]:
        per_shape["cvae"][shape_name]["inference_time_ms"] = cvae_latency_ms
    cvae_payload = np.load(CACHE_FILES["cvae"], allow_pickle=True)
    overlay["cvae"] = {
        "condition": np.asarray(cvae_payload["waypoints"][0], dtype=np.float32).tolist(),
        "target": np.asarray(cvae_payload["q_reference"][0], dtype=np.float32).tolist(),
        "prediction": np.asarray(cvae_payload["q_oracle_predicted"][0], dtype=np.float32).tolist(),
    }
    model_metadata["cvae"] = {
        "kind": "generative",
        "latency_benchmark": cvae_latency_metadata,
        "source_cache": CACHE_FILES["cvae"],
        "source_summary": "results/metrics/cvae/cvae_full_4500_evaluation_summary.json",
    }

    ordered_overall = {model_name: overall[model_name] for model_name in MODEL_ORDER}
    ordered_per_shape = {model_name: per_shape[model_name] for model_name in MODEL_ORDER}
    ordered_overlay = {model_name: overlay[model_name] for model_name in MODEL_ORDER}
    ordered_metadata = {model_name: model_metadata[model_name] for model_name in MODEL_ORDER}

    dataset_meta = cvae_summary.get("dataset", {})
    summary = {
        "overall": ordered_overall,
        "per_shape": ordered_per_shape,
        "overlay": ordered_overlay,
        "num_test_samples": int(cvae_summary.get("num_test_samples", 4500)),
        "dataset": dataset_meta,
        "model_metadata": ordered_metadata,
    }

    write_summary_files(summary, "results/metrics/summary.json", "results/metrics/summary.md")
    write_summary_files(
        summary,
        "results/metrics/final_full_test/full_cross_model_summary.json",
        "results/metrics/final_full_test/full_cross_model_summary.md",
    )
    plot_main_comparison(summary, "figures/comparison/main_comparison.png")
    plot_trajectory_overlay(summary, "figures/comparison/trajectory_overlay.png")
    generate_result_plots(summary, "results/plots")

    print(f"Final full-test summary written for {summary['num_test_samples']} samples")


if __name__ == "__main__":
    main()
