"""Utilities for evaluating and comparing trained models."""

from __future__ import annotations

import json
import os
from collections import Counter, defaultdict

import matplotlib.pyplot as plt
import numpy as np
import torch

from src.data.loaders import TrajectoryDataset
from src.data.normalization import Normalizer
from src.evaluate.metrics import (
    constraint_violation_rate,
    diversity_score,
    ee_orientation_error,
    ee_position_error,
    energy_proxy,
    inference_time,
    joint_rmse,
    path_tracking_error,
    smoothness_jerk,
)
from src.models.analytical_ik import AnalyticalIKBaseline
from src.models.cnn_baseline import CNNTrajectoryRegressor
from src.models.cnn_gru import CNNGRURegressor
from src.models.cnn_lstm import CNNLSTMRegressor
from src.models.cvae import ConditionalVAE
from src.models.diffusion import build_diffusion_model
from src.simulation.manipulator import ThreeLinkManipulator
from src.utils.config import load_config


def _build_baseline_model(cfg):
    model_cfg = cfg["model"]
    model_type = model_cfg["type"]
    if model_type == "cnn":
        return CNNTrajectoryRegressor(
            input_steps=model_cfg["input_steps"],
            input_dim=model_cfg["input_dim"],
            output_steps=model_cfg["output_steps"],
            output_dim=model_cfg["output_dim"],
            conv_channels=model_cfg["conv_channels"],
            kernel_sizes=model_cfg["kernel_sizes"],
            fc_dims=model_cfg["fc_dims"],
            activation=model_cfg["activation"],
            dropout=model_cfg["dropout"],
        )
    if model_type == "cnn_lstm":
        return CNNLSTMRegressor(
            input_steps=model_cfg["input_steps"],
            input_dim=model_cfg["input_dim"],
            output_steps=model_cfg["output_steps"],
            output_dim=model_cfg["output_dim"],
            conv_channels=model_cfg["conv_channels"],
            kernel_sizes=model_cfg["kernel_sizes"],
            lstm_hidden=model_cfg["lstm_hidden"],
            lstm_layers=model_cfg["lstm_layers"],
            lstm_dropout=model_cfg["lstm_dropout"],
            fc_dims=model_cfg["fc_dims"],
            activation=model_cfg["activation"],
            dropout=model_cfg["dropout"],
        )
    if model_type == "cnn_gru":
        return CNNGRURegressor(
            input_steps=model_cfg["input_steps"],
            input_dim=model_cfg["input_dim"],
            output_steps=model_cfg["output_steps"],
            output_dim=model_cfg["output_dim"],
            conv_channels=model_cfg["conv_channels"],
            kernel_sizes=model_cfg["kernel_sizes"],
            gru_hidden=model_cfg["gru_hidden"],
            gru_layers=model_cfg["gru_layers"],
            gru_dropout=model_cfg["gru_dropout"],
            fc_dims=model_cfg["fc_dims"],
            activation=model_cfg["activation"],
            dropout=model_cfg["dropout"],
        )
    raise ValueError(f"Unknown baseline model type: {model_type}")


def _build_cvae_model(cfg):
    model_cfg = cfg["model"]
    return ConditionalVAE(
        latent_dim=model_cfg["latent_dim"],
        encoder_type=model_cfg["encoder_type"],
        decoder_type=model_cfg["decoder_type"],
        condition_dim=model_cfg["condition_dim"],
        input_dim=model_cfg["input_dim"],
        output_dim=model_cfg["output_dim"],
        input_steps=model_cfg["input_steps"],
        output_steps=model_cfg["output_steps"],
        cond_conv_channels=model_cfg.get("cond_conv_channels"),
        cond_kernel_sizes=model_cfg.get("cond_kernel_sizes"),
        enc_hidden=model_cfg.get("enc_hidden", 128),
        enc_layers=model_cfg.get("enc_layers", 2),
        enc_dropout=model_cfg.get("enc_dropout", 0.1),
        dec_hidden=model_cfg.get("dec_hidden", 128),
        dec_layers=model_cfg.get("dec_layers", 2),
        dec_dropout=model_cfg.get("dec_dropout", 0.1),
        dec_fc_dims=model_cfg.get("dec_fc_dims"),
        dec_conv_channels=model_cfg.get("dec_conv_channels"),
        dec_kernel_sizes=model_cfg.get("dec_kernel_sizes"),
    )


def _shape_counts(shape_types) -> dict[str, int]:
    counts = Counter(str(shape) for shape in shape_types)
    return dict(sorted(counts.items()))


def _slice_dataset(dataset: TrajectoryDataset, indices: np.ndarray) -> TrajectoryDataset:
    dataset.waypoints = dataset.waypoints[indices]
    dataset.q_sequence = dataset.q_sequence[indices]
    dataset.start_pose = dataset.start_pose[indices]
    dataset.goal_pose = dataset.goal_pose[indices]
    dataset.shape_type = dataset.shape_type[indices]
    return dataset


def _allocate_stratified_counts(shape_counts: dict[str, int], max_samples: int) -> dict[str, int]:
    """Allocate a subset size across shapes while preserving coverage when possible."""
    shapes = sorted(shape_counts)
    allocation = {shape: 0 for shape in shapes}

    if max_samples <= 0 or not shapes:
        return allocation

    residual_capacity = dict(shape_counts)
    if max_samples >= len(shapes):
        for shape in shapes:
            allocation[shape] = 1
            residual_capacity[shape] -= 1
        remaining = max_samples - len(shapes)
    else:
        remaining = max_samples

    total_residual = sum(residual_capacity.values())
    if remaining <= 0 or total_residual <= 0:
        return allocation

    exact = {
        shape: remaining * residual_capacity[shape] / total_residual
        for shape in shapes
    }
    allocated = 0
    for shape in shapes:
        add = min(int(np.floor(exact[shape])), residual_capacity[shape])
        allocation[shape] += add
        allocated += add

    leftover = remaining - allocated
    order = sorted(
        shapes,
        key=lambda shape: (exact[shape] - np.floor(exact[shape]), residual_capacity[shape]),
        reverse=True,
    )

    while leftover > 0:
        progressed = False
        for shape in order:
            if allocation[shape] < shape_counts[shape]:
                allocation[shape] += 1
                leftover -= 1
                progressed = True
                if leftover == 0:
                    break
        if not progressed:
            break

    return allocation


def _sample_subset_indices(shape_types, max_samples=None, seed=42, strategy="stratified") -> np.ndarray:
    """Select a reproducible evaluation subset."""
    n_samples = len(shape_types)
    all_indices = np.arange(n_samples, dtype=int)
    if max_samples is None or max_samples >= n_samples:
        return all_indices
    if max_samples <= 0:
        return np.array([], dtype=int)

    rng = np.random.default_rng(seed)
    if strategy == "ordered":
        return all_indices[:max_samples]
    if strategy == "random":
        selected = rng.choice(all_indices, size=max_samples, replace=False)
        return np.sort(selected)
    if strategy != "stratified":
        raise ValueError(f"Unknown subset strategy: {strategy}")

    by_shape = defaultdict(list)
    for index, shape in enumerate(shape_types):
        by_shape[str(shape)].append(index)

    allocation = _allocate_stratified_counts(
        {shape: len(indices) for shape, indices in by_shape.items()},
        max_samples=max_samples,
    )

    selected = []
    for shape, indices in by_shape.items():
        take = allocation.get(shape, 0)
        if take <= 0:
            continue
        chosen = rng.choice(np.asarray(indices, dtype=int), size=take, replace=False)
        selected.extend(chosen.tolist())

    return np.sort(np.asarray(selected, dtype=int))


def load_test_dataset(data_dir, metadata_dir, max_samples=None, seed=42, subset_strategy="stratified"):
    """Load normalized test dataset and optionally sample a reproducible subset."""
    normalizer = Normalizer.load(os.path.join(metadata_dir, "normalization_stats.json"))
    dataset = TrajectoryDataset(
        os.path.join(data_dir, "test.npz"),
        normalizer=normalizer,
        input_mode="waypoints",
    )
    dataset.original_size = len(dataset)
    dataset.original_shape_counts = _shape_counts(dataset.shape_type)

    if max_samples is not None and max_samples < len(dataset):
        indices = _sample_subset_indices(
            dataset.shape_type,
            max_samples=max_samples,
            seed=seed,
            strategy=subset_strategy,
        )
        _slice_dataset(dataset, indices)
        dataset.selected_indices = indices
        dataset.subset_strategy = subset_strategy
    else:
        dataset.selected_indices = np.arange(len(dataset), dtype=int)
        dataset.subset_strategy = "full"

    dataset.shape_counts = _shape_counts(dataset.shape_type)
    return dataset, normalizer


def load_model_specs(device="cpu", include=None, include_diffusion_ddpm=False):
    """Load all requested trained models and expose a unified inference interface."""
    specs = {}

    if include is None:
        include = ["analytical_ik", "cnn", "cnn_lstm", "cnn_gru", "cvae", "diffusion_ddim"]

    if "analytical_ik" in include:
        model = AnalyticalIKBaseline()
        specs["analytical_ik"] = {
            "kind": "deterministic",
            "predict": lambda condition, num_samples=1, model=model: np.expand_dims(
                model.predict(condition), axis=0
            ),
            "timing": lambda condition, model=model: model.predict(condition),
            "uses_raw_waypoints": True,
            "outputs_normalized": False,
        }

    for name, cfg_path in {
        "cnn": "configs/baseline_cnn.yaml",
        "cnn_lstm": "configs/baseline_cnn_lstm.yaml",
        "cnn_gru": "configs/baseline_cnn_gru.yaml",
    }.items():
        if name not in include:
            continue
        cfg = load_config(cfg_path)
        model = _build_baseline_model(cfg).to(device)
        ckpt = torch.load(
            os.path.join("results", "checkpoints", name, f"{name}_best.pth"),
            map_location=device,
            weights_only=True,
        )
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        specs[name] = {
            "kind": "deterministic",
            "predict": lambda condition, num_samples=1, model=model: model(
                torch.from_numpy(condition).unsqueeze(0).to(device, dtype=torch.float32)
            ).detach().cpu().numpy(),
            "timing": lambda condition, model=model: model(
                torch.from_numpy(condition).unsqueeze(0).to(device, dtype=torch.float32)
            ),
            "uses_raw_waypoints": False,
            "outputs_normalized": True,
        }

    if "cvae" in include:
        cfg = load_config("configs/vae.yaml")
        model = _build_cvae_model(cfg).to(device)
        ckpt = torch.load(
            os.path.join("results", "checkpoints", "cvae", "cvae_best.pth"),
            map_location=device,
            weights_only=True,
        )
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        specs["cvae"] = {
            "kind": "generative",
            "predict": lambda condition, num_samples=10, model=model: model.generate(
                torch.from_numpy(condition).unsqueeze(0).to(device, dtype=torch.float32),
                num_samples=num_samples,
            ).detach().cpu().numpy(),
            "timing": lambda condition, model=model: model.generate(
                torch.from_numpy(condition).unsqueeze(0).to(device, dtype=torch.float32),
                num_samples=1,
            ),
            "uses_raw_waypoints": False,
            "outputs_normalized": True,
        }

    if "diffusion_ddim" in include or include_diffusion_ddpm:
        cfg = load_config("configs/diffusion.yaml")
        model, ema = build_diffusion_model(cfg)
        ckpt = torch.load(
            os.path.join("results", "checkpoints", "diffusion", "diffusion_best.pth"),
            map_location=device,
            weights_only=True,
        )
        # Use the trained model weights directly. The EMA shadow has stale
        # BatchNorm running statistics (EMA only tracked parameters, not
        # BN buffers), making its predictions unreliable.
        model.load_state_dict(ckpt["model_state_dict"])
        model = model.to(device)
        model.eval()

        if "diffusion_ddim" in include:
            specs["diffusion_ddim"] = {
                "kind": "generative",
                "predict": lambda condition, num_samples=10, model=model, cfg=cfg: model.generate(
                    torch.from_numpy(condition).unsqueeze(0).to(device, dtype=torch.float32),
                    num_samples=num_samples,
                    method="ddim",
                    num_steps=cfg["sampling"].get("ddim_steps", 50),
                    guidance_scale=cfg["sampling"].get("guidance_scale", 2.0),
                    seq_len=cfg["sampling"].get("seq_len", 100),
                ).detach().cpu().numpy(),
                "timing": lambda condition, model=model, cfg=cfg: model.generate(
                    torch.from_numpy(condition).unsqueeze(0).to(device, dtype=torch.float32),
                    num_samples=1,
                    method="ddim",
                    num_steps=cfg["sampling"].get("ddim_steps", 50),
                    guidance_scale=cfg["sampling"].get("guidance_scale", 2.0),
                    seq_len=cfg["sampling"].get("seq_len", 100),
                ),
                "uses_raw_waypoints": False,
                "outputs_normalized": True,
            }

        if include_diffusion_ddpm:
            specs["diffusion_ddpm"] = {
                "kind": "generative",
                "predict": lambda condition, num_samples=10, model=model, cfg=cfg: model.generate(
                    torch.from_numpy(condition).unsqueeze(0).to(device, dtype=torch.float32),
                    num_samples=num_samples,
                    method="ddpm",
                    guidance_scale=cfg["sampling"].get("guidance_scale", 2.0),
                    seq_len=cfg["sampling"].get("seq_len", 100),
                ).detach().cpu().numpy(),
                "timing": lambda condition, model=model, cfg=cfg: model.generate(
                    torch.from_numpy(condition).unsqueeze(0).to(device, dtype=torch.float32),
                    num_samples=1,
                    method="ddpm",
                    guidance_scale=cfg["sampling"].get("guidance_scale", 2.0),
                    seq_len=cfg["sampling"].get("seq_len", 100),
                ),
                "uses_raw_waypoints": False,
                "outputs_normalized": True,
            }

    return specs


def _metric_row(prediction, target, fk_fn, joint_limits, dt=1.0, path_method="dtw"):
    return {
        "joint_rmse": joint_rmse(prediction, target),
        "ee_position_error": ee_position_error(prediction, target, fk_fn),
        "ee_orientation_error": ee_orientation_error(prediction, target, fk_fn),
        "path_tracking_error": path_tracking_error(prediction, target, method=path_method),
        "energy_proxy": energy_proxy(prediction, method="velocity"),
        "smoothness_jerk": smoothness_jerk(prediction, dt=dt),
        "constraint_violation_rate": constraint_violation_rate(prediction, joint_limits),
    }


def _aggregate_rows(rows):
    metric_names = rows[0].keys()
    return {name: float(np.mean([row[name] for row in rows])) for name in metric_names}


def _extract_sample(dataset, normalizer, index, uses_raw_waypoints=False):
    condition_tensor, target_tensor = dataset[index]
    condition_norm = condition_tensor.numpy()
    target_norm = target_tensor.numpy()
    target = normalizer.inverse_transform(target_norm, "q_sequence")
    waypoints = normalizer.inverse_transform(condition_norm, "waypoints")
    shape = str(dataset.shape_type[index])
    model_input = waypoints if uses_raw_waypoints else condition_norm
    return {
        "condition_norm": condition_norm,
        "target": target,
        "waypoints": waypoints,
        "shape": shape,
        "model_input": model_input,
    }


def _benchmark_model_latency(
    spec,
    dataset,
    normalizer,
    seed=42,
    subset_strategy="stratified",
    n_conditions=3,
    n_runs=10,
):
    """Benchmark latency once per model on a representative condition subset."""
    if len(dataset) == 0:
        return 0.0, {"num_conditions": 0, "condition_indices": [], "per_condition_ms": []}

    n_conditions = max(1, min(int(n_conditions), len(dataset)))
    indices = _sample_subset_indices(
        dataset.shape_type,
        max_samples=n_conditions,
        seed=seed,
        strategy=subset_strategy,
    )

    latencies = []
    for index in indices:
        sample = _extract_sample(
            dataset,
            normalizer,
            index=index,
            uses_raw_waypoints=spec.get("uses_raw_waypoints", False),
        )
        latencies.append(
            inference_time(
                spec["timing"],
                sample["model_input"],
                n_runs=n_runs,
            )
        )

    return float(np.mean(latencies)), {
        "num_conditions": len(indices),
        "condition_indices": indices.tolist(),
        "per_condition_ms": [float(value) for value in latencies],
    }


def _dataset_metadata(dataset):
    metadata = {
        "original_size": int(getattr(dataset, "original_size", len(dataset))),
        "selected_size": int(len(dataset)),
        "subset_strategy": getattr(dataset, "subset_strategy", "full"),
        "shape_counts": _shape_counts(dataset.shape_type),
        "original_shape_counts": getattr(dataset, "original_shape_counts", _shape_counts(dataset.shape_type)),
    }
    if metadata["selected_size"] != metadata["original_size"]:
        metadata["selected_indices"] = getattr(dataset, "selected_indices", np.arange(len(dataset))).tolist()
    return metadata


def evaluate_model_set(model_specs, dataset, normalizer, config):
    """Evaluate all loaded models on the test dataset."""
    robot = ThreeLinkManipulator()
    evaluation_cfg = config["evaluation"]
    metrics_cfg = config["metrics"]
    generative_samples = evaluation_cfg.get("generative_samples", 10)
    inference_runs = evaluation_cfg.get("inference_runs", 10)
    latency_num_conditions = evaluation_cfg.get("latency_num_conditions", 3)
    latency_subset_strategy = evaluation_cfg.get("latency_subset_strategy", "stratified")
    cache_dir = evaluation_cfg.get("cache_trajectories_dir", None)
    cache_full_generative_samples = evaluation_cfg.get("cache_full_generative_samples", True)
    seed = evaluation_cfg.get("seed", 42)

    overall_summary = {}
    per_shape_summary = {}
    overlay_payload = {}
    model_metadata = {}

    for model_name, spec in model_specs.items():
        model_latency_ms, latency_metadata = _benchmark_model_latency(
            spec,
            dataset,
            normalizer,
            seed=seed,
            subset_strategy=latency_subset_strategy,
            n_conditions=latency_num_conditions,
            n_runs=inference_runs,
        )

        rows = []
        shape_rows = defaultdict(list)
        cached_predictions = []
        cached_oracle_predictions = []
        cached_references = []
        cached_shapes = []
        cached_waypoints = []

        for index in range(len(dataset)):
            sample = _extract_sample(
                dataset,
                normalizer,
                index=index,
                uses_raw_waypoints=spec.get("uses_raw_waypoints", False),
            )
            target = sample["target"]
            waypoints = sample["waypoints"]
            shape = sample["shape"]
            model_input = sample["model_input"]

            if spec["kind"] == "generative":
                predictions_array = spec["predict"](model_input, num_samples=generative_samples)
                if spec.get("outputs_normalized", True):
                    predictions = normalizer.inverse_transform(predictions_array, "q_sequence")
                else:
                    predictions = predictions_array
                predictions = np.asarray(predictions)

                sample_rows = [
                    _metric_row(
                        prediction,
                        target,
                        robot.forward_kinematics,
                        robot.joint_limits,
                        dt=metrics_cfg.get("dt", 1.0),
                        path_method=metrics_cfg.get("path_method", "dtw"),
                    )
                    for prediction in predictions
                ]
                oracle_idx = int(np.argmin([row["joint_rmse"] for row in sample_rows]))
                oracle_row = dict(sample_rows[oracle_idx])
                mean_row = _aggregate_rows(sample_rows)
                oracle_row["diversity_score"] = diversity_score(
                    predictions,
                    method=metrics_cfg.get("diversity_method", "dtw"),
                )
                oracle_row["inference_time_ms"] = model_latency_ms
                oracle_row["oracle_joint_rmse"] = oracle_row["joint_rmse"]
                oracle_row["mean_joint_rmse"] = mean_row["joint_rmse"]
                row = oracle_row
                best_prediction = predictions[oracle_idx]
                cached_prediction = predictions if cache_full_generative_samples else best_prediction
            else:
                prediction_array = spec["predict"](model_input, num_samples=1)[0]
                if spec.get("outputs_normalized", True):
                    prediction = normalizer.inverse_transform(prediction_array, "q_sequence")
                else:
                    prediction = prediction_array
                row = _metric_row(
                    prediction,
                    target,
                    robot.forward_kinematics,
                    robot.joint_limits,
                    dt=metrics_cfg.get("dt", 1.0),
                    path_method=metrics_cfg.get("path_method", "dtw"),
                )
                row["diversity_score"] = 0.0
                row["inference_time_ms"] = model_latency_ms
                row["oracle_joint_rmse"] = row["joint_rmse"]
                row["mean_joint_rmse"] = row["joint_rmse"]
                best_prediction = prediction
                cached_prediction = prediction

            rows.append(row)
            shape_rows[shape].append(row)
            cached_predictions.append(cached_prediction)
            cached_oracle_predictions.append(best_prediction)
            cached_references.append(target)
            cached_shapes.append(shape)
            cached_waypoints.append(waypoints)

            if index == 0:
                overlay_payload[model_name] = {
                    "condition": waypoints.tolist(),
                    "target": target.tolist(),
                    "prediction": best_prediction.tolist(),
                }

        overall_summary[model_name] = _aggregate_rows(rows)
        per_shape_summary[model_name] = {
            shape: _aggregate_rows(shape_metric_rows)
            for shape, shape_metric_rows in shape_rows.items()
        }
        model_metadata[model_name] = {
            "kind": spec["kind"],
            "latency_benchmark": latency_metadata,
        }

        if cache_dir is not None:
            os.makedirs(cache_dir, exist_ok=True)
            np.savez_compressed(
                os.path.join(cache_dir, f"{model_name}.npz"),
                q_predicted=np.asarray(cached_predictions),
                q_oracle_predicted=np.asarray(cached_oracle_predictions),
                q_reference=np.asarray(cached_references),
                shape_type=np.asarray(cached_shapes),
                waypoints=np.asarray(cached_waypoints),
            )

    return {
        "overall": overall_summary,
        "per_shape": per_shape_summary,
        "overlay": overlay_payload,
        "num_test_samples": len(dataset),
        "dataset": _dataset_metadata(dataset),
        "model_metadata": model_metadata,
    }


def write_summary_files(summary, summary_json_path, summary_md_path):
    """Persist JSON and markdown summaries."""
    os.makedirs(os.path.dirname(summary_json_path), exist_ok=True)
    with open(summary_json_path, "w") as f:
        json.dump(summary, f, indent=2)

    os.makedirs(os.path.dirname(summary_md_path), exist_ok=True)
    headers = [
        "Model",
        "Joint RMSE",
        "EE Pos",
        "EE Orient",
        "Path",
        "Energy",
        "Smoothness",
        "Inference (ms)",
        "Diversity",
    ]
    lines = ["# Evaluation Summary", ""]

    dataset_metadata = summary.get("dataset", {})
    if dataset_metadata:
        lines.append(f"- Test samples evaluated: {summary.get('num_test_samples', 0)}")
        lines.append(f"- Subset strategy: {dataset_metadata.get('subset_strategy', 'full')}")
        shape_counts = dataset_metadata.get("shape_counts", {})
        if shape_counts:
            rendered_counts = ", ".join(f"{shape}={count}" for shape, count in shape_counts.items())
            lines.append(f"- Shape counts: {rendered_counts}")
        lines.append("")

    lines.extend(
        [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join(["---"] * len(headers)) + " |",
        ]
    )
    for model_name, metrics in summary["overall"].items():
        diversity = metrics["diversity_score"] if metrics["diversity_score"] > 0 else "N/A"
        lines.append(
            "| "
            + " | ".join(
                [
                    model_name,
                    f"{metrics['joint_rmse']:.4f}",
                    f"{metrics['ee_position_error']:.4f}",
                    f"{metrics['ee_orientation_error']:.4f}",
                    f"{metrics['path_tracking_error']:.4f}",
                    f"{metrics['energy_proxy']:.4f}",
                    f"{metrics['smoothness_jerk']:.4f}",
                    f"{metrics['inference_time_ms']:.2f}",
                    f"{diversity:.4f}" if diversity != "N/A" else diversity,
                ]
            )
            + " |"
        )

    lines.extend(["", "## Per-Shape Breakdown", ""])
    for model_name, shapes in summary["per_shape"].items():
        lines.append(f"### {model_name}")
        lines.append("")
        lines.append("| Shape | Joint RMSE | EE Pos | Path |")
        lines.append("| --- | --- | --- | --- |")
        for shape_name, metrics in shapes.items():
            lines.append(
                f"| {shape_name} | {metrics['joint_rmse']:.4f} | {metrics['ee_position_error']:.4f} | {metrics['path_tracking_error']:.4f} |"
            )
        lines.append("")

    with open(summary_md_path, "w") as f:
        f.write("\n".join(lines))


def plot_main_comparison(summary, figure_path):
    """Create a compact bar chart for key metrics."""
    os.makedirs(os.path.dirname(figure_path), exist_ok=True)
    model_names = list(summary["overall"].keys())
    joint_vals = [summary["overall"][name]["joint_rmse"] for name in model_names]
    ee_vals = [summary["overall"][name]["ee_position_error"] for name in model_names]
    infer_vals = [summary["overall"][name]["inference_time_ms"] for name in model_names]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    axes[0].bar(model_names, joint_vals, color="#c0392b")
    axes[0].set_title("Joint RMSE")
    axes[0].tick_params(axis="x", rotation=35)
    axes[0].grid(True, alpha=0.3, axis="y")

    axes[1].bar(model_names, ee_vals, color="#1f618d")
    axes[1].set_title("EE Position Error")
    axes[1].tick_params(axis="x", rotation=35)
    axes[1].grid(True, alpha=0.3, axis="y")

    axes[2].bar(model_names, infer_vals, color="#117a65")
    axes[2].set_title("Inference Time (ms)")
    axes[2].tick_params(axis="x", rotation=35)
    axes[2].grid(True, alpha=0.3, axis="y")

    plt.suptitle("Model Comparison Overview", fontsize=15, fontweight="bold")
    plt.tight_layout()
    plt.savefig(figure_path, dpi=300)
    plt.close()


def plot_trajectory_overlay(summary, figure_path):
    """Overlay first-condition predictions from all models in task space."""
    if not summary.get("overlay"):
        return

    os.makedirs(os.path.dirname(figure_path), exist_ok=True)
    first_model = next(iter(summary["overlay"]))
    condition = np.asarray(summary["overlay"][first_model]["condition"])
    target = np.asarray(summary["overlay"][first_model]["target"])
    robot = ThreeLinkManipulator()

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    target_ee = np.asarray([robot.forward_kinematics(q) for q in target])
    ax.plot(condition[:, 0], condition[:, 1], "--", color="black", linewidth=1.0, label="Condition")
    ax.plot(target_ee[:, 0], target_ee[:, 1], color="#2c3e50", linewidth=2.2, label="Ground Truth")

    colors = plt.cm.tab10(np.linspace(0, 1, len(summary["overlay"])))
    for color, (model_name, payload) in zip(colors, summary["overlay"].items()):
        prediction = np.asarray(payload["prediction"])
        ee_path = np.asarray([robot.forward_kinematics(q) for q in prediction])
        ax.plot(
            ee_path[:, 0],
            ee_path[:, 1],
            color=color,
            linewidth=1.4,
            alpha=0.9,
            label=model_name,
        )

    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title("Trajectory Overlay on First Test Condition")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(figure_path, dpi=300)
    plt.close()
