"""Report generation for trajectory analysis: CSV, JSON, Markdown."""

from __future__ import annotations

import csv
import json
import os

import numpy as np

from src.trajectory_analysis.statistical_tests import compute_statistics


def _ensure_dir(path: str):
    os.makedirs(os.path.dirname(path) if not os.path.isdir(path) else path, exist_ok=True)


def generate_per_trajectory_csv(all_metrics: list[dict], output_path: str):
    """Save per-trajectory metrics as CSV.

    Parameters
    ----------
    all_metrics : list of dict
        Each dict has keys: sample_id, model_name, shape_type, + metric values.
    output_path : str
    """
    if not all_metrics:
        return
    _ensure_dir(os.path.dirname(output_path))
    fieldnames = list(all_metrics[0].keys())
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in all_metrics:
            writer.writerow({k: _serialize(v) for k, v in row.items()})


def generate_aggregate_json(all_metrics: list[dict], output_path: str):
    """Aggregate per-trajectory metrics and save as JSON.

    Groups by model_name, computes statistics for each numeric metric.
    """
    if not all_metrics:
        return
    _ensure_dir(os.path.dirname(output_path))

    by_model = {}
    for row in all_metrics:
        model = row.get("model_name", "unknown")
        by_model.setdefault(model, []).append(row)

    summary = {}
    for model, rows in by_model.items():
        model_summary = {}
        numeric_keys = [k for k in rows[0] if isinstance(rows[0][k], (int, float))]
        for key in numeric_keys:
            values = [r[key] for r in rows if isinstance(r.get(key), (int, float))]
            model_summary[key] = compute_statistics(values)
        summary[model] = model_summary

    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2, default=_serialize)


def generate_per_shape_report(all_metrics: list[dict], output_path: str):
    """Aggregate metrics per shape type per model and save as JSON."""
    if not all_metrics:
        return
    _ensure_dir(os.path.dirname(output_path))

    by_model_shape = {}
    for row in all_metrics:
        model = row.get("model_name", "unknown")
        shape = row.get("shape_type", "unknown")
        by_model_shape.setdefault(model, {}).setdefault(shape, []).append(row)

    summary = {}
    for model, shapes in by_model_shape.items():
        summary[model] = {}
        for shape, rows in shapes.items():
            shape_summary = {}
            numeric_keys = [k for k in rows[0] if isinstance(rows[0][k], (int, float))]
            for key in numeric_keys:
                values = [r[key] for r in rows if isinstance(r.get(key), (int, float))]
                shape_summary[key] = compute_statistics(values)
            summary[model][shape] = shape_summary

    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2, default=_serialize)


def generate_per_model_report(all_metrics: list[dict], output_path: str):
    """Alias for generate_aggregate_json with model-level grouping."""
    generate_aggregate_json(all_metrics, output_path)


def generate_comparison_table(all_metrics: list[dict], output_path: str):
    """Generate a Markdown comparison table across models.

    Parameters
    ----------
    all_metrics : list of dict
    output_path : str
        Path to .md file.
    """
    if not all_metrics:
        return
    _ensure_dir(os.path.dirname(output_path))

    by_model = {}
    for row in all_metrics:
        model = row.get("model_name", "unknown")
        by_model.setdefault(model, []).append(row)

    # Find numeric metric keys
    sample_row = all_metrics[0]
    metric_keys = sorted(k for k in sample_row if isinstance(sample_row[k], (int, float)))

    lines = ["# Trajectory Analysis — Model Comparison\n"]
    header = "| Model | " + " | ".join(metric_keys) + " |"
    sep = "|-------|" + "|".join(["-------"] * len(metric_keys)) + "|"
    lines.append(header)
    lines.append(sep)

    for model, rows in sorted(by_model.items()):
        values = []
        for key in metric_keys:
            vals = [r[key] for r in rows if isinstance(r.get(key), (int, float))]
            mean = float(np.mean(vals)) if vals else 0.0
            values.append(f"{mean:.6f}")
        lines.append(f"| {model} | " + " | ".join(values) + " |")

    lines.append("")
    with open(output_path, "w") as f:
        f.write("\n".join(lines))


def generate_analysis_summary_md(all_metrics: list[dict], output_path: str):
    """Generate a full Markdown summary report.

    Includes comparison table, per-shape breakdown, and key findings.
    """
    if not all_metrics:
        return
    _ensure_dir(os.path.dirname(output_path))

    lines = ["# Trajectory Analysis Summary\n"]
    lines.append(f"Total samples analyzed: {len(all_metrics)}\n")

    # Models
    models = sorted(set(r.get("model_name", "unknown") for r in all_metrics))
    lines.append(f"Models: {', '.join(models)}\n")

    # Shapes
    shapes = sorted(set(r.get("shape_type", "unknown") for r in all_metrics))
    lines.append(f"Shapes: {', '.join(shapes)}\n")

    lines.append("---\n")
    lines.append("See `comparison_table.md` for detailed model comparison.\n")
    lines.append("See `per_shape_metrics.json` for shape-level breakdown.\n")
    lines.append("See `per_trajectory_metrics.csv` for raw per-sample data.\n")

    with open(output_path, "w") as f:
        f.write("\n".join(lines))


def _serialize(obj):
    """JSON-safe serialization helper."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj
