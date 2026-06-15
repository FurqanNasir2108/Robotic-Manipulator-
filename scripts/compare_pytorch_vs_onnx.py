"""Compare PyTorch vs ONNX model output numerically."""

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.edge_deployment.config import load_edge_config
from src.edge_deployment.model_export import validate_onnx_export
from src.evaluate.comparisons import _build_baseline_model
from src.utils.config import load_config


def main():
    parser = argparse.ArgumentParser(description="Compare PyTorch vs ONNX output")
    parser.add_argument("--config", default="configs/edge_deployment.yaml")
    parser.add_argument("--n-samples", type=int, default=100)
    args = parser.parse_args()

    cfg = load_edge_config(args.config)
    onnx_dir = cfg["paths"]["checkpoints_dir"]
    metrics_dir = cfg["paths"]["metrics_dir"]
    os.makedirs(metrics_dir, exist_ok=True)

    input_shape = cfg["export"]["input_shape"]
    results = {}

    config_map = {
        "cnn": "configs/baseline_cnn.yaml",
        "cnn_lstm": "configs/baseline_cnn_lstm.yaml",
        "cnn_gru": "configs/baseline_cnn_gru.yaml",
    }

    for model_name in cfg["models"]["include"]:
        if model_name not in config_map:
            continue

        onnx_path = os.path.join(onnx_dir, f"{model_name}_fp32.onnx")
        if not os.path.exists(onnx_path):
            print(f"Skipping {model_name}: ONNX file not found")
            continue

        model_cfg = load_config(config_map[model_name])
        model = _build_baseline_model(model_cfg)
        ckpt = torch.load(
            os.path.join("results", "checkpoints", model_name, f"{model_name}_best.pth"),
            map_location="cpu", weights_only=True,
        )
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()

        # Run multiple test inputs
        deviations = []
        for _ in range(args.n_samples):
            test_input = torch.randn(*input_shape)
            result = validate_onnx_export(model, onnx_path, test_input)
            deviations.append(result["max_deviation"])

        results[model_name] = {
            "mean_max_deviation": float(np.mean(deviations)),
            "max_max_deviation": float(np.max(deviations)),
            "all_match": all(d < cfg["validation"]["max_joint_rmse_deviation"] for d in deviations),
        }
        print(f"{model_name}: mean_max_dev={results[model_name]['mean_max_deviation']:.8f}, "
              f"all_match={results[model_name]['all_match']}")

    with open(os.path.join(metrics_dir, "pytorch_vs_onnx_comparison.json"), "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nComparison saved to {metrics_dir}/pytorch_vs_onnx_comparison.json")


if __name__ == "__main__":
    main()
