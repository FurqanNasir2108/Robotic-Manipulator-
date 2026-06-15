"""Benchmark edge inference latency, throughput, and memory."""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.edge_deployment.config import load_edge_config
from src.edge_deployment.edge_benchmark import (
    benchmark_latency,
    benchmark_memory,
    benchmark_throughput,
    model_size_mb,
    pareto_analysis,
)
from src.edge_deployment.runtime_inference import ONNXInferenceSession


def main():
    parser = argparse.ArgumentParser(description="Benchmark edge inference")
    parser.add_argument("--config", default="configs/edge_deployment.yaml")
    args = parser.parse_args()

    cfg = load_edge_config(args.config)
    onnx_dir = cfg["paths"]["checkpoints_dir"]
    metrics_dir = cfg["paths"]["metrics_dir"]
    os.makedirs(metrics_dir, exist_ok=True)

    input_shape = cfg["export"]["input_shape"]
    test_inputs = [np.random.randn(*input_shape).astype(np.float32) for _ in range(100)]

    n_runs = cfg["benchmark"]["n_runs"]
    warmup = cfg["benchmark"]["warmup"]

    all_results = {}
    onnx_files = [f for f in os.listdir(onnx_dir) if f.endswith(".onnx")]

    for onnx_file in sorted(onnx_files):
        onnx_path = os.path.join(onnx_dir, onnx_file)
        model_name = onnx_file.replace(".onnx", "")
        print(f"\nBenchmarking {model_name}...")

        try:
            session = ONNXInferenceSession(onnx_path, device=cfg["benchmark"]["device"])
        except Exception as e:
            print(f"  Failed to load {onnx_file}: {e}")
            continue

        latency = benchmark_latency(session, test_inputs, n_runs=n_runs, warmup=warmup)
        throughput = benchmark_throughput(session, test_inputs, duration_sec=5.0)
        memory = benchmark_memory(session, test_inputs[0])
        size = model_size_mb(onnx_path)

        result = {
            **latency,
            "throughput_per_sec": throughput,
            **memory,
            "size_mb": size,
        }
        all_results[model_name] = result
        print(f"  Latency: {latency['median_ms']:.2f}ms (P95: {latency['p95_ms']:.2f}ms)")
        print(f"  Throughput: {throughput:.1f} traj/sec")
        print(f"  Size: {size:.2f} MB")

    # Save results
    with open(os.path.join(metrics_dir, "benchmark_results.json"), "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\nBenchmark results saved to {metrics_dir}")


if __name__ == "__main__":
    main()
