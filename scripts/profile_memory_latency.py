"""Profile memory and latency for deployed models."""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.edge_deployment.config import load_edge_config
from src.edge_deployment.device_monitor import DeviceMonitor
from src.edge_deployment.edge_benchmark import benchmark_latency, benchmark_memory, model_size_mb
from src.edge_deployment.runtime_inference import ONNXInferenceSession


def main():
    parser = argparse.ArgumentParser(description="Profile memory and latency")
    parser.add_argument("--config", default="configs/edge_deployment.yaml")
    args = parser.parse_args()

    cfg = load_edge_config(args.config)
    onnx_dir = cfg["paths"]["checkpoints_dir"]
    metrics_dir = cfg["paths"]["metrics_dir"]
    os.makedirs(metrics_dir, exist_ok=True)

    input_shape = cfg["export"]["input_shape"]
    test_inputs = [np.random.randn(*input_shape).astype(np.float32) for _ in range(50)]

    results = {}
    onnx_files = [f for f in os.listdir(onnx_dir) if f.endswith(".onnx")]

    for onnx_file in sorted(onnx_files):
        onnx_path = os.path.join(onnx_dir, onnx_file)
        model_name = onnx_file.replace(".onnx", "")
        print(f"\nProfiling {model_name}...")

        try:
            session = ONNXInferenceSession(onnx_path)
        except Exception as e:
            print(f"  Failed: {e}")
            continue

        monitor = DeviceMonitor()
        monitor.start()

        # Run inference with monitoring
        for inp in test_inputs[:20]:
            session.predict(inp)
            monitor.sample()

        monitor.stop()

        latency = benchmark_latency(session, test_inputs, n_runs=500, warmup=50)
        memory = benchmark_memory(session, test_inputs[0])
        size = model_size_mb(onnx_path)

        results[model_name] = {
            **latency,
            **memory,
            "size_mb": size,
            "monitor_summary": monitor.get_summary(),
        }
        print(f"  Latency: {latency['median_ms']:.2f}ms, RAM: {memory.get('peak_ram_mb', -1):.1f}MB, "
              f"Size: {size:.2f}MB")

    with open(os.path.join(metrics_dir, "memory_latency_profile.json"), "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nProfile saved to {metrics_dir}/memory_latency_profile.json")


if __name__ == "__main__":
    main()
