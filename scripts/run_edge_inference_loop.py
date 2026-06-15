"""Real-time edge inference loop: sensor → model → trajectory → controller."""

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.edge_deployment.config import load_edge_config
from src.edge_deployment.controller_interface import SimulatorController
from src.edge_deployment.device_monitor import DeviceMonitor
from src.edge_deployment.runtime_inference import ONNXInferenceSession
from src.edge_deployment.sensor_interface import SensorPreprocessor, SimulatedSensorStream


def main():
    parser = argparse.ArgumentParser(description="Run edge inference loop")
    parser.add_argument("--config", default="configs/edge_deployment.yaml")
    parser.add_argument("--onnx-model", default=None)
    parser.add_argument("--n-iterations", type=int, default=100)
    parser.add_argument("--data-dir", default="data/processed")
    args = parser.parse_args()

    cfg = load_edge_config(args.config)

    # Default to primary model
    if args.onnx_model is None:
        primary = cfg["deployment"]["primary_model"]
        args.onnx_model = os.path.join(cfg["paths"]["checkpoints_dir"], f"{primary}_fp32.onnx")

    # Load model
    session = ONNXInferenceSession(args.onnx_model, device=cfg["benchmark"]["device"])

    # Load test data for simulated sensor
    test_data = np.load(os.path.join(args.data_dir, "test.npz"), allow_pickle=True)
    sensor = SimulatedSensorStream(test_data)
    preprocessor = SensorPreprocessor()
    controller = SimulatorController()
    monitor = DeviceMonitor()

    # Run loop
    logs = []
    monitor.start()
    print(f"Running {args.n_iterations} inference iterations...")

    for i in range(args.n_iterations):
        condition = sensor.get_next_condition()
        processed = preprocessor.preprocess(condition)

        if processed.ndim == 2:
            processed = processed[np.newaxis, :]

        start = time.perf_counter()
        trajectory = session.predict(processed)
        latency_ms = (time.perf_counter() - start) * 1000.0

        trajectory = trajectory.squeeze(0) if trajectory.ndim == 3 else trajectory

        execution = controller.execute_trajectory(trajectory)
        monitor.sample()

        log_entry = {
            "iteration": i,
            "latency_ms": latency_ms,
            "success": execution["success"],
            "violation_rate": execution["violation_rate"],
        }
        logs.append(log_entry)

        if (i + 1) % 25 == 0:
            print(f"  Iteration {i + 1}/{args.n_iterations}: "
                  f"latency={latency_ms:.2f}ms, success={execution['success']}")

    monitor.stop()

    # Save logs
    metrics_dir = cfg["paths"]["metrics_dir"]
    os.makedirs(metrics_dir, exist_ok=True)
    with open(os.path.join(metrics_dir, "realtime_loop_log.json"), "w") as f:
        json.dump({"logs": logs, "monitor_summary": monitor.get_summary()}, f, indent=2)

    latencies = [l["latency_ms"] for l in logs]
    successes = [l["success"] for l in logs]
    print(f"\nResults: median_latency={np.median(latencies):.2f}ms, "
          f"success_rate={sum(successes)/len(successes)*100:.1f}%")
    print(f"Log saved to {metrics_dir}/realtime_loop_log.json")


if __name__ == "__main__":
    main()
