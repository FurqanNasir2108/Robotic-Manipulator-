"""Test simulated sensor stream."""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.edge_deployment.sensor_interface import SensorPreprocessor, SimulatedSensorStream


def main():
    parser = argparse.ArgumentParser(description="Test sensor stream")
    parser.add_argument("--data-dir", default="data/processed")
    parser.add_argument("--n-samples", type=int, default=10)
    args = parser.parse_args()

    test_data = np.load(os.path.join(args.data_dir, "test.npz"), allow_pickle=True)
    sensor = SimulatedSensorStream(test_data)
    preprocessor = SensorPreprocessor()

    print(f"Sensor stream: {len(sensor)} samples available")
    for i in range(args.n_samples):
        condition = sensor.get_next_condition()
        processed = preprocessor.preprocess(condition)
        print(f"  Sample {i}: shape_type={condition.get('shape_type')}, "
              f"waypoints_shape={condition['waypoints'].shape}, "
              f"processed_shape={processed.shape}")

    print("Sensor stream test passed.")


if __name__ == "__main__":
    main()
