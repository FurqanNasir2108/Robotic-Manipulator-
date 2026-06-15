"""Unit tests for edge deployment module."""

import os
import tempfile

import numpy as np
import pytest
import torch
import torch.nn as nn


class SimpleModel(nn.Module):
    """Minimal model for testing export."""
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(30, 300)

    def forward(self, x):
        b = x.shape[0]
        flat = x.reshape(b, -1)
        out = self.fc(flat)
        return out.reshape(b, 100, 3)


@pytest.fixture
def simple_model():
    model = SimpleModel()
    model.eval()
    return model


@pytest.fixture
def example_input():
    return torch.randn(1, 10, 3)


class TestModelExport:
    def test_export_to_torchscript(self, simple_model, example_input):
        from src.edge_deployment.model_export import export_to_torchscript
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "model.pt")
            result = export_to_torchscript(simple_model, example_input, path)
            assert os.path.exists(result)

    def test_export_to_onnx(self, simple_model, example_input):
        from src.edge_deployment.model_export import export_to_onnx
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "model.onnx")
            result = export_to_onnx(simple_model, example_input, path)
            assert os.path.exists(result)

    def test_validate_onnx_export(self, simple_model, example_input):
        from src.edge_deployment.model_export import export_to_onnx, validate_onnx_export
        ort = pytest.importorskip("onnxruntime")
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "model.onnx")
            export_to_onnx(simple_model, example_input, path)
            result = validate_onnx_export(simple_model, path, example_input)
            assert result["match"] is True
            assert result["max_deviation"] < 1e-5


class TestEdgeBenchmark:
    def test_model_size(self, simple_model, example_input):
        from src.edge_deployment.edge_benchmark import model_size_mb
        from src.edge_deployment.model_export import export_to_onnx
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "model.onnx")
            export_to_onnx(simple_model, example_input, path)
            size = model_size_mb(path)
            assert size > 0

    def test_pareto_analysis(self):
        from src.edge_deployment.edge_benchmark import pareto_analysis
        bench = {
            "model_a": {"mean_ms": 1.0, "size_mb": 0.5},
            "model_b": {"mean_ms": 10.0, "size_mb": 5.0},
        }
        accuracy = {
            "model_a": {"joint_rmse": 0.1},
            "model_b": {"joint_rmse": 0.01},
        }
        result = pareto_analysis(bench, accuracy)
        assert "pareto_models" in result
        assert len(result["pareto_models"]) >= 1


class TestONNXInference:
    def test_onnx_session_predict(self, simple_model, example_input):
        ort = pytest.importorskip("onnxruntime")
        from src.edge_deployment.model_export import export_to_onnx
        from src.edge_deployment.runtime_inference import ONNXInferenceSession
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "model.onnx")
            export_to_onnx(simple_model, example_input, path)
            session = ONNXInferenceSession(path)
            output = session.predict(example_input.numpy())
            assert output.shape == (1, 100, 3)


class TestDeviceMonitor:
    def test_monitor_basic(self):
        from src.edge_deployment.device_monitor import DeviceMonitor
        monitor = DeviceMonitor()
        monitor.start()
        monitor.sample()
        monitor.sample()
        monitor.stop()
        summary = monitor.get_summary()
        assert summary["n_samples"] == 2


class TestSensorInterface:
    def test_simulated_stream(self):
        from src.edge_deployment.sensor_interface import SimulatedSensorStream, SensorPreprocessor
        test_data = {"waypoints": np.random.randn(10, 5, 3)}
        sensor = SimulatedSensorStream(test_data)
        assert len(sensor) == 10
        condition = sensor.get_next_condition()
        assert "waypoints" in condition

        preprocessor = SensorPreprocessor()
        processed = preprocessor.preprocess(condition)
        assert processed.shape == (5, 3)

    def test_stream_reset(self):
        from src.edge_deployment.sensor_interface import SimulatedSensorStream
        test_data = {"waypoints": np.random.randn(3, 5, 3)}
        sensor = SimulatedSensorStream(test_data)
        for _ in range(5):
            sensor.get_next_condition()
        sensor.reset()
        condition = sensor.get_next_condition()
        assert condition["index"] == 0


class TestControllerInterface:
    def test_execute_trajectory(self):
        from src.edge_deployment.controller_interface import SimulatorController
        controller = SimulatorController()
        traj = np.zeros((10, 3))  # all zeros = within limits
        result = controller.execute_trajectory(traj)
        assert result["success"] is True
        assert result["n_steps"] == 10

    def test_validate_execution(self):
        from src.edge_deployment.controller_interface import SimulatorController
        controller = SimulatorController()
        planned = np.zeros((10, 3))
        executed = np.zeros((10, 3)) + 0.01
        result = controller.validate_execution(planned, executed)
        assert result["joint_rmse"] > 0


class TestTensorRTExport:
    def test_is_available(self):
        from src.edge_deployment.tensorrt_export import is_tensorrt_available
        # Just verify the function runs
        result = is_tensorrt_available()
        assert isinstance(result, bool)
