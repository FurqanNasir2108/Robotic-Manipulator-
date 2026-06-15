"""Integration test for real-time inference loop."""

import os
import tempfile

import numpy as np
import pytest
import torch
import torch.nn as nn


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(15, 300)

    def forward(self, x):
        b = x.shape[0]
        return self.fc(x.reshape(b, -1)).reshape(b, 100, 3)


@pytest.fixture
def onnx_model():
    ort = pytest.importorskip("onnxruntime")
    from src.edge_deployment.model_export import export_to_onnx

    model = TinyModel()
    model.eval()
    tmpdir = tempfile.mkdtemp()
    path = os.path.join(tmpdir, "tiny.onnx")
    export_to_onnx(model, torch.randn(1, 5, 3), path)
    yield path
    import shutil
    shutil.rmtree(tmpdir, ignore_errors=True)


class TestRealtimeLoop:
    def test_sensor_to_inference_loop(self, onnx_model):
        """Test: sensor → preprocess → inference → controller."""
        from src.edge_deployment.runtime_inference import ONNXInferenceSession
        from src.edge_deployment.sensor_interface import SimulatedSensorStream, SensorPreprocessor
        from src.edge_deployment.controller_interface import SimulatorController

        session = ONNXInferenceSession(onnx_model)
        sensor_data = {"waypoints": np.random.randn(5, 5, 3).astype(np.float32)}
        sensor = SimulatedSensorStream(sensor_data)
        preprocessor = SensorPreprocessor()
        controller = SimulatorController()

        results = []
        for _ in range(3):
            condition = sensor.get_next_condition()
            processed = preprocessor.preprocess(condition)
            trajectory = session.predict(processed[np.newaxis])
            exec_result = controller.execute_trajectory(trajectory[0])
            results.append(exec_result)

        assert len(results) == 3
        assert all(r["success"] for r in results)

    def test_monitoring_during_loop(self, onnx_model):
        """Test device monitoring during inference."""
        from src.edge_deployment.runtime_inference import ONNXInferenceSession
        from src.edge_deployment.device_monitor import DeviceMonitor

        session = ONNXInferenceSession(onnx_model)
        monitor = DeviceMonitor()
        monitor.start()

        for _ in range(5):
            inp = np.random.randn(1, 5, 3).astype(np.float32)
            session.predict(inp)
            monitor.sample()

        monitor.stop()
        summary = monitor.get_summary()
        assert summary["n_samples"] == 5
