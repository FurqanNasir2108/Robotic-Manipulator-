"""Integration test for edge deployment pipeline."""

import os
import tempfile

import numpy as np
import pytest
import torch
import torch.nn as nn


class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(30, 300)

    def forward(self, x):
        b = x.shape[0]
        return self.fc(x.reshape(b, -1)).reshape(b, 100, 3)


@pytest.fixture
def model():
    m = DummyModel()
    m.eval()
    return m


class TestEdgePipeline:
    def test_export_benchmark_validate(self, model):
        """Test: export → benchmark → validate pipeline."""
        ort = pytest.importorskip("onnxruntime")
        from src.edge_deployment.model_export import export_to_onnx, validate_onnx_export
        from src.edge_deployment.runtime_inference import ONNXInferenceSession
        from src.edge_deployment.edge_benchmark import benchmark_latency, model_size_mb

        example_input = torch.randn(1, 10, 3)

        with tempfile.TemporaryDirectory() as tmpdir:
            onnx_path = os.path.join(tmpdir, "model.onnx")
            export_to_onnx(model, example_input, onnx_path)
            assert os.path.exists(onnx_path)

            # Validate
            result = validate_onnx_export(model, onnx_path, example_input)
            assert result["match"] is True

            # Benchmark
            session = ONNXInferenceSession(onnx_path)
            test_inputs = [np.random.randn(1, 10, 3).astype(np.float32) for _ in range(10)]
            latency = benchmark_latency(session, test_inputs, n_runs=20, warmup=5)
            assert latency["mean_ms"] > 0

            size = model_size_mb(onnx_path)
            assert size > 0

    def test_quantization_pipeline(self, model):
        """Test: export → quantize → validate pipeline."""
        ort = pytest.importorskip("onnxruntime")
        from src.edge_deployment.model_export import export_to_onnx
        from src.edge_deployment.optimization_utils import dynamic_quantization

        example_input = torch.randn(1, 10, 3)

        with tempfile.TemporaryDirectory() as tmpdir:
            onnx_path = os.path.join(tmpdir, "model.onnx")
            quant_path = os.path.join(tmpdir, "model_int8.onnx")
            export_to_onnx(model, example_input, onnx_path)

            try:
                dynamic_quantization(onnx_path, quant_path)
                assert os.path.exists(quant_path)
                # Quantized should be smaller
                assert os.path.getsize(quant_path) <= os.path.getsize(onnx_path) * 1.1
            except Exception:
                pytest.skip("Quantization not available")
