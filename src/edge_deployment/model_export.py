"""Generic model export dispatcher: PyTorch → TorchScript / ONNX."""

from __future__ import annotations

import os
from pathlib import Path

import torch
import torch.nn as nn


def export_to_torchscript(model: nn.Module, example_input: torch.Tensor,
                          output_path: str) -> str:
    """Export a PyTorch model to TorchScript via tracing.

    Parameters
    ----------
    model : nn.Module
    example_input : torch.Tensor
    output_path : str

    Returns
    -------
    str
        Path to saved TorchScript model.
    """
    model.eval()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    traced = torch.jit.trace(model, example_input)
    traced.save(output_path)
    return output_path


def export_to_onnx(model: nn.Module, example_input: torch.Tensor,
                   output_path: str, opset_version: int = 17,
                   input_names: list[str] | None = None,
                   output_names: list[str] | None = None,
                   dynamic_axes: dict | None = None) -> str:
    """Export a PyTorch model to ONNX format.

    Parameters
    ----------
    model : nn.Module
    example_input : torch.Tensor
    output_path : str
    opset_version : int
    input_names : list of str or None
    output_names : list of str or None
    dynamic_axes : dict or None

    Returns
    -------
    str
        Path to saved ONNX model.
    """
    model.eval()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    if input_names is None:
        input_names = ["input"]
    if output_names is None:
        output_names = ["output"]

    torch.onnx.export(
        model,
        example_input,
        output_path,
        opset_version=opset_version,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
    )
    return output_path


def validate_onnx_export(pytorch_model: nn.Module, onnx_path: str,
                         test_input: torch.Tensor, rtol: float = 1e-5) -> dict:
    """Validate ONNX model output matches PyTorch output.

    Parameters
    ----------
    pytorch_model : nn.Module
    onnx_path : str
    test_input : torch.Tensor
    rtol : float

    Returns
    -------
    dict
        Keys: match (bool), max_deviation (float), mean_deviation (float).
    """
    import numpy as np

    try:
        import onnxruntime as ort
    except ImportError:
        return {"match": False, "error": "onnxruntime not installed"}

    pytorch_model.eval()
    with torch.no_grad():
        pt_output = pytorch_model(test_input).cpu().numpy()

    session = ort.InferenceSession(onnx_path)
    input_name = session.get_inputs()[0].name
    onnx_output = session.run(None, {input_name: test_input.cpu().numpy()})[0]

    max_dev = float(np.max(np.abs(pt_output - onnx_output)))
    mean_dev = float(np.mean(np.abs(pt_output - onnx_output)))

    return {
        "match": max_dev < rtol,
        "max_deviation": max_dev,
        "mean_deviation": mean_dev,
    }
