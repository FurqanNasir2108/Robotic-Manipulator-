"""Model optimization utilities: quantization, graph optimization."""

from __future__ import annotations

import os
import warnings


def quantize_dynamic(onnx_path: str, output_path: str,
                     weight_type: str = "int8") -> str:
    """Apply dynamic quantization to an ONNX model.

    Parameters
    ----------
    onnx_path : str
    output_path : str
    weight_type : str
        'int8' or 'uint8'.

    Returns
    -------
    str
        Path to quantized model.
    """
    try:
        from onnxruntime.quantization import quantize_dynamic, QuantType
    except ImportError:
        raise ImportError("onnxruntime.quantization is required. "
                          "Install via: pip install onnxruntime")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    qtype = QuantType.QInt8 if weight_type == "int8" else QuantType.QUInt8
    quantize_dynamic(onnx_path, output_path, weight_type=qtype)
    return output_path


def dynamic_quantization(onnx_path: str, output_path: str,
                         weight_type: str = "int8") -> str:
    """Backward-compatible alias for dynamic ONNX quantization."""
    return quantize_dynamic(onnx_path, output_path, weight_type=weight_type)


def quantize_static(onnx_path: str, output_path: str,
                    calibration_data: list) -> str:
    """Apply static quantization with calibration data.

    Parameters
    ----------
    onnx_path : str
    output_path : str
    calibration_data : list of np.ndarray

    Returns
    -------
    str
        Path to quantized model.
    """
    # TODO: Implement static quantization with calibration data reader
    raise NotImplementedError("Static quantization requires a calibration data reader. "
                              "Use quantize_dynamic as fallback.")


def optimize_onnx_graph(onnx_path: str, output_path: str,
                        optimization_level: str = "all") -> str:
    """Optimize ONNX graph using ONNX Runtime.

    Parameters
    ----------
    onnx_path : str
    output_path : str
    optimization_level : str
        'basic', 'extended', or 'all'.

    Returns
    -------
    str
        Path to optimized model.
    """
    try:
        import onnxruntime as ort
    except ImportError:
        raise ImportError("onnxruntime is required for graph optimization.")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    opt_level_map = {
        "basic": ort.GraphOptimizationLevel.ORT_ENABLE_BASIC,
        "extended": ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED,
        "all": ort.GraphOptimizationLevel.ORT_ENABLE_ALL,
    }

    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = opt_level_map.get(
        optimization_level, ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    )
    sess_options.optimized_model_filepath = output_path
    ort.InferenceSession(onnx_path, sess_options)
    return output_path


def convert_to_fp16(onnx_path: str, output_path: str) -> str:
    """Convert ONNX model weights from FP32 to FP16.

    Parameters
    ----------
    onnx_path : str
    output_path : str

    Returns
    -------
    str
        Path to FP16 model.
    """
    try:
        import onnx
        from onnx import numpy_helper
    except ImportError:
        raise ImportError("onnx is required for FP16 conversion.")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    model = onnx.load(onnx_path)

    # Convert float tensors to float16
    import numpy as np
    for tensor in model.graph.initializer:
        if tensor.data_type == onnx.TensorProto.FLOAT:
            arr = numpy_helper.to_array(tensor).astype(np.float16)
            new_tensor = numpy_helper.from_array(arr, name=tensor.name)
            tensor.CopyFrom(new_tensor)

    onnx.save(model, output_path)
    return output_path
