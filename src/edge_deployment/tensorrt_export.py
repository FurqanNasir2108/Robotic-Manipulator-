"""TensorRT export utilities (optional, NVIDIA only)."""

from __future__ import annotations

import os
import warnings

try:
    import tensorrt as trt
except ImportError:
    trt = None
    warnings.warn("TensorRT not available; skipping TensorRT export functionality.")


def is_tensorrt_available() -> bool:
    """Check if TensorRT is installed."""
    return trt is not None


def convert_onnx_to_tensorrt(onnx_path: str, engine_path: str,
                              fp16: bool = True, max_batch_size: int = 1) -> str:
    """Convert ONNX model to TensorRT engine.

    Parameters
    ----------
    onnx_path : str
    engine_path : str
    fp16 : bool
    max_batch_size : int

    Returns
    -------
    str
        Path to saved TensorRT engine.

    Raises
    ------
    RuntimeError
        If TensorRT is not available.
    """
    if trt is None:
        raise RuntimeError("TensorRT is not installed.")

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)

    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(parser.get_error(i))
            raise RuntimeError("Failed to parse ONNX model for TensorRT conversion.")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)

    if fp16 and builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)

    os.makedirs(os.path.dirname(engine_path), exist_ok=True)
    engine = builder.build_serialized_network(network, config)
    if engine is None:
        raise RuntimeError("TensorRT engine build failed.")

    with open(engine_path, "wb") as f:
        f.write(engine)

    return engine_path


def validate_tensorrt_engine(engine_path: str, test_input) -> dict:
    """Validate TensorRT engine produces output.

    Parameters
    ----------
    engine_path : str
    test_input : np.ndarray

    Returns
    -------
    dict
        Keys: valid (bool), output_shape.
    """
    if trt is None:
        return {"valid": False, "error": "TensorRT not installed"}

    # TODO: Implement full TensorRT inference validation
    return {"valid": os.path.exists(engine_path), "note": "basic file check only"}
