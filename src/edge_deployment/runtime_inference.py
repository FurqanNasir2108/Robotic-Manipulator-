"""ONNX Runtime and TensorRT inference wrappers."""

from __future__ import annotations

import warnings

import numpy as np


class ONNXInferenceSession:
    """Unified inference wrapper around ONNX Runtime.

    Parameters
    ----------
    onnx_path : str
        Path to ONNX model file.
    device : str
        'cpu' or 'cuda'.
    """

    def __init__(self, onnx_path: str, device: str = "cpu"):
        try:
            import onnxruntime as ort
        except ImportError:
            raise ImportError("onnxruntime is required for ONNX inference. "
                              "Install via: pip install onnxruntime")

        providers = ["CPUExecutionProvider"]
        if device == "cuda":
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

        self.session = ort.InferenceSession(onnx_path, providers=providers)
        self.input_names = [inp.name for inp in self.session.get_inputs()]
        self.output_names = [out.name for out in self.session.get_outputs()]
        self.onnx_path = onnx_path

    def predict(self, condition: np.ndarray) -> np.ndarray:
        """Run inference with a single input.

        Parameters
        ----------
        condition : np.ndarray

        Returns
        -------
        np.ndarray
        """
        feed = {self.input_names[0]: condition.astype(np.float32)}
        return self.session.run(self.output_names, feed)[0]

    def predict_cvae(self, z: np.ndarray, condition: np.ndarray) -> np.ndarray:
        """Run cVAE decoder inference with z and condition inputs.

        Parameters
        ----------
        z : np.ndarray
            Latent vector.
        condition : np.ndarray

        Returns
        -------
        np.ndarray
        """
        feed = {
            self.input_names[0]: z.astype(np.float32),
            self.input_names[1]: condition.astype(np.float32),
        }
        return self.session.run(self.output_names, feed)[0]

    def predict_diffusion_step(self, x_t: np.ndarray, t: np.ndarray,
                                condition: np.ndarray) -> np.ndarray:
        """Run single diffusion denoiser step.

        Parameters
        ----------
        x_t : np.ndarray
            Noisy trajectory.
        t : np.ndarray
            Timestep.
        condition : np.ndarray

        Returns
        -------
        np.ndarray
            Predicted noise.
        """
        feed = {
            self.input_names[0]: x_t.astype(np.float32),
            self.input_names[1]: t.astype(np.float32),
            self.input_names[2]: condition.astype(np.float32),
        }
        return self.session.run(self.output_names, feed)[0]


class TensorRTInferenceSession:
    """TensorRT inference wrapper (optional, NVIDIA only).

    Parameters
    ----------
    engine_path : str
    device : str
    """

    def __init__(self, engine_path: str, device: str = "cuda"):
        try:
            import tensorrt as trt
            import pycuda.driver as cuda
            import pycuda.autoinit
        except ImportError:
            raise ImportError("TensorRT and PyCUDA are required for TensorRT inference.")

        self.engine_path = engine_path
        warnings.warn("TensorRT inference session is a placeholder. "
                       "Full implementation requires TensorRT runtime setup.")

    def predict(self, condition: np.ndarray) -> np.ndarray:
        """Run TensorRT inference.

        Parameters
        ----------
        condition : np.ndarray

        Returns
        -------
        np.ndarray
        """
        raise NotImplementedError("TensorRT inference not yet implemented. "
                                  "Use ONNXInferenceSession as fallback.")
