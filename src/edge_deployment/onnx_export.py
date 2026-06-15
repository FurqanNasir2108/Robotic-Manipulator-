"""ONNX-specific export utilities for cVAE and diffusion models."""

from __future__ import annotations

import os

import numpy as np
import torch
import torch.nn as nn


def export_cvae_decoder(cvae_model: nn.Module, example_z: torch.Tensor,
                        example_condition: torch.Tensor, output_path: str,
                        opset_version: int = 17) -> str:
    """Export cVAE decoder to ONNX (z + condition → trajectory).

    The full generate() cannot be exported due to torch.randn.
    Instead, export decode() so z is sampled externally.

    Parameters
    ----------
    cvae_model : nn.Module
        Trained ConditionalVAE model.
    example_z : torch.Tensor
        Example latent vector, shape (1, latent_dim).
    example_condition : torch.Tensor
        Example condition input, shape (1, W, D).
    output_path : str
    opset_version : int

    Returns
    -------
    str
        Path to saved ONNX model.
    """
    cvae_model.eval()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    class CVAEDecoderWrapper(nn.Module):
        def __init__(self, cvae):
            super().__init__()
            self.condition_encoder = cvae.condition_encoder
            self.decoder = cvae.decoder

        def forward(self, z, condition):
            cond_emb = self.condition_encoder(condition)
            return self.decoder(z, cond_emb)

    wrapper = CVAEDecoderWrapper(cvae_model)
    wrapper.eval()

    torch.onnx.export(
        wrapper,
        (example_z, example_condition),
        output_path,
        opset_version=opset_version,
        input_names=["z", "condition"],
        output_names=["trajectory"],
    )
    return output_path


def export_diffusion_denoiser(diffusion_model: nn.Module,
                              example_x_t: torch.Tensor,
                              example_t: torch.Tensor,
                              example_condition: torch.Tensor,
                              output_path: str,
                              opset_version: int = 17) -> str:
    """Export diffusion denoiser to ONNX (single denoise step).

    The iterative DDIM/DDPM loop cannot be exported as one ONNX graph.
    Export the denoiser so the sampling loop runs in Python.

    Parameters
    ----------
    diffusion_model : nn.Module
    example_x_t : torch.Tensor
        Noisy trajectory, shape (1, T, 3).
    example_t : torch.Tensor
        Timestep, shape (1,).
    example_condition : torch.Tensor
        Condition, shape (1, W, D).
    output_path : str

    Returns
    -------
    str
        Path to saved ONNX model.
    """
    diffusion_model.eval()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    class DenoiserWrapper(nn.Module):
        def __init__(self, diffusion):
            super().__init__()
            self.denoiser = diffusion.denoiser
            self.condition_encoder = diffusion.condition_encoder

        def forward(self, x_t, t, condition):
            cond_emb = self.condition_encoder(condition)
            return self.denoiser(x_t, t, cond_emb)

    wrapper = DenoiserWrapper(diffusion_model)
    wrapper.eval()

    torch.onnx.export(
        wrapper,
        (example_x_t, example_t, example_condition),
        output_path,
        opset_version=opset_version,
        input_names=["x_t", "t", "condition"],
        output_names=["noise_pred"],
    )
    return output_path
