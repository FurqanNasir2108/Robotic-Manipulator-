"""Export trained models to ONNX format."""

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.edge_deployment.config import load_edge_config
from src.edge_deployment.model_export import export_to_onnx, validate_onnx_export
from src.edge_deployment.onnx_export import export_cvae_decoder, export_diffusion_denoiser
from src.utils.config import load_config


def main():
    parser = argparse.ArgumentParser(description="Export models to ONNX")
    parser.add_argument("--config", default="configs/edge_deployment.yaml")
    parser.add_argument("--model", default=None, help="Specific model to export (default: all)")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    cfg = load_edge_config(args.config)
    output_dir = cfg["paths"]["checkpoints_dir"]
    os.makedirs(output_dir, exist_ok=True)

    models_to_export = cfg["models"]["include"] if args.model is None else [args.model]

    for model_name in models_to_export:
        print(f"\nExporting {model_name}...")
        try:
            if model_name in ("cnn", "cnn_lstm", "cnn_gru"):
                _export_baseline(model_name, output_dir, cfg, args.device)
            elif model_name == "cvae":
                _export_cvae(output_dir, cfg, args.device)
            elif model_name == "diffusion_ddim":
                _export_diffusion(output_dir, cfg, args.device)
            else:
                print(f"  Skipping {model_name}: no export path defined")
        except Exception as e:
            print(f"  Failed to export {model_name}: {e}")

    print(f"\nExported models saved to {output_dir}")


def _export_baseline(model_name, output_dir, cfg, device):
    from src.evaluate.comparisons import _build_baseline_model
    config_map = {
        "cnn": "configs/baseline_cnn.yaml",
        "cnn_lstm": "configs/baseline_cnn_lstm.yaml",
        "cnn_gru": "configs/baseline_cnn_gru.yaml",
    }
    model_cfg = load_config(config_map[model_name])
    model = _build_baseline_model(model_cfg).to(device)
    ckpt = torch.load(
        os.path.join("results", "checkpoints", model_name, f"{model_name}_best.pth"),
        map_location=device, weights_only=True,
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    input_shape = cfg["export"]["input_shape"]
    example_input = torch.randn(*input_shape, device=device)
    onnx_path = os.path.join(output_dir, f"{model_name}_fp32.onnx")
    export_to_onnx(model, example_input, onnx_path,
                   opset_version=cfg["export"]["opset_version"])

    result = validate_onnx_export(model, onnx_path, example_input)
    print(f"  {model_name}: max_deviation={result['max_deviation']:.8f}, match={result['match']}")


def _export_cvae(output_dir, cfg, device):
    from src.evaluate.comparisons import _build_cvae_model
    model_cfg = load_config("configs/vae.yaml")
    model = _build_cvae_model(model_cfg).to(device)
    ckpt = torch.load(
        os.path.join("results", "checkpoints", "cvae", "cvae_best.pth"),
        map_location=device, weights_only=True,
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    latent_dim = model_cfg["model"]["latent_dim"]
    example_z = torch.randn(1, latent_dim, device=device)
    input_shape = cfg["export"]["input_shape"]
    example_condition = torch.randn(*input_shape, device=device)

    onnx_path = os.path.join(output_dir, "cvae_fp32.onnx")
    export_cvae_decoder(model, example_z, example_condition, onnx_path,
                        opset_version=cfg["export"]["opset_version"])
    print(f"  cvae decoder exported to {onnx_path}")


def _export_diffusion(output_dir, cfg, device):
    from src.models.diffusion import build_diffusion_model
    model_cfg = load_config("configs/diffusion.yaml")
    model, _ = build_diffusion_model(model_cfg)
    ckpt = torch.load(
        os.path.join("results", "checkpoints", "diffusion", "diffusion_best.pth"),
        map_location=device, weights_only=True,
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()

    output_shape = cfg["export"]["output_shape"]
    example_x_t = torch.randn(*output_shape, device=device)
    example_t = torch.tensor([500], device=device)
    input_shape = cfg["export"]["input_shape"]
    example_condition = torch.randn(*input_shape, device=device)

    onnx_path = os.path.join(output_dir, "diffusion_denoiser_fp32.onnx")
    export_diffusion_denoiser(model, example_x_t, example_t, example_condition,
                              onnx_path, opset_version=cfg["export"]["opset_version"])
    print(f"  diffusion denoiser exported to {onnx_path}")


if __name__ == "__main__":
    main()
