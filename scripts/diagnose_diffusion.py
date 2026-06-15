"""Diagnostic script to identify why the diffusion model produces bad trajectories."""

import os
import sys
import torch
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.utils.config import load_config
from src.data.normalization import Normalizer
from src.data.loaders import TrajectoryDataset
from src.models.diffusion import build_diffusion_model

device = 'cpu'


def load_model_and_data():
    cfg = load_config('configs/diffusion.yaml')
    model, ema = build_diffusion_model(cfg)

    ckpt = torch.load(
        'results/checkpoints/diffusion/diffusion_best.pth',
        map_location=device, weights_only=True,
    )

    # Load EMA model (used in evaluation)
    ema.load_state_dict(ckpt['ema_state_dict'])
    ema_model = ema.shadow.to(device)
    ema_model.eval()

    # Load non-EMA model
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(device)
    model.eval()

    # Load data
    normalizer = Normalizer.load('data/metadata/normalization_stats.json')
    dataset = TrajectoryDataset('data/processed/test.npz', normalizer=normalizer, input_mode='waypoints')

    return model, ema_model, cfg, normalizer, dataset


def test_single_step_denoising(model, dataset, label=""):
    """Check if the model can predict noise for a single step."""
    print(f"\n{'='*60}")
    print(f"TEST: Single-step noise prediction ({label})")
    print(f"{'='*60}")

    condition, target = dataset[0]
    condition = condition.unsqueeze(0).to(device, dtype=torch.float32)
    target = target.unsqueeze(0).to(device, dtype=torch.float32)

    x_0 = target.permute(0, 2, 1)  # (1, 3, T)
    print(f"x_0 shape: {x_0.shape}, range: [{x_0.min():.4f}, {x_0.max():.4f}]")

    # Encode condition
    cond = model.condition_encoder(condition)
    print(f"cond shape: {cond.shape}, range: [{cond.min():.4f}, {cond.max():.4f}]")
    print(f"cond norm: {cond.norm():.4f}")

    for t_val in [0, 10, 100, 500, 999]:
        t = torch.tensor([t_val], device=device)
        noise = torch.randn_like(x_0)
        x_t, true_noise = model.q_sample(x_0, t, noise)

        pred_noise = model.denoiser(x_t, t, cond)

        mse = ((pred_noise - true_noise) ** 2).mean().item()
        print(f"  t={t_val:4d} | x_t range [{x_t.min():.3f}, {x_t.max():.3f}] | "
              f"noise_pred range [{pred_noise.min():.3f}, {pred_noise.max():.3f}] | MSE={mse:.6f}")


def test_alpha_bar_values(model):
    """Check the noise schedule values."""
    print(f"\n{'='*60}")
    print("TEST: Alpha-bar schedule values")
    print(f"{'='*60}")
    ab = model.alpha_bar
    print(f"alpha_bar[0]   = {ab[0]:.6f}  (should be ~1.0)")
    print(f"alpha_bar[1]   = {ab[1]:.6f}")
    print(f"alpha_bar[10]  = {ab[10]:.6f}")
    print(f"alpha_bar[100] = {ab[100]:.6f}")
    print(f"alpha_bar[500] = {ab[500]:.6f}")
    print(f"alpha_bar[999] = {ab[999]:.6f}  (should be ~0.0)")


def test_generation_step_by_step(model, dataset, cfg):
    """Trace DDIM sampling and look for divergence."""
    print(f"\n{'='*60}")
    print("TEST: DDIM sampling trace (5 steps)")
    print(f"{'='*60}")

    condition, target = dataset[0]
    waypoints = condition.unsqueeze(0).to(device, dtype=torch.float32)
    seq_len = cfg['sampling'].get('seq_len', 100)

    cond = model.condition_encoder(waypoints)
    print(f"Condition embedding: shape={cond.shape}, norm={cond.norm():.4f}")

    x = torch.randn(1, model.joint_dim, seq_len, device=device)
    print(f"Initial noise: mean={x.mean():.4f}, std={x.std():.4f}, range=[{x.min():.3f}, {x.max():.3f}]")

    num_steps = 5  # Just trace 5 steps
    ts = torch.linspace(model.num_timesteps - 1, 0, steps=num_steps, device=device)
    timesteps = [int(v) for v in ts.long().tolist()]

    for idx, i in enumerate(timesteps):
        t = torch.full((1,), i, device=device, dtype=torch.long)
        noise_pred = model.denoiser(x, t, cond)

        # CFG
        noise_uncond = model.denoiser(x, t, torch.zeros_like(cond))
        guidance_scale = 2.0
        noise_guided = noise_uncond + guidance_scale * (noise_pred - noise_uncond)

        alpha_bar_t = model.alpha_bar[i]
        if idx + 1 < len(timesteps):
            alpha_bar_prev = model.alpha_bar[timesteps[idx + 1]]
        else:
            alpha_bar_prev = torch.tensor(1.0, device=device)

        pred_x0 = (x - torch.sqrt(1 - alpha_bar_t) * noise_guided) / torch.sqrt(alpha_bar_t)
        pred_x0_clamped = torch.clamp(pred_x0, -5.0, 5.0)

        dir_xt = torch.sqrt(torch.clamp(1 - alpha_bar_prev, min=0.0)) * noise_guided
        x_new = torch.sqrt(alpha_bar_prev) * pred_x0_clamped + dir_xt

        print(f"  Step {idx} (t={i:4d}) | alpha_bar={alpha_bar_t:.6f} | "
              f"noise_pred: [{noise_pred.min():.3f}, {noise_pred.max():.3f}] | "
              f"noise_uncond: [{noise_uncond.min():.3f}, {noise_uncond.max():.3f}] | "
              f"pred_x0: [{pred_x0.min():.3f}, {pred_x0.max():.3f}] | "
              f"x_new: [{x_new.min():.3f}, {x_new.max():.3f}]")
        x = x_new

    # Compare with ground truth
    gt = target.numpy()  # (T, 3)
    gen = x.permute(0, 2, 1).squeeze(0).detach().numpy()  # (T, 3)
    print(f"\n  Ground truth range: [{gt.min():.3f}, {gt.max():.3f}]")
    print(f"  Generated range:    [{gen.min():.3f}, {gen.max():.3f}]")
    print(f"  RMSE (normalized):  {np.sqrt(np.mean((gen - gt) ** 2)):.4f}")


def test_full_generation(model, dataset, normalizer, cfg):
    """Generate with full DDIM steps and compare."""
    print(f"\n{'='*60}")
    print("TEST: Full DDIM generation (75 steps)")
    print(f"{'='*60}")

    condition, target = dataset[0]
    waypoints = condition.unsqueeze(0).to(device, dtype=torch.float32)

    for gs in [0.0, 1.0, 2.0]:
        with torch.no_grad():
            gen = model.generate(
                waypoints, num_samples=1, method='ddim',
                num_steps=cfg['sampling'].get('ddim_steps', 75),
                guidance_scale=gs,
                seq_len=cfg['sampling'].get('seq_len', 100),
            )
        gen_np = gen.squeeze(0).numpy()
        gt_np = target.numpy()
        rmse_norm = np.sqrt(np.mean((gen_np - gt_np) ** 2))

        gt_denorm = normalizer.inverse_transform(gt_np, 'q_sequence')
        gen_denorm = normalizer.inverse_transform(gen_np, 'q_sequence')
        rmse_denorm = np.sqrt(np.mean((gen_denorm - gt_denorm) ** 2))

        print(f"  guidance_scale={gs:.1f} | "
              f"gen range [{gen_np.min():.3f}, {gen_np.max():.3f}] | "
              f"RMSE(norm)={rmse_norm:.4f} | RMSE(denorm)={rmse_denorm:.4f}")


def compare_ema_vs_original(model, ema_model, dataset):
    """Check if EMA and original model produce different results."""
    print(f"\n{'='*60}")
    print("TEST: EMA vs original model weight comparison")
    print(f"{'='*60}")

    # Compare parameter norms
    orig_params = dict(model.named_parameters())
    ema_params = dict(ema_model.named_parameters())

    total_diff = 0.0
    total_norm = 0.0
    for name in orig_params:
        if name in ema_params:
            diff = (orig_params[name] - ema_params[name]).abs().mean().item()
            norm = orig_params[name].abs().mean().item()
            total_diff += diff
            total_norm += norm
    print(f"  Avg absolute parameter diff (EMA vs original): {total_diff / len(orig_params):.6f}")
    print(f"  Avg absolute parameter norm: {total_norm / len(orig_params):.6f}")

    # Check buffer consistency
    orig_buffers = dict(model.named_buffers())
    ema_buffers = dict(ema_model.named_buffers())
    print(f"  Original model buffers: {len(orig_buffers)}")
    print(f"  EMA model buffers: {len(ema_buffers)}")
    for name in orig_buffers:
        if name in ema_buffers:
            match = torch.allclose(orig_buffers[name], ema_buffers[name])
            if not match:
                print(f"  MISMATCH buffer: {name}")


def test_data_shapes(dataset):
    """Check the actual data dimensions."""
    print(f"\n{'='*60}")
    print("TEST: Data shapes and ranges")
    print(f"{'='*60}")
    condition, target = dataset[0]
    print(f"  Condition (waypoints) shape: {condition.shape}")
    print(f"  Target (q_sequence) shape: {target.shape}")
    print(f"  Condition range: [{condition.min():.4f}, {condition.max():.4f}]")
    print(f"  Target range: [{target.min():.4f}, {target.max():.4f}]")
    print(f"  Dataset size: {len(dataset)}")


if __name__ == '__main__':
    model, ema_model, cfg, normalizer, dataset = load_model_and_data()

    test_data_shapes(dataset)
    test_alpha_bar_values(ema_model)
    compare_ema_vs_original(model, ema_model, dataset)
    test_single_step_denoising(ema_model, dataset, label="EMA model")
    test_single_step_denoising(model, dataset, label="Original model")
    test_generation_step_by_step(ema_model, dataset, cfg)
    test_full_generation(ema_model, dataset, normalizer, cfg)
    test_full_generation(model, dataset, normalizer, cfg)

    print("\n" + "="*60)
    print("DIAGNOSIS COMPLETE")
    print("="*60)
