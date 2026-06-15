"""
Train Conditional VAE for trajectory generation.

Usage:
    python scripts/train_vae.py
    python scripts/train_vae.py --config configs/vae.yaml
"""

import os
import sys
import argparse
import logging
import json
import time

import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.utils.config import load_config
from src.utils.reproducibility import set_seed
from src.data.normalization import Normalizer
from src.data.loaders import get_dataloaders
from src.models.cvae import ConditionalVAE
from src.train.losses import VAELoss
from src.train.schedulers import get_scheduler

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def build_cvae(cfg):
    """Build ConditionalVAE from config."""
    m = cfg['model']
    return ConditionalVAE(
        latent_dim=m['latent_dim'],
        encoder_type=m['encoder_type'],
        decoder_type=m['decoder_type'],
        condition_dim=m['condition_dim'],
        input_dim=m['input_dim'],
        output_dim=m['output_dim'],
        input_steps=m['input_steps'],
        output_steps=m['output_steps'],
        cond_conv_channels=m.get('cond_conv_channels'),
        cond_kernel_sizes=m.get('cond_kernel_sizes'),
        enc_hidden=m.get('enc_hidden', 128),
        enc_layers=m.get('enc_layers', 2),
        enc_dropout=m.get('enc_dropout', 0.1),
        dec_hidden=m.get('dec_hidden', 128),
        dec_layers=m.get('dec_layers', 2),
        dec_dropout=m.get('dec_dropout', 0.1),
        dec_fc_dims=m.get('dec_fc_dims'),
        dec_conv_channels=m.get('dec_conv_channels'),
        dec_kernel_sizes=m.get('dec_kernel_sizes'),
    )


def build_vae_loss(cfg):
    """Build VAELoss from config."""
    lc = cfg['losses']
    return VAELoss(
        recon_weight=lc.get('recon_weight', 1.0),
        kl_weight=lc.get('kl_weight', 0.01),
        energy_weight=lc.get('energy_weight', 0.001),
        smoothness_weight=lc.get('smoothness_weight', 0.001),
        kl_annealing=lc.get('kl_annealing', 'cyclical'),
        kl_warmup_epochs=lc.get('kl_warmup_epochs', 50),
        kl_cycle_epochs=lc.get('kl_cycle_epochs', 20),
        free_bits_lambda=lc.get('free_bits_lambda', 0.1),
    )


def train_epoch(model, train_loader, loss_fn, optimizer, scaler, device, use_amp, grad_clip):
    """Run one training epoch for the cVAE."""
    model.train()
    total_losses = {}
    n_batches = 0

    for condition, target in train_loader:
        condition = condition.to(device, dtype=torch.float32)
        target = target.to(device, dtype=torch.float32)

        optimizer.zero_grad()
        with autocast('cuda', enabled=use_amp):
            recon, mu, log_var = model(condition, target)
            loss, loss_dict = loss_fn(recon, target, mu, log_var)

        scaler.scale(loss).backward()
        if grad_clip:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()

        for k, v in loss_dict.items():
            total_losses[k] = total_losses.get(k, 0.0) + v
        n_batches += 1

    return {k: v / max(n_batches, 1) for k, v in total_losses.items()}


@torch.no_grad()
def validate_epoch(model, val_loader, loss_fn, device, use_amp):
    """Run validation for the cVAE."""
    model.eval()
    total_losses = {}
    n_batches = 0

    for condition, target in val_loader:
        condition = condition.to(device, dtype=torch.float32)
        target = target.to(device, dtype=torch.float32)

        with autocast('cuda', enabled=use_amp):
            recon, mu, log_var = model(condition, target)
            loss, loss_dict = loss_fn(recon, target, mu, log_var)

        for k, v in loss_dict.items():
            total_losses[k] = total_losses.get(k, 0.0) + v
        n_batches += 1

    return {k: v / max(n_batches, 1) for k, v in total_losses.items()}


def save_checkpoint(model, optimizer, epoch, best_val_loss, path):
    """Save model checkpoint."""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_val_loss': best_val_loss,
    }, path)


def main():
    parser = argparse.ArgumentParser(description='Train Conditional VAE')
    parser.add_argument('--config', type=str, default='configs/vae.yaml')
    args = parser.parse_args()

    cfg = load_config(args.config)
    tcfg = cfg['training']
    set_seed(tcfg['seed'])

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    use_amp = tcfg.get('use_amp', False) and device != 'cpu'
    logger.info(f"Device: {device}, AMP: {use_amp}")

    # Load data
    norm_path = os.path.join(cfg['data']['metadata_dir'], 'normalization_stats.json')
    normalizer = Normalizer.load(norm_path)
    loaders = get_dataloaders(
        data_dir=cfg['data']['dir'],
        normalizer=normalizer,
        batch_size=tcfg['batch_size'],
        input_mode=cfg['data']['input_mode'],
    )

    # Build model
    model = build_cvae(cfg)
    model = model.to(device)
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"cVAE parameters: {param_count:,}")

    # Loss, optimizer, scheduler
    loss_fn = build_vae_loss(cfg)
    optimizer = torch.optim.Adam(model.parameters(), lr=tcfg['lr'])
    scheduler = get_scheduler(optimizer, tcfg.get('scheduler', 'cosine'),
                              total_epochs=tcfg['epochs'])
    scaler = GradScaler('cuda', enabled=use_amp)
    grad_clip = tcfg.get('grad_clip', None)

    # Directories
    ckpt_dir = 'results/checkpoints/cvae'
    log_dir = 'results/logs/cvae'
    metrics_dir = 'results/metrics/cvae'
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)

    # TensorBoard
    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter(log_dir=log_dir)

    # Training loop
    best_val_recon = float('inf')
    patience_counter = 0
    patience = tcfg.get('early_stop_patience', 60)
    epochs = tcfg['epochs']

    history = {
        'train_total': [], 'train_recon': [], 'train_kl': [],
        'train_energy': [], 'train_smoothness': [],
        'val_total': [], 'val_recon': [], 'val_kl': [],
        'val_energy': [], 'val_smoothness': [], 'kl_weight': [],
    }

    logger.info(f"Training cVAE for {epochs} epochs")
    logger.info(f"KL annealing: {cfg['losses']['kl_annealing']}, "
                f"warmup: {cfg['losses']['kl_warmup_epochs']}, "
                f"cycle: {cfg['losses']['kl_cycle_epochs']}")

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        loss_fn.set_epoch(epoch)

        train_metrics = train_epoch(model, loaders['train'], loss_fn, optimizer,
                                    scaler, device, use_amp, grad_clip)
        val_metrics = validate_epoch(model, loaders['val'], loss_fn, device, use_amp)
        elapsed = time.time() - t0

        # Record history
        for key in ['total', 'recon', 'kl', 'energy', 'smoothness']:
            history[f'train_{key}'].append(train_metrics[key])
            history[f'val_{key}'].append(val_metrics[key])
        history['kl_weight'].append(train_metrics.get('kl_weight', 0))

        # TensorBoard
        for key in ['total', 'recon', 'kl', 'energy', 'smoothness']:
            writer.add_scalar(f'cvae/train_{key}', train_metrics[key], epoch)
            writer.add_scalar(f'cvae/val_{key}', val_metrics[key], epoch)
        writer.add_scalar('cvae/kl_weight', train_metrics.get('kl_weight', 0), epoch)

        if scheduler:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_metrics['total'])
            else:
                scheduler.step()
            writer.add_scalar('cvae/lr', optimizer.param_groups[0]['lr'], epoch)

        # Logging
        if epoch % 10 == 0 or epoch == 1:
            logger.info(
                f"  Epoch {epoch:4d}/{epochs} | "
                f"train_total={train_metrics['total']:.6f} recon={train_metrics['recon']:.6f} "
                f"kl={train_metrics['kl']:.4f} | "
                f"val_recon={val_metrics['recon']:.6f} | "
                f"kl_w={train_metrics.get('kl_weight', 0):.4f} | {elapsed:.1f}s"
            )

        # Best model checkpoint (based on val reconstruction loss)
        if val_metrics['recon'] < best_val_recon:
            best_val_recon = val_metrics['recon']
            patience_counter = 0
            save_checkpoint(model, optimizer, epoch, best_val_recon,
                            os.path.join(ckpt_dir, 'cvae_best.pth'))
        else:
            patience_counter += 1

        # Periodic checkpoint
        if epoch % 100 == 0:
            save_checkpoint(model, optimizer, epoch, best_val_recon,
                            os.path.join(ckpt_dir, f'cvae_epoch{epoch}.pth'))

        # Early stopping
        if patience > 0 and patience_counter >= patience:
            logger.info(f"  Early stopping at epoch {epoch}")
            break

    writer.close()

    # Save history
    hist_path = os.path.join(metrics_dir, 'cvae_history.json')
    with open(hist_path, 'w') as f:
        json.dump(history, f, indent=2)

    logger.info(f"\nTraining complete!")
    logger.info(f"  Best val reconstruction loss: {best_val_recon:.6f}")
    logger.info(f"  History saved to: {hist_path}")
    logger.info(f"  Best checkpoint: {os.path.join(ckpt_dir, 'cvae_best.pth')}")


if __name__ == '__main__':
    main()