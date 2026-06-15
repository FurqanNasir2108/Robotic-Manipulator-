"""
Train Conditional Diffusion Model for trajectory generation.

Usage:
    python scripts/train_diffusion.py
    python scripts/train_diffusion.py --config configs/diffusion.yaml
"""

import os
import sys
import argparse
import logging
import json
import time

import torch
from torch.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.utils.config import load_config
from src.utils.reproducibility import set_seed
from src.data.normalization import Normalizer
from src.data.loaders import get_dataloaders
from src.models.diffusion import build_diffusion_model

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def train_epoch(model, train_loader, optimizer, scaler, device, use_amp, grad_clip):
    """Run one training epoch."""
    model.train()
    total_loss = 0.0
    n_batches = 0

    for condition, target in train_loader:
        condition = condition.to(device, dtype=torch.float32)
        target = target.to(device, dtype=torch.float32)

        optimizer.zero_grad()
        with autocast('cuda', enabled=use_amp):
            loss = model.training_loss(target, condition)

        scaler.scale(loss).backward()
        if grad_clip:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def validate_epoch(model, val_loader, device, use_amp):
    """Run validation."""
    model.eval()
    total_loss = 0.0
    n_batches = 0

    for condition, target in val_loader:
        condition = condition.to(device, dtype=torch.float32)
        target = target.to(device, dtype=torch.float32)

        with autocast('cuda', enabled=use_amp):
            loss = model.training_loss(target, condition)

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


def save_checkpoint(model, ema, optimizer, epoch, best_val_loss, path):
    """Save model checkpoint with EMA weights."""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'ema_state_dict': ema.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_val_loss': best_val_loss,
    }, path)


def main():
    parser = argparse.ArgumentParser(description='Train Conditional Diffusion Model')
    parser.add_argument('--config', type=str, default='configs/diffusion.yaml')
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
    model, ema = build_diffusion_model(cfg)
    model = model.to(device)
    ema.shadow = ema.shadow.to(device)
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Diffusion model parameters: {param_count:,}")

    # Optimizer
    if tcfg.get('optimizer', 'adamw') == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), lr=tcfg['lr'],
                                       weight_decay=tcfg.get('weight_decay', 0.01))
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=tcfg['lr'])

    # Scheduler
    from src.train.schedulers import get_scheduler
    scheduler = get_scheduler(optimizer, 'cosine', T_max=tcfg['epochs'])

    scaler = GradScaler('cuda', enabled=use_amp)
    grad_clip = tcfg.get('grad_clip', None)

    # Directories
    ckpt_dir = 'results/checkpoints/diffusion'
    log_dir = 'results/logs/diffusion'
    metrics_dir = 'results/metrics/diffusion'
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)

    writer = SummaryWriter(log_dir=log_dir)

    # Training loop
    best_val_loss = float('inf')
    patience_counter = 0
    patience = tcfg.get('early_stop_patience', 80)
    epochs = tcfg['epochs']
    log_interval = tcfg.get('log_interval', 10)

    history = {'train_losses': [], 'val_losses': []}

    logger.info(f"Training diffusion model for {epochs} epochs")
    logger.info(f"Timesteps: {cfg['model']['num_timesteps']}, "
                f"Schedule: {cfg['model']['beta_schedule']}, "
                f"Guidance dropout: {cfg['model'].get('guidance_dropout', 0.1)}")

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        train_loss = train_epoch(model, loaders['train'], optimizer, scaler,
                                 device, use_amp, grad_clip)
        val_loss = validate_epoch(model, loaders['val'], device, use_amp)
        elapsed = time.time() - t0

        # Update EMA
        ema.update(model)

        history['train_losses'].append(train_loss)
        history['val_losses'].append(val_loss)

        # TensorBoard
        writer.add_scalar('diffusion/train_loss', train_loss, epoch)
        writer.add_scalar('diffusion/val_loss', val_loss, epoch)

        if scheduler:
            scheduler.step()
            writer.add_scalar('diffusion/lr', optimizer.param_groups[0]['lr'], epoch)

        # Logging
        if epoch % log_interval == 0 or epoch == 1:
            logger.info(
                f"  Epoch {epoch:4d}/{epochs} | "
                f"train={train_loss:.6f} val={val_loss:.6f} | {elapsed:.1f}s"
            )

        # Best model checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            save_checkpoint(model, ema, optimizer, epoch, best_val_loss,
                            os.path.join(ckpt_dir, 'diffusion_best.pth'))
        else:
            patience_counter += 1

        # Periodic checkpoint
        if epoch % 100 == 0:
            save_checkpoint(model, ema, optimizer, epoch, best_val_loss,
                            os.path.join(ckpt_dir, f'diffusion_epoch{epoch}.pth'))

        # Early stopping
        if patience > 0 and patience_counter >= patience:
            logger.info(f"  Early stopping at epoch {epoch}")
            break

    writer.close()

    # Save history
    hist_path = os.path.join(metrics_dir, 'diffusion_history.json')
    with open(hist_path, 'w') as f:
        json.dump(history, f, indent=2)

    logger.info(f"\nTraining complete!")
    logger.info(f"  Best val loss: {best_val_loss:.6f}")
    logger.info(f"  History saved to: {hist_path}")
    logger.info(f"  Best checkpoint: {os.path.join(ckpt_dir, 'diffusion_best.pth')}")


if __name__ == '__main__':
    main()