"""
Generic training loop supporting all model types.
Includes checkpoint saving, early stopping, TensorBoard logging, and AMP.
"""

import os
import time
import logging
import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter

logger = logging.getLogger(__name__)


class Trainer:
    """
    Generic trainer for trajectory prediction models.

    Parameters
    ----------
    model : nn.Module
    optimizer : torch.optim.Optimizer
    loss_fn : nn.Module
    scheduler : LR scheduler or None
    device : str
    checkpoint_dir : str
    log_dir : str
    use_amp : bool
        Enable mixed-precision training.
    grad_clip : float or None
        Max gradient norm for clipping.
    early_stop_patience : int
        Number of epochs without improvement before stopping. 0 = disabled.
    """

    def __init__(self, model, optimizer, loss_fn, scheduler=None,
                 device='cpu', checkpoint_dir='results/checkpoints',
                 log_dir='results/logs', use_amp=False, grad_clip=None,
                 early_stop_patience=0):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.scheduler = scheduler
        self.device = device
        self.checkpoint_dir = checkpoint_dir
        self.log_dir = log_dir
        self.use_amp = use_amp and device != 'cpu'
        self.grad_clip = grad_clip
        self.early_stop_patience = early_stop_patience

        self.scaler = GradScaler('cuda', enabled=self.use_amp)
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=log_dir)

        self.best_val_loss = float('inf')
        self.epochs_without_improvement = 0

    def train_epoch(self, train_loader):
        """Run one training epoch. Returns average loss."""
        self.model.train()
        total_loss = 0.0
        n_batches = 0
        for condition, target in train_loader:
            condition = condition.to(self.device, dtype=torch.float32)
            target = target.to(self.device, dtype=torch.float32)

            self.optimizer.zero_grad()
            with autocast('cuda', enabled=self.use_amp):
                pred = self.model(condition)
                loss = self.loss_fn(pred, target)

            self.scaler.scale(loss).backward()
            if self.grad_clip:
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += loss.item()
            n_batches += 1
        return total_loss / max(n_batches, 1)

    @torch.no_grad()
    def validate(self, val_loader):
        """Run validation. Returns average loss."""
        self.model.eval()
        total_loss = 0.0
        n_batches = 0
        for condition, target in val_loader:
            condition = condition.to(self.device, dtype=torch.float32)
            target = target.to(self.device, dtype=torch.float32)
            with autocast('cuda', enabled=self.use_amp):
                pred = self.model(condition)
                loss = self.loss_fn(pred, target)
            total_loss += loss.item()
            n_batches += 1
        return total_loss / max(n_batches, 1)

    def fit(self, train_loader, val_loader, epochs, model_name='model'):
        """
        Full training loop.

        Parameters
        ----------
        train_loader : DataLoader
        val_loader : DataLoader
        epochs : int
        model_name : str
            Prefix for checkpoint filenames.

        Returns
        -------
        dict with 'train_losses' and 'val_losses' lists.
        """
        train_losses = []
        val_losses = []

        logger.info(f"Training {model_name} for {epochs} epochs on {self.device}")
        for epoch in range(1, epochs + 1):
            t0 = time.time()
            train_loss = self.train_epoch(train_loader)
            val_loss = self.validate(val_loader)
            elapsed = time.time() - t0

            train_losses.append(train_loss)
            val_losses.append(val_loss)

            self.writer.add_scalar(f'{model_name}/train_loss', train_loss, epoch)
            self.writer.add_scalar(f'{model_name}/val_loss', val_loss, epoch)
            if self.scheduler:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()
                self.writer.add_scalar(f'{model_name}/lr',
                                       self.optimizer.param_groups[0]['lr'], epoch)

            if epoch % 10 == 0 or epoch == 1:
                logger.info(f"  Epoch {epoch:4d}/{epochs} | "
                            f"train={train_loss:.6f} val={val_loss:.6f} | {elapsed:.1f}s")

            # Checkpoint best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.epochs_without_improvement = 0
                self._save_checkpoint(model_name, epoch, is_best=True)
            else:
                self.epochs_without_improvement += 1

            # Periodic checkpoint
            if epoch % 50 == 0:
                self._save_checkpoint(model_name, epoch, is_best=False)

            # Early stopping
            if self.early_stop_patience > 0 and \
               self.epochs_without_improvement >= self.early_stop_patience:
                logger.info(f"  Early stopping at epoch {epoch}")
                break

        self.writer.close()
        return {'train_losses': train_losses, 'val_losses': val_losses}

    def _save_checkpoint(self, model_name, epoch, is_best=False):
        """Save model checkpoint."""
        state = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
        }
        if is_best:
            path = os.path.join(self.checkpoint_dir, f'{model_name}_best.pth')
        else:
            path = os.path.join(self.checkpoint_dir, f'{model_name}_epoch{epoch}.pth')
        torch.save(state, path)