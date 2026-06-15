"""
Loss functions for trajectory prediction training.
"""

import torch
import torch.nn as nn


class TrajectoryMSELoss(nn.Module):
    """
    MSE loss on joint trajectories with optional endpoint weighting.

    Parameters
    ----------
    endpoint_weight : float
        Extra weight for the first and last waypoint predictions.
    """

    def __init__(self, endpoint_weight=1.0):
        super().__init__()
        self.endpoint_weight = endpoint_weight

    def forward(self, pred, target):
        """
        Parameters
        ----------
        pred : Tensor of shape (B, N, 3)
        target : Tensor of shape (B, N, 3)

        Returns
        -------
        Tensor (scalar)
        """
        mse = (pred - target) ** 2  # (B, N, 3)
        if self.endpoint_weight != 1.0:
            weights = torch.ones(pred.shape[1], device=pred.device)
            weights[0] = self.endpoint_weight
            weights[-1] = self.endpoint_weight
            weights = weights / weights.mean()
            mse = mse * weights.unsqueeze(0).unsqueeze(-1)
        return mse.mean()


class TrajectoryL1Loss(nn.Module):
    """L1 (MAE) loss on joint trajectories."""

    def forward(self, pred, target):
        return (pred - target).abs().mean()


class CombinedTrajectoryLoss(nn.Module):
    """
    Combines MSE with an energy-aware smoothness term.

    Parameters
    ----------
    mse_weight : float
    smoothness_weight : float
        Weight for the velocity-smoothness penalty.
    endpoint_weight : float
    """

    def __init__(self, mse_weight=1.0, smoothness_weight=0.01, endpoint_weight=1.0):
        super().__init__()
        self.mse_loss = TrajectoryMSELoss(endpoint_weight=endpoint_weight)
        self.mse_weight = mse_weight
        self.smoothness_weight = smoothness_weight

    def forward(self, pred, target):
        mse = self.mse_loss(pred, target)
        # Smoothness: penalize large velocity changes (jerk proxy)
        vel = pred[:, 1:, :] - pred[:, :-1, :]
        acc = vel[:, 1:, :] - vel[:, :-1, :]
        smoothness = (acc ** 2).mean()
        return self.mse_weight * mse + self.smoothness_weight * smoothness


class VAELoss(nn.Module):
    """
    Loss for Conditional VAE: reconstruction + KL + energy + smoothness.

    Parameters
    ----------
    recon_weight : float
    kl_weight : float
        Target KL weight (before annealing).
    energy_weight : float
    smoothness_weight : float
    endpoint_weight : float
    kl_annealing : str
        'monotonic', 'cyclical', or 'free_bits'.
    kl_warmup_epochs : int
    kl_cycle_epochs : int
    free_bits_lambda : float
    """

    def __init__(self, recon_weight=1.0, kl_weight=0.01, energy_weight=0.001,
                 smoothness_weight=0.001, endpoint_weight=2.0,
                 kl_annealing='cyclical', kl_warmup_epochs=50,
                 kl_cycle_epochs=20, free_bits_lambda=0.1):
        super().__init__()
        self.recon_loss = TrajectoryMSELoss(endpoint_weight=endpoint_weight)
        self.recon_weight = recon_weight
        self.kl_weight_target = kl_weight
        self.energy_weight = energy_weight
        self.smoothness_weight = smoothness_weight
        self.kl_annealing = kl_annealing
        self.kl_warmup_epochs = kl_warmup_epochs
        self.kl_cycle_epochs = kl_cycle_epochs
        self.free_bits_lambda = free_bits_lambda
        self._current_epoch = 0

    def set_epoch(self, epoch):
        """Update current epoch for KL annealing schedule."""
        self._current_epoch = epoch

    def get_kl_weight(self):
        """Compute current KL weight based on annealing schedule."""
        epoch = self._current_epoch
        if self.kl_annealing == 'monotonic':
            return self.kl_weight_target * min(1.0, epoch / max(self.kl_warmup_epochs, 1))
        elif self.kl_annealing == 'cyclical':
            cycle_pos = epoch % self.kl_cycle_epochs
            ramp = min(1.0, cycle_pos / max(self.kl_cycle_epochs * 0.5, 1))
            warmup_scale = min(1.0, epoch / max(self.kl_warmup_epochs, 1))
            return self.kl_weight_target * ramp * warmup_scale
        elif self.kl_annealing == 'free_bits':
            return self.kl_weight_target
        else:
            return self.kl_weight_target

    def kl_divergence(self, mu, log_var):
        """Compute KL divergence against standard normal prior."""
        # KL per dimension: 0.5 * (mu^2 + sigma^2 - 1 - log(sigma^2))
        kl_per_dim = -0.5 * (1 + log_var - mu.pow(2) - log_var.exp())  # (B, d)

        if self.kl_annealing == 'free_bits':
            # Free bits: only penalize KL above lambda per dimension
            kl_per_dim = torch.clamp(kl_per_dim, min=self.free_bits_lambda)

        return kl_per_dim.sum(dim=-1).mean()  # scalar

    def forward(self, pred, target, mu, log_var):
        """
        Parameters
        ----------
        pred : (B, T, 3) reconstructed trajectory
        target : (B, T, 3) ground-truth trajectory
        mu : (B, d) latent mean
        log_var : (B, d) latent log-variance

        Returns
        -------
        total_loss : scalar
        loss_dict : dict with individual loss components
        """
        recon = self.recon_loss(pred, target)
        kl = self.kl_divergence(mu, log_var)
        kl_w = self.get_kl_weight()

        # Energy: penalize large joint velocities
        vel = pred[:, 1:, :] - pred[:, :-1, :]
        energy = (vel ** 2).mean()

        # Smoothness: penalize jerk (third derivative)
        acc = vel[:, 1:, :] - vel[:, :-1, :]
        jerk = acc[:, 1:, :] - acc[:, :-1, :]
        smoothness = (jerk ** 2).mean()

        total = (self.recon_weight * recon
                 + kl_w * kl
                 + self.energy_weight * energy
                 + self.smoothness_weight * smoothness)

        loss_dict = {
            'total': total.item(),
            'recon': recon.item(),
            'kl': kl.item(),
            'kl_weight': kl_w,
            'energy': energy.item(),
            'smoothness': smoothness.item(),
        }
        return total, loss_dict