"""
Conditional Diffusion Model (DDPM/DDIM) for trajectory generation.

Implements:
- Linear and cosine beta noise schedules
- 1D U-Net denoiser with time and condition embeddings
- DDPM and DDIM sampling
- Classifier-free guidance
- EMA weight averaging
"""

import math
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F


# ──────────────────────────── Noise Schedules ─────────────────────────────

def linear_beta_schedule(num_timesteps, beta_start=1e-4, beta_end=0.02):
    """Linear beta schedule from DDPM paper."""
    return torch.linspace(beta_start, beta_end, num_timesteps)


def cosine_beta_schedule(num_timesteps, s=0.008):
    """Cosine beta schedule (Nichol & Dhariwal, 2021)."""
    steps = torch.arange(num_timesteps + 1, dtype=torch.float64)
    alpha_bar = torch.cos((steps / num_timesteps + s) / (1 + s) * math.pi / 2) ** 2
    alpha_bar = alpha_bar / alpha_bar[0]
    betas = 1 - (alpha_bar[1:] / alpha_bar[:-1])
    return torch.clamp(betas, 0.0001, 0.9999).float()


# ──────────────────────────── Time Embedding ──────────────────────────────

class SinusoidalTimeEmbedding(nn.Module):
    """Sinusoidal positional encoding for diffusion timestep."""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        device = t.device
        half = self.dim // 2
        emb = math.log(10000) / (half - 1)
        emb = torch.exp(torch.arange(half, device=device, dtype=torch.float32) * -emb)
        emb = t.float().unsqueeze(1) * emb.unsqueeze(0)
        return torch.cat([emb.sin(), emb.cos()], dim=-1)


# ──────────────────────────── 1D U-Net Blocks ─────────────────────────────

class ResBlock1D(nn.Module):
    """Residual block with time embedding injection for 1D convolutions."""

    def __init__(self, in_ch, out_ch, time_dim, kernel_size=3, dropout=0.1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size, padding=kernel_size // 2)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size, padding=kernel_size // 2)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.time_proj = nn.Linear(time_dim, out_ch)
        self.dropout = nn.Dropout(dropout)
        self.skip = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, t_emb):
        h = F.silu(self.bn1(self.conv1(x)))
        h = h + self.time_proj(t_emb).unsqueeze(-1)
        h = F.silu(self.bn2(self.conv2(h)))
        h = self.dropout(h)
        return h + self.skip(x)


class DownBlock(nn.Module):
    """Downsample block: ResBlock + average pooling."""

    def __init__(self, in_ch, out_ch, time_dim, kernel_size=3, dropout=0.1):
        super().__init__()
        self.res = ResBlock1D(in_ch, out_ch, time_dim, kernel_size, dropout)
        self.pool = nn.AvgPool1d(2)

    def forward(self, x, t_emb):
        h = self.res(x, t_emb)
        return self.pool(h), h  # pooled, skip


class UpBlock(nn.Module):
    """Upsample block: interpolate + concat skip + ResBlock."""

    def __init__(self, in_ch, skip_ch, out_ch, time_dim, kernel_size=3, dropout=0.1):
        super().__init__()
        self.res = ResBlock1D(in_ch + skip_ch, out_ch, time_dim, kernel_size, dropout)

    def forward(self, x, skip, t_emb):
        x = F.interpolate(x, size=skip.shape[-1], mode='nearest')
        x = torch.cat([x, skip], dim=1)
        return self.res(x, t_emb)


# ──────────────────────────── 1D U-Net Denoiser ───────────────────────────

class UNet1D(nn.Module):
    """
    1D U-Net denoiser for diffusion model.

    Operates on trajectory tensors of shape (B, C_in, T) where C_in = joint_dim.
    Time step and condition are injected via embeddings.
    """

    def __init__(self, in_channels=3, channels=(64, 128, 256),
                 time_embed_dim=128, condition_dim=64, kernel_size=3,
                 dropout=0.1):
        super().__init__()
        self.time_embed = nn.Sequential(
            SinusoidalTimeEmbedding(time_embed_dim),
            nn.Linear(time_embed_dim, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )
        # Condition projection to add as extra input channels
        self.cond_proj = nn.Linear(condition_dim, in_channels)

        # Input: in_channels * 2 (noisy traj + projected condition)
        c_in = in_channels * 2

        # Encoder (downsampling) — each produces a skip connection
        self.downs = nn.ModuleList()
        ch_prev = c_in
        for ch in channels:
            self.downs.append(DownBlock(ch_prev, ch, time_embed_dim, kernel_size, dropout))
            ch_prev = ch

        # Bottleneck
        self.mid = ResBlock1D(channels[-1], channels[-1], time_embed_dim, kernel_size, dropout)

        # Decoder (upsampling) — symmetric to encoder
        # ups[i] receives skip from the corresponding encoder level
        self.ups = nn.ModuleList()
        rev_channels = list(reversed(channels))
        for i in range(len(rev_channels) - 1):
            # skip from same level has rev_channels[i] channels
            self.ups.append(UpBlock(rev_channels[i], rev_channels[i], rev_channels[i + 1],
                                    time_embed_dim, kernel_size, dropout))

        # Final: concat with first encoder skip (channels[0] ch)
        final_in = rev_channels[-1] if len(rev_channels) > 1 else rev_channels[0]
        self.final_up = UpBlock(final_in, channels[0], channels[0], time_embed_dim,
                                kernel_size, dropout)
        self.out_conv = nn.Conv1d(channels[0], in_channels, 1)
        self.out_time_dim = time_embed_dim

    def forward(self, x_noisy, t, cond):
        """
        Parameters
        ----------
        x_noisy : (B, C_in, T) noisy trajectory
        t : (B,) integer timesteps
        cond : (B, condition_dim) condition embedding

        Returns
        -------
        noise_pred : (B, C_in, T) predicted noise
        """
        t_emb = self.time_embed(t)  # (B, time_embed_dim)

        # Project condition and expand to match sequence length
        cond_expanded = self.cond_proj(cond).unsqueeze(-1).expand(-1, -1, x_noisy.shape[-1])
        x = torch.cat([x_noisy, cond_expanded], dim=1)  # (B, 2*C_in, T)

        # Encoder
        skips = []
        h = x
        for down in self.downs:
            h, skip = down(h, t_emb)
            skips.append(skip)

        # Bottleneck
        h = self.mid(h, t_emb)

        # Decoder — pair ups with deep skips first
        for up, skip in zip(self.ups, reversed(skips[1:])):
            h = up(h, skip, t_emb)

        # Final upsample + concat with first (shallowest) skip
        h = self.final_up(h, skips[0], t_emb)
        h = self.out_conv(h)
        return h


# ──────────────────────────── Condition Encoder ───────────────────────────

class DiffusionConditionEncoder(nn.Module):
    """Encode task-space waypoints into condition embedding (reuses cVAE design)."""

    def __init__(self, input_dim=3, conv_channels=None, kernel_sizes=None,
                 condition_dim=64, dropout=0.1):
        super().__init__()
        if conv_channels is None:
            conv_channels = [32, 64]
        if kernel_sizes is None:
            kernel_sizes = [5, 3]

        layers = []
        in_ch = input_dim
        for ch, ks in zip(conv_channels, kernel_sizes):
            layers.extend([
                nn.Conv1d(in_ch, ch, ks, padding=ks // 2),
                nn.BatchNorm1d(ch),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            ])
            in_ch = ch
        self.conv = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.proj = nn.Linear(conv_channels[-1], condition_dim)

    def forward(self, waypoints):
        """
        Parameters
        ----------
        waypoints : (B, T, input_dim)

        Returns
        -------
        cond : (B, condition_dim)
        """
        x = waypoints.permute(0, 2, 1)
        x = self.conv(x)
        x = self.pool(x).squeeze(-1)
        return self.proj(x)


# ──────────────────────────── EMA Helper ──────────────────────────────────

class EMA:
    """Exponential Moving Average for model weights."""

    def __init__(self, model, decay=0.9999):
        self.decay = decay
        self.shadow = copy.deepcopy(model)
        self.shadow.eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    def update(self, model):
        with torch.no_grad():
            for s_param, m_param in zip(self.shadow.parameters(), model.parameters()):
                s_param.data.mul_(self.decay).add_(m_param.data, alpha=1 - self.decay)
            # Copy BatchNorm running statistics (buffers, not parameters)
            for s_mod, m_mod in zip(self.shadow.modules(), model.modules()):
                if isinstance(s_mod, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.SyncBatchNorm)):
                    s_mod.running_mean.copy_(m_mod.running_mean)
                    s_mod.running_var.copy_(m_mod.running_var)
                    s_mod.num_batches_tracked.copy_(m_mod.num_batches_tracked)

    def state_dict(self):
        return self.shadow.state_dict()

    def load_state_dict(self, state_dict):
        self.shadow.load_state_dict(state_dict)


# ──────────────────────────── Diffusion Model ─────────────────────────────

class ConditionalDiffusion(nn.Module):
    """
    Full conditional diffusion model with DDPM/DDIM sampling.

    Parameters
    ----------
    denoiser : nn.Module
        The noise prediction network (e.g. UNet1D).
    condition_encoder : nn.Module
        Encoder for conditioning input (waypoints).
    num_timesteps : int
    beta_schedule : str
        'linear' or 'cosine'.
    guidance_dropout : float
        Probability of dropping condition during training (for CFG).
    joint_dim : int
        Number of joint dimensions (3 for 3-link manipulator).
    """

    def __init__(self, denoiser, condition_encoder, num_timesteps=1000,
                 beta_schedule='cosine', guidance_dropout=0.1, joint_dim=3):
        super().__init__()
        self.denoiser = denoiser
        self.condition_encoder = condition_encoder
        self.num_timesteps = num_timesteps
        self.guidance_dropout = guidance_dropout
        self.joint_dim = joint_dim

        # Compute noise schedule
        if beta_schedule == 'linear':
            betas = linear_beta_schedule(num_timesteps)
        elif beta_schedule == 'cosine':
            betas = cosine_beta_schedule(num_timesteps)
        else:
            raise ValueError(f"Unknown beta schedule: {beta_schedule}")

        alphas = 1.0 - betas
        alpha_bar = torch.cumprod(alphas, dim=0)

        # Register as buffers (move to device with model)
        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alpha_bar', alpha_bar)
        self.register_buffer('sqrt_alpha_bar', torch.sqrt(alpha_bar))
        self.register_buffer('sqrt_one_minus_alpha_bar', torch.sqrt(1.0 - alpha_bar))
        self.register_buffer('sqrt_recip_alpha', torch.sqrt(1.0 / alphas))
        self.register_buffer('posterior_variance',
                             betas * (1.0 - F.pad(alpha_bar[:-1], (1, 0), value=1.0)) /
                             (1.0 - alpha_bar))

    def q_sample(self, x_0, t, noise=None):
        """Forward diffusion: add noise to x_0 at timestep t."""
        if noise is None:
            noise = torch.randn_like(x_0)
        sqrt_ab = self.sqrt_alpha_bar[t].view(-1, 1, 1)
        sqrt_omab = self.sqrt_one_minus_alpha_bar[t].view(-1, 1, 1)
        return sqrt_ab * x_0 + sqrt_omab * noise, noise

    def training_loss(self, x_0, waypoints):
        """
        Compute training loss: MSE between predicted and actual noise.

        Parameters
        ----------
        x_0 : (B, T, joint_dim) clean trajectory (channels-last)
        waypoints : (B, T, input_dim) conditioning waypoints

        Returns
        -------
        loss : scalar
        """
        B = x_0.shape[0]
        t = torch.randint(0, self.num_timesteps, (B,), device=x_0.device)

        # Encode condition
        cond = self.condition_encoder(waypoints)  # (B, condition_dim)

        # Classifier-free guidance: randomly drop condition
        if self.training and self.guidance_dropout > 0:
            mask = torch.rand(B, device=x_0.device) < self.guidance_dropout
            cond = cond.clone()
            cond[mask] = 0.0

        # Forward diffusion
        x_0_channels = x_0.permute(0, 2, 1)  # (B, C, T)
        x_t, noise = self.q_sample(x_0_channels, t)

        # Predict noise
        noise_pred = self.denoiser(x_t, t, cond)

        return F.mse_loss(noise_pred, noise)

    @torch.no_grad()
    def ddpm_sample(self, waypoints, seq_len=100, guidance_scale=1.0):
        """
        DDPM sampling (full T steps).

        Parameters
        ----------
        waypoints : (B, T, input_dim)
        seq_len : int
        guidance_scale : float (1.0 = no guidance)

        Returns
        -------
        x_0 : (B, T, joint_dim)
        """
        self.eval()
        B = waypoints.shape[0]
        device = waypoints.device

        cond = self.condition_encoder(waypoints)
        x = torch.randn(B, self.joint_dim, seq_len, device=device)

        for i in reversed(range(self.num_timesteps)):
            t = torch.full((B,), i, device=device, dtype=torch.long)

            # Conditional prediction
            noise_pred = self.denoiser(x, t, cond)

            # Classifier-free guidance
            if guidance_scale != 1.0:
                noise_uncond = self.denoiser(x, t, torch.zeros_like(cond))
                noise_pred = noise_uncond + guidance_scale * (noise_pred - noise_uncond)

            # DDPM reverse step
            alpha = self.alphas[i]
            alpha_bar = self.alpha_bar[i]
            beta = self.betas[i]

            mean = self.sqrt_recip_alpha[i] * (
                x - beta / self.sqrt_one_minus_alpha_bar[i] * noise_pred
            )

            if i > 0:
                noise = torch.randn_like(x)
                sigma = torch.sqrt(self.posterior_variance[i])
                x = mean + sigma * noise
            else:
                x = mean

        return x.permute(0, 2, 1)  # (B, T, joint_dim)

    @torch.no_grad()
    def ddim_sample(self, waypoints, seq_len=100, num_steps=50,
                    guidance_scale=1.0, eta=0.0):
        """
        DDIM sampling (accelerated, fewer steps).

        Parameters
        ----------
        waypoints : (B, T, input_dim)
        seq_len : int
        num_steps : int
            Number of DDIM steps (< num_timesteps).
        guidance_scale : float
        eta : float
            0 = deterministic DDIM, 1 = DDPM-equivalent.

        Returns
        -------
        x_0 : (B, T, joint_dim)
        """
        self.eval()
        B = waypoints.shape[0]
        device = waypoints.device

        cond = self.condition_encoder(waypoints)
        x = torch.randn(B, self.joint_dim, seq_len, device=device)

        # Sub-sample timesteps uniformly from T-1 -> 0 with deduped integer indices.
        num_steps = max(2, min(num_steps, self.num_timesteps))
        ts = torch.linspace(self.num_timesteps - 1, 0, steps=num_steps, device=device)
        timesteps = [int(v) for v in ts.long().tolist()]
        deduped = []
        for t_idx in timesteps:
            if not deduped or t_idx != deduped[-1]:
                deduped.append(t_idx)
        timesteps = deduped

        for idx, i in enumerate(timesteps):
            t = torch.full((B,), i, device=device, dtype=torch.long)

            noise_pred = self.denoiser(x, t, cond)

            if guidance_scale != 1.0:
                noise_uncond = self.denoiser(x, t, torch.zeros_like(cond))
                noise_pred = noise_uncond + guidance_scale * (noise_pred - noise_uncond)

            alpha_bar_t = self.alpha_bar[i]
            if idx + 1 < len(timesteps):
                alpha_bar_prev = self.alpha_bar[timesteps[idx + 1]]
            else:
                alpha_bar_prev = torch.tensor(1.0, device=device)

            # Predicted x_0
            pred_x0 = (x - torch.sqrt(torch.clamp(1 - alpha_bar_t, min=1e-12)) * noise_pred) / \
                      torch.sqrt(torch.clamp(alpha_bar_t, min=1e-12))
            pred_x0 = torch.clamp(pred_x0, -5.0, 5.0)

            # Direction pointing to x_t
            sigma = eta * torch.sqrt(
                torch.clamp((1 - alpha_bar_prev) / torch.clamp(1 - alpha_bar_t, min=1e-12), min=0.0)
                * torch.clamp(1 - alpha_bar_t / torch.clamp(alpha_bar_prev, min=1e-12), min=0.0)
            )
            dir_xt = torch.sqrt(torch.clamp(1 - alpha_bar_prev - sigma ** 2, min=0.0)) * noise_pred

            noise = torch.randn_like(x) if idx + 1 < len(timesteps) else 0.0
            x = torch.sqrt(alpha_bar_prev) * pred_x0 + dir_xt + sigma * noise

        return x.permute(0, 2, 1)  # (B, T, joint_dim)

    @torch.no_grad()
    def generate(self, waypoints, num_samples=1, method='ddim',
                 num_steps=50, guidance_scale=2.0, seq_len=100):
        """
        Generate trajectories (unified interface).

        Parameters
        ----------
        waypoints : (B, T, input_dim) or (1, T, input_dim)
        num_samples : int per condition
        method : 'ddpm' or 'ddim'
        num_steps : int (for DDIM)
        guidance_scale : float
        seq_len : int

        Returns
        -------
        trajectories : (B * num_samples, T, joint_dim)
        """
        B = waypoints.shape[0]
        if num_samples > 1:
            waypoints = waypoints.unsqueeze(1).expand(B, num_samples, -1, -1)
            waypoints = waypoints.reshape(B * num_samples, waypoints.shape[2], waypoints.shape[3])

        if method == 'ddpm':
            return self.ddpm_sample(waypoints, seq_len, guidance_scale)
        elif method == 'ddim':
            return self.ddim_sample(waypoints, seq_len, num_steps, guidance_scale)
        else:
            raise ValueError(f"Unknown sampling method: {method}")


def build_diffusion_model(cfg):
    """
    Build ConditionalDiffusion from config dict.

    Parameters
    ----------
    cfg : dict with 'model' key

    Returns
    -------
    ConditionalDiffusion, EMA
    """
    m = cfg['model']

    condition_encoder = DiffusionConditionEncoder(
        input_dim=m.get('input_dim', 3),
        conv_channels=m.get('cond_conv_channels', [32, 64]),
        kernel_sizes=m.get('cond_kernel_sizes', [5, 3]),
        condition_dim=m.get('condition_dim', 64),
        dropout=m.get('dropout', 0.1),
    )

    denoiser = UNet1D(
        in_channels=m.get('joint_dim', 3),
        channels=m.get('channels', [64, 128, 256]),
        time_embed_dim=m.get('time_embed_dim', 128),
        condition_dim=m.get('condition_dim', 64),
        kernel_size=m.get('kernel_size', 3),
        dropout=m.get('dropout', 0.1),
    )

    model = ConditionalDiffusion(
        denoiser=denoiser,
        condition_encoder=condition_encoder,
        num_timesteps=m.get('num_timesteps', 1000),
        beta_schedule=m.get('beta_schedule', 'cosine'),
        guidance_dropout=m.get('guidance_dropout', 0.1),
        joint_dim=m.get('joint_dim', 3),
    )

    ema = EMA(model, decay=m.get('ema_decay', 0.9999))
    return model, ema