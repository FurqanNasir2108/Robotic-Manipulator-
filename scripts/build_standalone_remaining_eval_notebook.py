import json
import textwrap
from pathlib import Path


def md(source: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": textwrap.dedent(source).lstrip("\n").splitlines(keepends=True),
    }


def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": textwrap.dedent(source).lstrip("\n").splitlines(keepends=True),
    }


cells = [
    md(
        """
        # Standalone Remaining-Models Evaluation

        This notebook is intentionally **self-contained**. It does **not import anything from `src/` or any other repo module**.

        It evaluates the remaining models under the project protocol:

        - `analytical_ik`
        - `cnn`
        - `cnn_lstm`
        - `cnn_gru`
        - `diffusion_ddim`

        It can also merge those results with the already produced **full cVAE evaluation summary** to create a final cross-model comparison package.

        ## What you need to upload

        The notebook still needs the trained artifacts and dataset files. Put them under one root folder such that these paths exist:

        - `data/processed/test.npz`
        - `data/metadata/normalization_stats.json`
        - `results/checkpoints/cnn/cnn_best.pth`
        - `results/checkpoints/cnn_lstm/cnn_lstm_best.pth`
        - `results/checkpoints/cnn_gru/cnn_gru_best.pth`
        - `results/checkpoints/diffusion/diffusion_best.pth`

        Optional for the merged final table:

        - `results/metrics/cvae/cvae_full_4500_evaluation_summary.json`

        ## Usage

        1. Upload the artifact bundle or mount Google Drive.
        2. Set `ARTIFACT_ROOT` in the config cell.
        3. Keep `RUN_MODE = "smoke"` for a tiny validation run.
        4. Switch to `RUN_MODE = "full"` for the actual 4,500-sample evaluation.
        """
    ),
    code(
        """
        %pip install -q numpy scipy matplotlib tqdm
        """
    ),
    code(
        """
        import copy
        import json
        import math
        import os
        import time
        from collections import Counter, defaultdict
        from pathlib import Path

        import matplotlib.pyplot as plt
        import numpy as np
        import torch
        import torch.nn as nn
        import torch.nn.functional as F

        try:
            plt.style.use("seaborn-v0_8-whitegrid")
        except OSError:
            pass


        class Normalizer:
            def __init__(self, strategy="z_score"):
                if strategy not in ("min_max", "z_score"):
                    raise ValueError(f"Unknown normalization strategy: {strategy}")
                self.strategy = strategy
                self.stats = {}

            def transform(self, data, name):
                stats = self.stats[name]
                if self.strategy == "min_max":
                    mn = np.array(stats["min"], dtype=np.float32)
                    mx = np.array(stats["max"], dtype=np.float32)
                    denom = np.clip(mx - mn, 1e-8, None)
                    return (data - mn) / denom
                mean = np.array(stats["mean"], dtype=np.float32)
                std = np.array(stats["std"], dtype=np.float32)
                return (data - mean) / std

            def inverse_transform(self, data, name):
                stats = self.stats[name]
                if self.strategy == "min_max":
                    mn = np.array(stats["min"], dtype=np.float32)
                    mx = np.array(stats["max"], dtype=np.float32)
                    return data * (mx - mn) + mn
                mean = np.array(stats["mean"], dtype=np.float32)
                std = np.array(stats["std"], dtype=np.float32)
                return data * std + mean

            @classmethod
            def load(cls, path):
                with open(path, "r", encoding="utf-8") as f:
                    payload = json.load(f)
                norm = cls(strategy=payload["strategy"])
                norm.stats = payload["stats"]
                return norm


        class TrajectoryDataset:
            def __init__(self, npz_path, normalizer=None, input_mode="waypoints"):
                data = np.load(npz_path, allow_pickle=True)
                self.waypoints = data["waypoints"].astype(np.float32)
                self.q_sequence = data["q_sequence"].astype(np.float32)
                self.start_pose = data["start_pose"].astype(np.float32)
                self.goal_pose = data["goal_pose"].astype(np.float32)
                self.shape_type = data["shape_type"]
                self.input_mode = input_mode
                self.normalizer = normalizer

                if normalizer is not None:
                    self.waypoints = normalizer.transform(self.waypoints, "waypoints").astype(np.float32)
                    self.q_sequence = normalizer.transform(self.q_sequence, "q_sequence").astype(np.float32)

            def __len__(self):
                return len(self.q_sequence)

            def __getitem__(self, idx):
                target = self.q_sequence[idx]
                if self.input_mode == "waypoints":
                    condition = self.waypoints[idx]
                else:
                    condition = np.concatenate([self.start_pose[idx], self.goal_pose[idx]], axis=0)
                return condition.astype(np.float32), target.astype(np.float32)


        class ThreeLinkManipulator:
            def __init__(self):
                self.link_lengths = np.array([1.0, 1.0, 0.5], dtype=np.float32)
                self.joint_limits = np.array(
                    [
                        [-math.pi, math.pi],
                        [-math.pi, math.pi],
                        [-math.pi, math.pi],
                    ],
                    dtype=np.float32,
                )
                self.base_position = np.array([0.0, 0.0], dtype=np.float32)

            def forward_kinematics(self, q):
                l1, l2, l3 = self.link_lengths
                q1, q2, q3 = q
                x = self.base_position[0] + l1 * np.cos(q1) + l2 * np.cos(q1 + q2) + l3 * np.cos(q1 + q2 + q3)
                y = self.base_position[1] + l1 * np.sin(q1) + l2 * np.sin(q1 + q2) + l3 * np.sin(q1 + q2 + q3)
                theta = q1 + q2 + q3
                return np.array([x, y, theta], dtype=np.float32)

            def check_joint_limits(self, q):
                q = np.asarray(q)
                return np.all((q >= self.joint_limits[:, 0]) & (q <= self.joint_limits[:, 1]))

            def inverse_kinematics(self, pose, elbow="up"):
                x, y, theta = pose
                l1, l2, l3 = self.link_lengths
                wx = x - l3 * np.cos(theta)
                wy = y - l3 * np.sin(theta)
                dx = wx - self.base_position[0]
                dy = wy - self.base_position[1]
                D = (dx**2 + dy**2 - l1**2 - l2**2) / (2 * l1 * l2)
                if np.abs(D) > 1.0:
                    return []
                q2_options = [np.arccos(D), -np.arccos(D)]
                if elbow == "up":
                    q2_options = [q2_options[0]]
                elif elbow == "down":
                    q2_options = [q2_options[1]]
                solutions = []
                for q2 in q2_options:
                    k1 = l1 + l2 * np.cos(q2)
                    k2 = l2 * np.sin(q2)
                    q1 = np.arctan2(dy, dx) - np.arctan2(k2, k1)
                    q3 = theta - q1 - q2
                    q = np.array([q1, q2, q3], dtype=np.float32)
                    if self.check_joint_limits(q):
                        solutions.append(q)
                return solutions


        class AnalyticalIKBaseline:
            def __init__(self):
                self.robot = ThreeLinkManipulator()

            def predict(self, waypoints):
                n_steps = len(waypoints)
                q_seq = np.zeros((n_steps, 3), dtype=np.float32)
                for i in range(n_steps):
                    solutions = self.robot.inverse_kinematics(waypoints[i], elbow="both")
                    if len(solutions) == 0:
                        return None
                    if i == 0:
                        energies = [np.sum(s**2) for s in solutions]
                        q_seq[0] = solutions[int(np.argmin(energies))]
                    else:
                        dists = [np.linalg.norm(s - q_seq[i - 1]) for s in solutions]
                        q_seq[i] = solutions[int(np.argmin(dists))]
                return q_seq


        def _get_activation(name):
            activations = {
                "relu": nn.ReLU(inplace=True),
                "gelu": nn.GELU(),
                "leaky_relu": nn.LeakyReLU(0.1, inplace=True),
                "tanh": nn.Tanh(),
            }
            if name not in activations:
                raise ValueError(f"Unknown activation: {name}")
            return activations[name]


        class CNNTrajectoryRegressor(nn.Module):
            def __init__(
                self,
                input_steps=100,
                input_dim=3,
                output_steps=100,
                output_dim=3,
                conv_channels=None,
                kernel_sizes=None,
                fc_dims=None,
                activation="relu",
                dropout=0.1,
            ):
                super().__init__()
                conv_channels = conv_channels or [32, 64, 128]
                kernel_sizes = kernel_sizes or [5, 5, 3]
                fc_dims = fc_dims or [256, 128]
                self.output_steps = output_steps
                self.output_dim = output_dim
                act = _get_activation(activation)
                conv_layers = []
                in_ch = input_dim
                for ch, ks in zip(conv_channels, kernel_sizes):
                    conv_layers.extend(
                        [
                            nn.Conv1d(in_ch, ch, ks, padding=ks // 2),
                            nn.BatchNorm1d(ch),
                            act,
                            nn.Dropout(dropout),
                        ]
                    )
                    in_ch = ch
                self.conv = nn.Sequential(*conv_layers)
                self.pool = nn.AdaptiveAvgPool1d(1)
                fc_layers = []
                fc_in = conv_channels[-1]
                for dim in fc_dims:
                    fc_layers.extend([nn.Linear(fc_in, dim), act, nn.Dropout(dropout)])
                    fc_in = dim
                fc_layers.append(nn.Linear(fc_in, output_steps * output_dim))
                self.fc = nn.Sequential(*fc_layers)

            def forward(self, x):
                x = x.permute(0, 2, 1)
                x = self.conv(x)
                x = self.pool(x).squeeze(-1)
                x = self.fc(x)
                return x.view(-1, self.output_steps, self.output_dim)


        class CNNLSTMRegressor(nn.Module):
            def __init__(
                self,
                input_steps=100,
                input_dim=3,
                output_steps=100,
                output_dim=3,
                conv_channels=None,
                kernel_sizes=None,
                lstm_hidden=128,
                lstm_layers=2,
                lstm_dropout=0.1,
                fc_dims=None,
                activation="relu",
                dropout=0.1,
            ):
                super().__init__()
                conv_channels = conv_channels or [32, 64]
                kernel_sizes = kernel_sizes or [5, 3]
                fc_dims = fc_dims or [64]
                act = _get_activation(activation)
                conv_layers = []
                in_ch = input_dim
                for ch, ks in zip(conv_channels, kernel_sizes):
                    conv_layers.extend(
                        [
                            nn.Conv1d(in_ch, ch, ks, padding=ks // 2),
                            nn.BatchNorm1d(ch),
                            act,
                            nn.Dropout(dropout),
                        ]
                    )
                    in_ch = ch
                self.encoder = nn.Sequential(*conv_layers)
                self.lstm = nn.LSTM(
                    input_size=conv_channels[-1],
                    hidden_size=lstm_hidden,
                    num_layers=lstm_layers,
                    batch_first=True,
                    dropout=lstm_dropout if lstm_layers > 1 else 0.0,
                )
                fc_layers = []
                fc_in = lstm_hidden
                for dim in fc_dims:
                    fc_layers.extend([nn.Linear(fc_in, dim), act, nn.Dropout(dropout)])
                    fc_in = dim
                fc_layers.append(nn.Linear(fc_in, output_dim))
                self.output_proj = nn.Sequential(*fc_layers)

            def forward(self, x):
                enc = self.encoder(x.permute(0, 2, 1))
                enc = enc.permute(0, 2, 1)
                lstm_out, _ = self.lstm(enc)
                return self.output_proj(lstm_out)


        class CNNGRURegressor(nn.Module):
            def __init__(
                self,
                input_steps=100,
                input_dim=3,
                output_steps=100,
                output_dim=3,
                conv_channels=None,
                kernel_sizes=None,
                gru_hidden=128,
                gru_layers=2,
                gru_dropout=0.1,
                fc_dims=None,
                activation="relu",
                dropout=0.1,
            ):
                super().__init__()
                conv_channels = conv_channels or [32, 64]
                kernel_sizes = kernel_sizes or [5, 3]
                fc_dims = fc_dims or [64]
                act = _get_activation(activation)
                conv_layers = []
                in_ch = input_dim
                for ch, ks in zip(conv_channels, kernel_sizes):
                    conv_layers.extend(
                        [
                            nn.Conv1d(in_ch, ch, ks, padding=ks // 2),
                            nn.BatchNorm1d(ch),
                            act,
                            nn.Dropout(dropout),
                        ]
                    )
                    in_ch = ch
                self.encoder = nn.Sequential(*conv_layers)
                self.gru = nn.GRU(
                    input_size=conv_channels[-1],
                    hidden_size=gru_hidden,
                    num_layers=gru_layers,
                    batch_first=True,
                    dropout=gru_dropout if gru_layers > 1 else 0.0,
                )
                fc_layers = []
                fc_in = gru_hidden
                for dim in fc_dims:
                    fc_layers.extend([nn.Linear(fc_in, dim), act, nn.Dropout(dropout)])
                    fc_in = dim
                fc_layers.append(nn.Linear(fc_in, output_dim))
                self.output_proj = nn.Sequential(*fc_layers)

            def forward(self, x):
                enc = self.encoder(x.permute(0, 2, 1))
                enc = enc.permute(0, 2, 1)
                gru_out, _ = self.gru(enc)
                return self.output_proj(gru_out)
        """
    ),
    code(
        """
        def linear_beta_schedule(num_timesteps, beta_start=1e-4, beta_end=0.02):
            return torch.linspace(beta_start, beta_end, num_timesteps)


        def cosine_beta_schedule(num_timesteps, s=0.008):
            steps = torch.arange(num_timesteps + 1, dtype=torch.float64)
            alpha_bar = torch.cos((steps / num_timesteps + s) / (1 + s) * math.pi / 2) ** 2
            alpha_bar = alpha_bar / alpha_bar[0]
            betas = 1 - (alpha_bar[1:] / alpha_bar[:-1])
            return torch.clamp(betas, 0.0001, 0.9999).float()


        class SinusoidalTimeEmbedding(nn.Module):
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


        class ResBlock1D(nn.Module):
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
            def __init__(self, in_ch, out_ch, time_dim, kernel_size=3, dropout=0.1):
                super().__init__()
                self.res = ResBlock1D(in_ch, out_ch, time_dim, kernel_size, dropout)
                self.pool = nn.AvgPool1d(2)

            def forward(self, x, t_emb):
                h = self.res(x, t_emb)
                return self.pool(h), h


        class UpBlock(nn.Module):
            def __init__(self, in_ch, skip_ch, out_ch, time_dim, kernel_size=3, dropout=0.1):
                super().__init__()
                self.res = ResBlock1D(in_ch + skip_ch, out_ch, time_dim, kernel_size, dropout)

            def forward(self, x, skip, t_emb):
                x = F.interpolate(x, size=skip.shape[-1], mode="nearest")
                x = torch.cat([x, skip], dim=1)
                return self.res(x, t_emb)


        class UNet1D(nn.Module):
            def __init__(
                self,
                in_channels=3,
                channels=(64, 128, 256),
                time_embed_dim=128,
                condition_dim=64,
                kernel_size=3,
                dropout=0.1,
            ):
                super().__init__()
                self.time_embed = nn.Sequential(
                    SinusoidalTimeEmbedding(time_embed_dim),
                    nn.Linear(time_embed_dim, time_embed_dim),
                    nn.SiLU(),
                    nn.Linear(time_embed_dim, time_embed_dim),
                )
                self.cond_proj = nn.Linear(condition_dim, in_channels)
                c_in = in_channels * 2
                self.downs = nn.ModuleList()
                ch_prev = c_in
                for ch in channels:
                    self.downs.append(DownBlock(ch_prev, ch, time_embed_dim, kernel_size, dropout))
                    ch_prev = ch
                self.mid = ResBlock1D(channels[-1], channels[-1], time_embed_dim, kernel_size, dropout)
                self.ups = nn.ModuleList()
                rev_channels = list(reversed(channels))
                for i in range(len(rev_channels) - 1):
                    self.ups.append(
                        UpBlock(
                            rev_channels[i],
                            rev_channels[i],
                            rev_channels[i + 1],
                            time_embed_dim,
                            kernel_size,
                            dropout,
                        )
                    )
                final_in = rev_channels[-1] if len(rev_channels) > 1 else rev_channels[0]
                self.final_up = UpBlock(final_in, channels[0], channels[0], time_embed_dim, kernel_size, dropout)
                self.out_conv = nn.Conv1d(channels[0], in_channels, 1)

            def forward(self, x_noisy, t, cond):
                t_emb = self.time_embed(t)
                cond_expanded = self.cond_proj(cond).unsqueeze(-1).expand(-1, -1, x_noisy.shape[-1])
                x = torch.cat([x_noisy, cond_expanded], dim=1)
                skips = []
                h = x
                for down in self.downs:
                    h, skip = down(h, t_emb)
                    skips.append(skip)
                h = self.mid(h, t_emb)
                for up, skip in zip(self.ups, reversed(skips[1:])):
                    h = up(h, skip, t_emb)
                h = self.final_up(h, skips[0], t_emb)
                return self.out_conv(h)


        class DiffusionConditionEncoder(nn.Module):
            def __init__(self, input_dim=3, conv_channels=None, kernel_sizes=None, condition_dim=64, dropout=0.1):
                super().__init__()
                conv_channels = conv_channels or [32, 64]
                kernel_sizes = kernel_sizes or [5, 3]
                layers = []
                in_ch = input_dim
                for ch, ks in zip(conv_channels, kernel_sizes):
                    layers.extend(
                        [
                            nn.Conv1d(in_ch, ch, ks, padding=ks // 2),
                            nn.BatchNorm1d(ch),
                            nn.ReLU(inplace=True),
                            nn.Dropout(dropout),
                        ]
                    )
                    in_ch = ch
                self.conv = nn.Sequential(*layers)
                self.pool = nn.AdaptiveAvgPool1d(1)
                self.proj = nn.Linear(conv_channels[-1], condition_dim)

            def forward(self, waypoints):
                x = waypoints.permute(0, 2, 1)
                x = self.conv(x)
                x = self.pool(x).squeeze(-1)
                return self.proj(x)


        class ConditionalDiffusion(nn.Module):
            def __init__(self, denoiser, condition_encoder, num_timesteps=1000, beta_schedule="cosine", guidance_dropout=0.1, joint_dim=3):
                super().__init__()
                self.denoiser = denoiser
                self.condition_encoder = condition_encoder
                self.num_timesteps = num_timesteps
                self.guidance_dropout = guidance_dropout
                self.joint_dim = joint_dim
                if beta_schedule == "linear":
                    betas = linear_beta_schedule(num_timesteps)
                elif beta_schedule == "cosine":
                    betas = cosine_beta_schedule(num_timesteps)
                else:
                    raise ValueError(f"Unknown beta schedule: {beta_schedule}")
                alphas = 1.0 - betas
                alpha_bar = torch.cumprod(alphas, dim=0)
                self.register_buffer("betas", betas)
                self.register_buffer("alphas", alphas)
                self.register_buffer("alpha_bar", alpha_bar)
                self.register_buffer("sqrt_alpha_bar", torch.sqrt(alpha_bar))
                self.register_buffer("sqrt_one_minus_alpha_bar", torch.sqrt(1.0 - alpha_bar))
                self.register_buffer("sqrt_recip_alpha", torch.sqrt(1.0 / alphas))
                self.register_buffer(
                    "posterior_variance",
                    betas * (1.0 - F.pad(alpha_bar[:-1], (1, 0), value=1.0)) / (1.0 - alpha_bar),
                )

            @torch.no_grad()
            def ddim_sample(self, waypoints, seq_len=100, num_steps=50, guidance_scale=1.0, eta=0.0):
                self.eval()
                batch_size = waypoints.shape[0]
                device = waypoints.device
                cond = self.condition_encoder(waypoints)
                x = torch.randn(batch_size, self.joint_dim, seq_len, device=device)
                num_steps = max(2, min(num_steps, self.num_timesteps))
                ts = torch.linspace(self.num_timesteps - 1, 0, steps=num_steps, device=device)
                timesteps = [int(v) for v in ts.long().tolist()]
                deduped = []
                for t_idx in timesteps:
                    if not deduped or t_idx != deduped[-1]:
                        deduped.append(t_idx)
                timesteps = deduped
                for idx, i in enumerate(timesteps):
                    t = torch.full((batch_size,), i, device=device, dtype=torch.long)
                    noise_pred = self.denoiser(x, t, cond)
                    if guidance_scale != 1.0:
                        noise_uncond = self.denoiser(x, t, torch.zeros_like(cond))
                        noise_pred = noise_uncond + guidance_scale * (noise_pred - noise_uncond)
                    alpha_bar_t = self.alpha_bar[i]
                    if idx + 1 < len(timesteps):
                        alpha_bar_prev = self.alpha_bar[timesteps[idx + 1]]
                    else:
                        alpha_bar_prev = torch.tensor(1.0, device=device)
                    pred_x0 = (
                        x - torch.sqrt(torch.clamp(1 - alpha_bar_t, min=1e-12)) * noise_pred
                    ) / torch.sqrt(torch.clamp(alpha_bar_t, min=1e-12))
                    pred_x0 = torch.clamp(pred_x0, -5.0, 5.0)
                    sigma = eta * torch.sqrt(
                        torch.clamp((1 - alpha_bar_prev) / torch.clamp(1 - alpha_bar_t, min=1e-12), min=0.0)
                        * torch.clamp(1 - alpha_bar_t / torch.clamp(alpha_bar_prev, min=1e-12), min=0.0)
                    )
                    dir_xt = torch.sqrt(torch.clamp(1 - alpha_bar_prev - sigma**2, min=0.0)) * noise_pred
                    noise = torch.randn_like(x) if idx + 1 < len(timesteps) else 0.0
                    x = torch.sqrt(alpha_bar_prev) * pred_x0 + dir_xt + sigma * noise
                return x.permute(0, 2, 1)

            @torch.no_grad()
            def generate(self, waypoints, num_samples=1, method="ddim", num_steps=50, guidance_scale=2.0, seq_len=100):
                batch_size = waypoints.shape[0]
                if num_samples > 1:
                    waypoints = waypoints.unsqueeze(1).expand(batch_size, num_samples, -1, -1)
                    waypoints = waypoints.reshape(batch_size * num_samples, waypoints.shape[2], waypoints.shape[3])
                if method != "ddim":
                    raise ValueError("This standalone notebook currently supports DDIM evaluation only.")
                return self.ddim_sample(waypoints, seq_len=seq_len, num_steps=num_steps, guidance_scale=guidance_scale)


        def build_diffusion_model():
            model = ConditionalDiffusion(
                denoiser=UNet1D(
                    in_channels=3,
                    channels=[64, 128, 256],
                    time_embed_dim=128,
                    condition_dim=64,
                    kernel_size=3,
                    dropout=0.1,
                ),
                condition_encoder=DiffusionConditionEncoder(
                    input_dim=3,
                    conv_channels=[32, 64],
                    kernel_sizes=[5, 3],
                    condition_dim=64,
                    dropout=0.1,
                ),
                num_timesteps=1000,
                beta_schedule="cosine",
                guidance_dropout=0.1,
                joint_dim=3,
            )
            return model
        """
    ),
    code(
        """
        def _to_numpy(array_like):
            if hasattr(array_like, "detach"):
                return array_like.detach().cpu().numpy()
            return np.asarray(array_like)


        def _wrap_angle(angle):
            return (angle + np.pi) % (2 * np.pi) - np.pi


        def _trajectory_point_dist(a, b):
            return float(np.linalg.norm(np.asarray(a) - np.asarray(b)))


        def _dtw_distance(path_a, path_b):
            path_a = _to_numpy(path_a)
            path_b = _to_numpy(path_b)
            n_steps_a = len(path_a)
            n_steps_b = len(path_b)
            cost = np.full((n_steps_a + 1, n_steps_b + 1), np.inf, dtype=np.float64)
            cost[0, 0] = 0.0
            for i in range(1, n_steps_a + 1):
                for j in range(1, n_steps_b + 1):
                    dist = _trajectory_point_dist(path_a[i - 1], path_b[j - 1])
                    cost[i, j] = dist + min(cost[i - 1, j], cost[i, j - 1], cost[i - 1, j - 1])
            return float(cost[n_steps_a, n_steps_b] / max(n_steps_a + n_steps_b, 1))


        def joint_rmse(pred, gt):
            pred = _to_numpy(pred)
            gt = _to_numpy(gt)
            return float(np.sqrt(np.mean((pred - gt) ** 2)))


        def ee_position_error(pred, gt, fk_fn):
            pred = _to_numpy(pred)
            gt = _to_numpy(gt)
            pred_poses = np.asarray([fk_fn(q) for q in pred])
            gt_poses = np.asarray([fk_fn(q) for q in gt])
            return float(np.mean(np.linalg.norm(pred_poses[:, :2] - gt_poses[:, :2], axis=1)))


        def ee_orientation_error(pred, gt, fk_fn):
            pred = _to_numpy(pred)
            gt = _to_numpy(gt)
            pred_theta = np.asarray([fk_fn(q)[2] for q in pred])
            gt_theta = np.asarray([fk_fn(q)[2] for q in gt])
            return float(np.mean(np.abs(_wrap_angle(pred_theta - gt_theta))))


        def path_tracking_error(pred, gt, method="dtw"):
            pred = _to_numpy(pred)
            gt = _to_numpy(gt)
            if method != "dtw":
                raise ValueError("This standalone notebook currently uses DTW path tracking.")
            return _dtw_distance(pred, gt)


        def energy_proxy(trajectory):
            trajectory = _to_numpy(trajectory)
            velocity = np.diff(trajectory, axis=0)
            return float(np.mean(np.sum(velocity ** 2, axis=-1)))


        def smoothness_jerk(trajectory, dt):
            trajectory = _to_numpy(trajectory)
            velocity = np.diff(trajectory, axis=0) / dt
            acceleration = np.diff(velocity, axis=0) / dt
            jerk = np.diff(acceleration, axis=0) / dt
            if jerk.size == 0:
                return 0.0
            value = float(np.mean(np.sum(jerk ** 2, axis=-1)))
            return 0.0 if abs(value) < 1e-12 else value


        def inference_time(model_fn, condition, n_runs=10):
            latencies_ms = []
            for _ in range(n_runs):
                start = time.perf_counter()
                output = model_fn(condition)
                if hasattr(output, "detach"):
                    output = output.detach()
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                end = time.perf_counter()
                latencies_ms.append((end - start) * 1000.0)
            return float(np.median(latencies_ms))


        def diversity_score(samples):
            samples = _to_numpy(samples)
            n_samples = len(samples)
            if n_samples < 2:
                return 0.0
            distances = []
            for i in range(n_samples):
                for j in range(i + 1, n_samples):
                    distances.append(path_tracking_error(samples[i], samples[j], method="dtw"))
            return float(np.mean(distances))


        def constraint_violation_rate(trajectory, joint_limits):
            trajectory = _to_numpy(trajectory)
            joint_limits = _to_numpy(joint_limits)
            lower = joint_limits[:, 0]
            upper = joint_limits[:, 1]
            violations = (trajectory < lower) | (trajectory > upper)
            return float(np.mean(violations))


        def _shape_counts(shape_types):
            counts = Counter(str(shape) for shape in shape_types)
            return dict(sorted(counts.items()))


        def _slice_dataset(dataset, indices):
            dataset.waypoints = dataset.waypoints[indices]
            dataset.q_sequence = dataset.q_sequence[indices]
            dataset.start_pose = dataset.start_pose[indices]
            dataset.goal_pose = dataset.goal_pose[indices]
            dataset.shape_type = dataset.shape_type[indices]
            return dataset


        def _allocate_stratified_counts(shape_counts, max_samples):
            shapes = sorted(shape_counts)
            allocation = {shape: 0 for shape in shapes}
            if max_samples <= 0 or not shapes:
                return allocation
            residual_capacity = dict(shape_counts)
            if max_samples >= len(shapes):
                for shape in shapes:
                    allocation[shape] = 1
                    residual_capacity[shape] -= 1
                remaining = max_samples - len(shapes)
            else:
                remaining = max_samples
            total_residual = sum(residual_capacity.values())
            if remaining <= 0 or total_residual <= 0:
                return allocation
            exact = {shape: remaining * residual_capacity[shape] / total_residual for shape in shapes}
            allocated = 0
            for shape in shapes:
                add = min(int(np.floor(exact[shape])), residual_capacity[shape])
                allocation[shape] += add
                allocated += add
            leftover = remaining - allocated
            order = sorted(
                shapes,
                key=lambda shape: (exact[shape] - np.floor(exact[shape]), residual_capacity[shape]),
                reverse=True,
            )
            while leftover > 0:
                progressed = False
                for shape in order:
                    if allocation[shape] < shape_counts[shape]:
                        allocation[shape] += 1
                        leftover -= 1
                        progressed = True
                        if leftover == 0:
                            break
                if not progressed:
                    break
            return allocation


        def _sample_subset_indices(shape_types, max_samples=None, seed=42, strategy="stratified"):
            n_samples = len(shape_types)
            all_indices = np.arange(n_samples, dtype=int)
            if max_samples is None or max_samples >= n_samples:
                return all_indices
            if max_samples <= 0:
                return np.array([], dtype=int)
            rng = np.random.default_rng(seed)
            if strategy == "ordered":
                return all_indices[:max_samples]
            if strategy == "random":
                selected = rng.choice(all_indices, size=max_samples, replace=False)
                return np.sort(selected)
            if strategy != "stratified":
                raise ValueError(f"Unknown subset strategy: {strategy}")
            by_shape = defaultdict(list)
            for index, shape in enumerate(shape_types):
                by_shape[str(shape)].append(index)
            allocation = _allocate_stratified_counts(
                {shape: len(indices) for shape, indices in by_shape.items()},
                max_samples=max_samples,
            )
            selected = []
            for shape, indices in by_shape.items():
                take = allocation.get(shape, 0)
                if take <= 0:
                    continue
                chosen = rng.choice(np.asarray(indices, dtype=int), size=take, replace=False)
                selected.extend(chosen.tolist())
            return np.sort(np.asarray(selected, dtype=int))


        def load_test_dataset(data_dir, metadata_dir, max_samples=None, seed=42, subset_strategy="stratified"):
            normalizer = Normalizer.load(str(Path(metadata_dir) / "normalization_stats.json"))
            dataset = TrajectoryDataset(str(Path(data_dir) / "test.npz"), normalizer=normalizer, input_mode="waypoints")
            dataset.original_size = len(dataset)
            dataset.original_shape_counts = _shape_counts(dataset.shape_type)
            if max_samples is not None and max_samples < len(dataset):
                indices = _sample_subset_indices(
                    dataset.shape_type,
                    max_samples=max_samples,
                    seed=seed,
                    strategy=subset_strategy,
                )
                _slice_dataset(dataset, indices)
                dataset.selected_indices = indices
                dataset.subset_strategy = subset_strategy
            else:
                dataset.selected_indices = np.arange(len(dataset), dtype=int)
                dataset.subset_strategy = "full"
            dataset.shape_counts = _shape_counts(dataset.shape_type)
            return dataset, normalizer


        def _extract_state_dict(payload):
            if isinstance(payload, dict) and "model_state_dict" in payload:
                return payload["model_state_dict"]
            return payload


        def load_model_specs(device, artifact_root):
            artifact_root = Path(artifact_root)
            specs = {}

            robot_model = AnalyticalIKBaseline()
            specs["analytical_ik"] = {
                "kind": "deterministic",
                "predict": lambda condition, num_samples=1, model=robot_model: np.expand_dims(model.predict(condition), axis=0),
                "timing": lambda condition, model=robot_model: model.predict(condition),
                "uses_raw_waypoints": True,
                "outputs_normalized": False,
            }

            cnn = CNNTrajectoryRegressor(
                input_steps=100,
                input_dim=3,
                output_steps=100,
                output_dim=3,
                conv_channels=[32, 64, 128],
                kernel_sizes=[5, 5, 3],
                fc_dims=[256, 128],
                activation="relu",
                dropout=0.1,
            ).to(device)
            payload = torch.load(artifact_root / "results" / "checkpoints" / "cnn" / "cnn_best.pth", map_location=device)
            cnn.load_state_dict(_extract_state_dict(payload))
            cnn.eval()
            specs["cnn"] = {
                "kind": "deterministic",
                "predict": lambda condition, num_samples=1, model=cnn: model(torch.from_numpy(condition).unsqueeze(0).to(device=device, dtype=torch.float32)).detach().cpu().numpy(),
                "timing": lambda condition, model=cnn: model(torch.from_numpy(condition).unsqueeze(0).to(device=device, dtype=torch.float32)),
                "uses_raw_waypoints": False,
                "outputs_normalized": True,
            }

            cnn_lstm = CNNLSTMRegressor(
                input_steps=100,
                input_dim=3,
                output_steps=100,
                output_dim=3,
                conv_channels=[32, 64],
                kernel_sizes=[5, 3],
                lstm_hidden=128,
                lstm_layers=2,
                lstm_dropout=0.1,
                fc_dims=[64],
                activation="relu",
                dropout=0.1,
            ).to(device)
            payload = torch.load(artifact_root / "results" / "checkpoints" / "cnn_lstm" / "cnn_lstm_best.pth", map_location=device)
            cnn_lstm.load_state_dict(_extract_state_dict(payload))
            cnn_lstm.eval()
            specs["cnn_lstm"] = {
                "kind": "deterministic",
                "predict": lambda condition, num_samples=1, model=cnn_lstm: model(torch.from_numpy(condition).unsqueeze(0).to(device=device, dtype=torch.float32)).detach().cpu().numpy(),
                "timing": lambda condition, model=cnn_lstm: model(torch.from_numpy(condition).unsqueeze(0).to(device=device, dtype=torch.float32)),
                "uses_raw_waypoints": False,
                "outputs_normalized": True,
            }

            cnn_gru = CNNGRURegressor(
                input_steps=100,
                input_dim=3,
                output_steps=100,
                output_dim=3,
                conv_channels=[32, 64],
                kernel_sizes=[5, 3],
                gru_hidden=128,
                gru_layers=2,
                gru_dropout=0.1,
                fc_dims=[64],
                activation="relu",
                dropout=0.1,
            ).to(device)
            payload = torch.load(artifact_root / "results" / "checkpoints" / "cnn_gru" / "cnn_gru_best.pth", map_location=device)
            cnn_gru.load_state_dict(_extract_state_dict(payload))
            cnn_gru.eval()
            specs["cnn_gru"] = {
                "kind": "deterministic",
                "predict": lambda condition, num_samples=1, model=cnn_gru: model(torch.from_numpy(condition).unsqueeze(0).to(device=device, dtype=torch.float32)).detach().cpu().numpy(),
                "timing": lambda condition, model=cnn_gru: model(torch.from_numpy(condition).unsqueeze(0).to(device=device, dtype=torch.float32)),
                "uses_raw_waypoints": False,
                "outputs_normalized": True,
            }

            diffusion = build_diffusion_model().to(device)
            payload = torch.load(artifact_root / "results" / "checkpoints" / "diffusion" / "diffusion_best.pth", map_location=device)
            diffusion.load_state_dict(_extract_state_dict(payload))
            diffusion.eval()
            specs["diffusion_ddim"] = {
                "kind": "generative",
                "predict": lambda condition, num_samples=10, model=diffusion: model.generate(
                    torch.from_numpy(condition).unsqueeze(0).to(device=device, dtype=torch.float32),
                    num_samples=num_samples,
                    method="ddim",
                    num_steps=50,
                    guidance_scale=2.0,
                    seq_len=100,
                ).detach().cpu().numpy(),
                "timing": lambda condition, model=diffusion: model.generate(
                    torch.from_numpy(condition).unsqueeze(0).to(device=device, dtype=torch.float32),
                    num_samples=1,
                    method="ddim",
                    num_steps=50,
                    guidance_scale=2.0,
                    seq_len=100,
                ),
                "uses_raw_waypoints": False,
                "outputs_normalized": True,
            }

            return specs


        def _metric_row(prediction, target, fk_fn, joint_limits, dt=1.0):
            return {
                "joint_rmse": joint_rmse(prediction, target),
                "ee_position_error": ee_position_error(prediction, target, fk_fn),
                "ee_orientation_error": ee_orientation_error(prediction, target, fk_fn),
                "path_tracking_error": path_tracking_error(prediction, target, method="dtw"),
                "energy_proxy": energy_proxy(prediction),
                "smoothness_jerk": smoothness_jerk(prediction, dt=dt),
                "constraint_violation_rate": constraint_violation_rate(prediction, joint_limits),
            }


        def _aggregate_rows(rows):
            metric_names = rows[0].keys()
            return {name: float(np.mean([row[name] for row in rows])) for name in metric_names}


        def _extract_sample(dataset, normalizer, index, uses_raw_waypoints=False):
            condition_norm, target_norm = dataset[index]
            target = normalizer.inverse_transform(target_norm, "q_sequence")
            waypoints = normalizer.inverse_transform(condition_norm, "waypoints")
            shape = str(dataset.shape_type[index])
            model_input = waypoints if uses_raw_waypoints else condition_norm
            return {
                "target": target,
                "waypoints": waypoints,
                "shape": shape,
                "model_input": model_input,
            }


        def _benchmark_model_latency(spec, dataset, normalizer, seed=42, subset_strategy="stratified", n_conditions=3, n_runs=10):
            if len(dataset) == 0:
                return 0.0, {"num_conditions": 0, "condition_indices": [], "per_condition_ms": []}
            n_conditions = max(1, min(int(n_conditions), len(dataset)))
            indices = _sample_subset_indices(dataset.shape_type, max_samples=n_conditions, seed=seed, strategy=subset_strategy)
            latencies = []
            for index in indices:
                sample = _extract_sample(dataset, normalizer, index=index, uses_raw_waypoints=spec.get("uses_raw_waypoints", False))
                latencies.append(inference_time(spec["timing"], sample["model_input"], n_runs=n_runs))
            return float(np.mean(latencies)), {
                "num_conditions": len(indices),
                "condition_indices": indices.tolist(),
                "per_condition_ms": [float(value) for value in latencies],
            }


        def _dataset_metadata(dataset):
            metadata = {
                "original_size": int(getattr(dataset, "original_size", len(dataset))),
                "selected_size": int(len(dataset)),
                "subset_strategy": getattr(dataset, "subset_strategy", "full"),
                "shape_counts": _shape_counts(dataset.shape_type),
                "original_shape_counts": getattr(dataset, "original_shape_counts", _shape_counts(dataset.shape_type)),
            }
            if metadata["selected_size"] != metadata["original_size"]:
                metadata["selected_indices"] = getattr(dataset, "selected_indices", np.arange(len(dataset))).tolist()
            return metadata


        def evaluate_model_set(model_specs, dataset, normalizer, output_cache_dir, generative_samples=10, inference_runs=10, latency_num_conditions=3, latency_subset_strategy="stratified", cache_full_generative_samples=True, seed=42):
            robot = ThreeLinkManipulator()
            overall_summary = {}
            per_shape_summary = {}
            overlay_payload = {}
            model_metadata = {}

            for model_name, spec in model_specs.items():
                model_latency_ms, latency_metadata = _benchmark_model_latency(
                    spec,
                    dataset,
                    normalizer,
                    seed=seed,
                    subset_strategy=latency_subset_strategy,
                    n_conditions=latency_num_conditions,
                    n_runs=inference_runs,
                )
                rows = []
                shape_rows = defaultdict(list)
                cached_predictions = []
                cached_oracle_predictions = []
                cached_references = []
                cached_shapes = []
                cached_waypoints = []

                for index in range(len(dataset)):
                    sample = _extract_sample(dataset, normalizer, index=index, uses_raw_waypoints=spec.get("uses_raw_waypoints", False))
                    target = sample["target"]
                    waypoints = sample["waypoints"]
                    shape = sample["shape"]
                    model_input = sample["model_input"]

                    if spec["kind"] == "generative":
                        predictions_array = spec["predict"](model_input, num_samples=generative_samples)
                        predictions = normalizer.inverse_transform(predictions_array, "q_sequence") if spec.get("outputs_normalized", True) else predictions_array
                        predictions = np.asarray(predictions)
                        sample_rows = [
                            _metric_row(prediction, target, robot.forward_kinematics, robot.joint_limits, dt=1.0)
                            for prediction in predictions
                        ]
                        oracle_idx = int(np.argmin([row["joint_rmse"] for row in sample_rows]))
                        oracle_row = dict(sample_rows[oracle_idx])
                        mean_row = _aggregate_rows(sample_rows)
                        oracle_row["diversity_score"] = diversity_score(predictions)
                        oracle_row["inference_time_ms"] = model_latency_ms
                        oracle_row["oracle_joint_rmse"] = oracle_row["joint_rmse"]
                        oracle_row["mean_joint_rmse"] = mean_row["joint_rmse"]
                        row = oracle_row
                        best_prediction = predictions[oracle_idx]
                        cached_prediction = predictions if cache_full_generative_samples else best_prediction
                    else:
                        prediction_array = spec["predict"](model_input, num_samples=1)[0]
                        prediction = normalizer.inverse_transform(prediction_array, "q_sequence") if spec.get("outputs_normalized", True) else prediction_array
                        row = _metric_row(prediction, target, robot.forward_kinematics, robot.joint_limits, dt=1.0)
                        row["diversity_score"] = 0.0
                        row["inference_time_ms"] = model_latency_ms
                        row["oracle_joint_rmse"] = row["joint_rmse"]
                        row["mean_joint_rmse"] = row["joint_rmse"]
                        best_prediction = prediction
                        cached_prediction = prediction

                    rows.append(row)
                    shape_rows[shape].append(row)
                    cached_predictions.append(cached_prediction)
                    cached_oracle_predictions.append(best_prediction)
                    cached_references.append(target)
                    cached_shapes.append(shape)
                    cached_waypoints.append(waypoints)

                    if index == 0:
                        overlay_payload[model_name] = {
                            "condition": waypoints.tolist(),
                            "target": target.tolist(),
                            "prediction": best_prediction.tolist(),
                        }

                overall_summary[model_name] = _aggregate_rows(rows)
                per_shape_summary[model_name] = {shape: _aggregate_rows(shape_metric_rows) for shape, shape_metric_rows in shape_rows.items()}
                model_metadata[model_name] = {
                    "kind": spec["kind"],
                    "latency_benchmark": latency_metadata,
                }

                output_cache_dir.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(
                    output_cache_dir / f"{model_name}.npz",
                    q_predicted=np.asarray(cached_predictions, dtype=object if spec["kind"] == "generative" and cache_full_generative_samples else np.float32),
                    q_oracle_predicted=np.asarray(cached_oracle_predictions),
                    q_reference=np.asarray(cached_references),
                    shape_type=np.asarray(cached_shapes),
                    waypoints=np.asarray(cached_waypoints),
                )

            return {
                "overall": overall_summary,
                "per_shape": per_shape_summary,
                "overlay": overlay_payload,
                "num_test_samples": len(dataset),
                "dataset": _dataset_metadata(dataset),
                "model_metadata": model_metadata,
            }


        def write_summary_files(summary, summary_json_path, summary_md_path):
            summary_json_path = Path(summary_json_path)
            summary_md_path = Path(summary_md_path)
            summary_json_path.parent.mkdir(parents=True, exist_ok=True)
            summary_md_path.parent.mkdir(parents=True, exist_ok=True)
            with summary_json_path.open("w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2)
            headers = [
                "Model",
                "Joint RMSE",
                "EE Pos",
                "EE Orient",
                "Path",
                "Energy",
                "Smoothness",
                "Inference (ms)",
                "Diversity",
            ]
            lines = ["# Evaluation Summary", ""]
            dataset_metadata = summary.get("dataset", {})
            if dataset_metadata:
                lines.append(f"- Test samples evaluated: {summary.get('num_test_samples', 0)}")
                lines.append(f"- Subset strategy: {dataset_metadata.get('subset_strategy', 'full')}")
                shape_counts = dataset_metadata.get("shape_counts", {})
                if shape_counts:
                    rendered_counts = ", ".join(f"{shape}={count}" for shape, count in shape_counts.items())
                    lines.append(f"- Shape counts: {rendered_counts}")
                lines.append("")
            lines.extend(
                [
                    "| " + " | ".join(headers) + " |",
                    "| " + " | ".join(["---"] * len(headers)) + " |",
                ]
            )
            for model_name, metrics in summary["overall"].items():
                diversity = metrics["diversity_score"] if metrics["diversity_score"] > 0 else "N/A"
                lines.append(
                    "| "
                    + " | ".join(
                        [
                            model_name,
                            f"{metrics['joint_rmse']:.4f}",
                            f"{metrics['ee_position_error']:.4f}",
                            f"{metrics['ee_orientation_error']:.4f}",
                            f"{metrics['path_tracking_error']:.4f}",
                            f"{metrics['energy_proxy']:.4f}",
                            f"{metrics['smoothness_jerk']:.4f}",
                            f"{metrics['inference_time_ms']:.2f}",
                            f"{diversity:.4f}" if diversity != "N/A" else diversity,
                        ]
                    )
                    + " |"
                )
            lines.extend(["", "## Per-Shape Breakdown", ""])
            for model_name, shapes in summary["per_shape"].items():
                lines.append(f"### {model_name}")
                lines.append("")
                lines.append("| Shape | Joint RMSE | EE Pos | Path |")
                lines.append("| --- | --- | --- | --- |")
                for shape_name, metrics in shapes.items():
                    lines.append(
                        f"| {shape_name} | {metrics['joint_rmse']:.4f} | {metrics['ee_position_error']:.4f} | {metrics['path_tracking_error']:.4f} |"
                    )
                lines.append("")
            summary_md_path.write_text("\\n".join(lines), encoding="utf-8")


        def plot_main_comparison(summary, figure_path):
            figure_path = Path(figure_path)
            figure_path.parent.mkdir(parents=True, exist_ok=True)
            model_names = list(summary["overall"].keys())
            joint_vals = [summary["overall"][name]["joint_rmse"] for name in model_names]
            ee_vals = [summary["overall"][name]["ee_position_error"] for name in model_names]
            infer_vals = [summary["overall"][name]["inference_time_ms"] for name in model_names]
            fig, axes = plt.subplots(1, 3, figsize=(16, 5))
            axes[0].bar(model_names, joint_vals, color="#c0392b")
            axes[0].set_title("Joint RMSE")
            axes[0].tick_params(axis="x", rotation=35)
            axes[0].grid(True, alpha=0.3, axis="y")
            axes[1].bar(model_names, ee_vals, color="#1f618d")
            axes[1].set_title("EE Position Error")
            axes[1].tick_params(axis="x", rotation=35)
            axes[1].grid(True, alpha=0.3, axis="y")
            axes[2].bar(model_names, infer_vals, color="#117a65")
            axes[2].set_title("Inference Time (ms)")
            axes[2].tick_params(axis="x", rotation=35)
            axes[2].grid(True, alpha=0.3, axis="y")
            plt.suptitle("Model Comparison Overview", fontsize=15, fontweight="bold")
            plt.tight_layout()
            plt.savefig(figure_path, dpi=300)
            plt.close()


        def plot_trajectory_overlay(summary, figure_path):
            if not summary.get("overlay"):
                return
            figure_path = Path(figure_path)
            figure_path.parent.mkdir(parents=True, exist_ok=True)
            first_model = next(iter(summary["overlay"]))
            condition = np.asarray(summary["overlay"][first_model]["condition"])
            target = np.asarray(summary["overlay"][first_model]["target"])
            robot = ThreeLinkManipulator()
            fig, ax = plt.subplots(1, 1, figsize=(8, 6))
            target_ee = np.asarray([robot.forward_kinematics(q) for q in target])
            ax.plot(condition[:, 0], condition[:, 1], "--", color="black", linewidth=1.0, label="Condition")
            ax.plot(target_ee[:, 0], target_ee[:, 1], color="#2c3e50", linewidth=2.2, label="Ground Truth")
            colors = plt.cm.tab10(np.linspace(0, 1, len(summary["overlay"])))
            for color, (model_name, payload) in zip(colors, summary["overlay"].items()):
                prediction = np.asarray(payload["prediction"])
                ee_path = np.asarray([robot.forward_kinematics(q) for q in prediction])
                ax.plot(ee_path[:, 0], ee_path[:, 1], color=color, linewidth=1.4, alpha=0.9, label=model_name)
            ax.set_xlabel("x (m)")
            ax.set_ylabel("y (m)")
            ax.set_title("Trajectory Overlay on First Test Condition")
            ax.set_aspect("equal")
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)
            plt.tight_layout()
            plt.savefig(figure_path, dpi=300)
            plt.close()


        def _overall_metric(summary, metric_name):
            models = list(summary["overall"].keys())
            values = [float(summary["overall"][model].get(metric_name, 0.0)) for model in models]
            return models, values


        def _per_shape_matrix(summary, metric_name):
            models = list(summary["per_shape"].keys())
            shapes = sorted({shape for per_model in summary["per_shape"].values() for shape in per_model.keys()})
            matrix = np.full((len(models), len(shapes)), np.nan, dtype=float)
            for row, model in enumerate(models):
                for col, shape in enumerate(shapes):
                    metrics = summary["per_shape"].get(model, {}).get(shape)
                    if metrics is not None:
                        matrix[row, col] = float(metrics.get(metric_name, np.nan))
            return models, shapes, matrix


        def plot_metric_grid(summary, output_path):
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            metrics = [
                ("joint_rmse", "Joint RMSE", "#b03a2e", False),
                ("ee_position_error", "EE Position Error", "#2874a6", False),
                ("smoothness_jerk", "Smoothness Jerk", "#af7ac5", True),
                ("inference_time_ms", "Inference Time (ms)", "#117864", True),
            ]
            fig, axes = plt.subplots(2, 2, figsize=(14, 9))
            axes = axes.ravel()
            for ax, (metric_name, title, color, log_scale) in zip(axes, metrics):
                models, values = _overall_metric(summary, metric_name)
                ax.bar(models, values, color=color, alpha=0.88)
                if log_scale and any(value > 0 for value in values):
                    ax.set_yscale("log")
                ax.set_title(title, fontweight="bold")
                ax.tick_params(axis="x", rotation=30)
                ax.grid(True, alpha=0.25, axis="y")
            dataset_meta = summary.get("dataset", {})
            sample_count = summary.get("num_test_samples", 0)
            subset_strategy = dataset_meta.get("subset_strategy", "full")
            plt.suptitle(f"Evaluation Overview ({sample_count} samples, {subset_strategy} subset)", fontsize=15, fontweight="bold")
            plt.tight_layout()
            plt.savefig(output_path, dpi=300)
            plt.close()


        def plot_accuracy_latency_tradeoff(summary, output_path):
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            models = list(summary["overall"].keys())
            latencies = [float(summary["overall"][model]["inference_time_ms"]) for model in models]
            rmse = [float(summary["overall"][model]["joint_rmse"]) for model in models]
            diversity = [float(summary["overall"][model].get("diversity_score", 0.0)) for model in models]
            sizes = [120 + 380 * value for value in diversity]
            colors = plt.cm.plasma(np.linspace(0.15, 0.9, len(models)))
            fig, ax = plt.subplots(figsize=(10, 7))
            for color, model, latency, score, size in zip(colors, models, latencies, rmse, sizes):
                ax.scatter(latency, score, s=size, color=color, alpha=0.8, edgecolors="black", linewidths=0.6)
                ax.annotate(model, (latency, score), textcoords="offset points", xytext=(6, 6), fontsize=9)
            if any(latency > 0 for latency in latencies):
                ax.set_xscale("log")
            ax.set_xlabel("Inference Time (ms, log scale)")
            ax.set_ylabel("Joint RMSE")
            ax.set_title("Accuracy vs Latency Trade-off", fontweight="bold")
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(output_path, dpi=300)
            plt.close()


        def plot_per_shape_heatmap(summary, metric_name, output_path):
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            models, shapes, matrix = _per_shape_matrix(summary, metric_name)
            if not shapes:
                return
            fig, ax = plt.subplots(figsize=(max(7, len(shapes) * 1.4), max(5, len(models) * 0.75)))
            masked = np.ma.masked_invalid(matrix)
            image = ax.imshow(masked, cmap="YlGnBu", aspect="auto")
            ax.set_xticks(np.arange(len(shapes)))
            ax.set_xticklabels(shapes, rotation=30, ha="right")
            ax.set_yticks(np.arange(len(models)))
            ax.set_yticklabels(models)
            ax.set_title(f"Per-Shape {metric_name}", fontweight="bold")
            for row in range(masked.shape[0]):
                for col in range(masked.shape[1]):
                    value = masked[row, col]
                    if not np.ma.is_masked(value):
                        ax.text(col, row, f"{float(value):.3f}", ha="center", va="center", fontsize=8)
            fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
            plt.tight_layout()
            plt.savefig(output_path, dpi=300)
            plt.close()


        def plot_generative_oracle_gap(summary, output_path):
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            models = []
            oracle = []
            mean = []
            for model_name, metrics in summary["overall"].items():
                oracle_rmse = float(metrics.get("oracle_joint_rmse", metrics.get("joint_rmse", 0.0)))
                mean_rmse = float(metrics.get("mean_joint_rmse", oracle_rmse))
                if mean_rmse > oracle_rmse or metrics.get("diversity_score", 0.0) > 0:
                    models.append(model_name)
                    oracle.append(oracle_rmse)
                    mean.append(mean_rmse)
            if not models:
                return
            x = np.arange(len(models))
            width = 0.38
            fig, ax = plt.subplots(figsize=(9, 6))
            ax.bar(x - width / 2, oracle, width, label="Oracle", color="#1f618d")
            ax.bar(x + width / 2, mean, width, label="Mean Sample", color="#cb4335")
            ax.set_xticks(x)
            ax.set_xticklabels(models, rotation=25, ha="right")
            ax.set_ylabel("Joint RMSE")
            ax.set_title("Generative Oracle vs Mean RMSE", fontweight="bold")
            ax.grid(True, alpha=0.25, axis="y")
            ax.legend()
            plt.tight_layout()
            plt.savefig(output_path, dpi=300)
            plt.close()


        def generate_result_plots(summary, output_dir):
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            plot_metric_grid(summary, output_dir / "metric_grid.png")
            plot_accuracy_latency_tradeoff(summary, output_dir / "accuracy_latency_tradeoff.png")
            plot_per_shape_heatmap(summary, "joint_rmse", output_dir / "per_shape_joint_rmse_heatmap.png")
            plot_generative_oracle_gap(summary, output_dir / "generative_oracle_gap.png")


        def merge_cvae_full_summary(remaining_summary, cvae_summary_path):
            with open(cvae_summary_path, "r", encoding="utf-8") as f:
                cvae_summary = json.load(f)
            merged = copy.deepcopy(remaining_summary)
            merged["overall"]["cvae"] = cvae_summary["overall"]
            merged["per_shape"]["cvae"] = cvae_summary["per_shape"]
            merged.setdefault("model_metadata", {})["cvae"] = {
                "kind": "generative",
                "source": str(cvae_summary_path),
            }
            return merged
        """
    ),
    code(
        """
        RUN_MODE = "smoke"  # change to "full" for the real Colab run
        DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

        # Set this to the folder that contains data/ and results/
        ARTIFACT_ROOT = Path.cwd()

        SMOKE_MAX_TEST_SAMPLES = 6
        FULL_MAX_TEST_SAMPLES = None
        SUBSET_STRATEGY = "stratified"

        GENERATIVE_SAMPLES = 10
        INFERENCE_RUNS = 10
        LATENCY_NUM_CONDITIONS = 3
        CACHE_FULL_GENERATIVE_SAMPLES = True
        MERGE_WITH_FULL_CVAE = True

        OUTPUT_DIR = ARTIFACT_ROOT / "standalone_remaining_eval_run"
        METRICS_DIR = OUTPUT_DIR / "metrics"
        PLOTS_DIR = OUTPUT_DIR / "plots"
        CACHE_DIR = OUTPUT_DIR / "generated_trajectories"
        for folder in [OUTPUT_DIR, METRICS_DIR, PLOTS_DIR, CACHE_DIR]:
            folder.mkdir(parents=True, exist_ok=True)

        cvae_full_summary_path = ARTIFACT_ROOT / "results" / "metrics" / "cvae" / "cvae_full_4500_evaluation_summary.json"
        data_dir = ARTIFACT_ROOT / "data" / "processed"
        metadata_dir = ARTIFACT_ROOT / "data" / "metadata"

        required_paths = [
            data_dir / "test.npz",
            metadata_dir / "normalization_stats.json",
            ARTIFACT_ROOT / "results" / "checkpoints" / "cnn" / "cnn_best.pth",
            ARTIFACT_ROOT / "results" / "checkpoints" / "cnn_lstm" / "cnn_lstm_best.pth",
            ARTIFACT_ROOT / "results" / "checkpoints" / "cnn_gru" / "cnn_gru_best.pth",
            ARTIFACT_ROOT / "results" / "checkpoints" / "diffusion" / "diffusion_best.pth",
        ]
        missing = [str(path) for path in required_paths if not path.exists()]
        if missing:
            raise FileNotFoundError("Missing required artifacts:\\n" + "\\n".join(missing))

        max_test_samples = SMOKE_MAX_TEST_SAMPLES if RUN_MODE == "smoke" else FULL_MAX_TEST_SAMPLES
        print("Run mode:", RUN_MODE)
        print("Device:", DEVICE)
        print("Artifact root:", ARTIFACT_ROOT)
        print("Max test samples:", max_test_samples)
        """
    ),
    code(
        """
        dataset, normalizer = load_test_dataset(
            data_dir=data_dir,
            metadata_dir=metadata_dir,
            max_samples=max_test_samples,
            seed=42,
            subset_strategy=SUBSET_STRATEGY,
        )

        model_specs = load_model_specs(device=DEVICE, artifact_root=ARTIFACT_ROOT)

        summary = evaluate_model_set(
            model_specs=model_specs,
            dataset=dataset,
            normalizer=normalizer,
            output_cache_dir=CACHE_DIR,
            generative_samples=GENERATIVE_SAMPLES,
            inference_runs=INFERENCE_RUNS,
            latency_num_conditions=LATENCY_NUM_CONDITIONS,
            latency_subset_strategy=SUBSET_STRATEGY,
            cache_full_generative_samples=CACHE_FULL_GENERATIVE_SAMPLES,
            seed=42,
        )

        write_summary_files(
            summary,
            METRICS_DIR / "remaining_models_summary.json",
            METRICS_DIR / "remaining_models_summary.md",
        )
        plot_main_comparison(summary, PLOTS_DIR / "remaining_models_main_comparison.png")
        plot_trajectory_overlay(summary, PLOTS_DIR / "remaining_models_overlay.png")
        generate_result_plots(summary, PLOTS_DIR)

        print("Evaluated models:", list(summary["overall"].keys()))
        print("Samples evaluated:", summary["num_test_samples"])
        print("Summary JSON:", METRICS_DIR / "remaining_models_summary.json")
        """
    ),
    code(
        """
        merged_summary = None
        if RUN_MODE == "full" and MERGE_WITH_FULL_CVAE and cvae_full_summary_path.exists():
            merged_summary = merge_cvae_full_summary(summary, cvae_full_summary_path)
            merged_metrics_dir = METRICS_DIR / "merged_with_cvae"
            merged_plots_dir = PLOTS_DIR / "merged_with_cvae"
            merged_metrics_dir.mkdir(parents=True, exist_ok=True)
            merged_plots_dir.mkdir(parents=True, exist_ok=True)

            merged_json = merged_metrics_dir / "full_cross_model_summary.json"
            merged_md = merged_metrics_dir / "full_cross_model_summary.md"
            write_summary_files(merged_summary, merged_json, merged_md)
            plot_main_comparison(merged_summary, merged_plots_dir / "main_comparison.png")
            plot_trajectory_overlay(merged_summary, merged_plots_dir / "trajectory_overlay.png")
            generate_result_plots(merged_summary, merged_plots_dir)

            print("Merged summary written to:", merged_json)
        else:
            print("Merged cVAE comparison skipped.")
        """
    ),
    code(
        """
        summary["overall"]
        """
    ),
]


notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "version": "3.10",
        },
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}


output_path = Path("notebooks/standalone_remaining_eval_notebook.ipynb")
output_path.parent.mkdir(parents=True, exist_ok=True)
output_path.write_text(json.dumps(notebook, indent=2), encoding="utf-8")
print(f"Wrote {output_path}")
