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
        # Standalone cVAE Full Pipeline Notebook

        This notebook is intentionally **self-contained**. It does **not import anything from this repo**.

        It includes:

        - dataset generation for the 3-link planar manipulator
        - analytical IK based trajectory supervision
        - cVAE model definition
        - training with KL annealing
        - full 4,500-sample test evaluation with `K=10` samples per condition
        - metrics, cached predictions, figures, and summary files

        It is designed for **Google Colab / another GPU machine**, and it writes all artifacts under one output folder.
        """
    ),
    code(
        """
        %pip install -q numpy scipy pandas matplotlib seaborn scikit-learn tqdm
        """
    ),
    code(
        """
        import json
        import math
        import os
        import random
        import shutil
        import time
        import zipfile
        from collections import defaultdict
        from pathlib import Path

        import matplotlib.pyplot as plt
        import numpy as np
        import pandas as pd
        import seaborn as sns
        import torch
        import torch.nn as nn
        from scipy.interpolate import CubicSpline
        from sklearn.manifold import TSNE
        from torch.amp import GradScaler, autocast
        from torch.utils.data import DataLoader, Dataset
        from tqdm.auto import tqdm

        sns.set_theme(style="whitegrid")

        CONFIG = {
            "seed": 42,
            "output_dir": "standalone_cvae_full_run",
            "force_regenerate_data": False,
            "force_retrain": False,
            "force_reevaluate": False,
            "num_waypoints": 100,
            "link_lengths": [1.0, 1.0, 0.5],
            "joint_limits": [[-math.pi, math.pi], [-math.pi, math.pi], [-math.pi, math.pi]],
            "samples_per_shape": {
                "circle": 5000,
                "square": 5000,
                "pentagon": 5000,
                "line": 5000,
                "random_smooth": 5000,
            },
            "test_only_shapes": {
                "hexagon": 2000,
            },
            "split_ratios": {
                "train": 0.8,
                "val": 0.1,
                "test": 0.1,
            },
            "shape_params": {
                "circle": {
                    "center": [1.5, 0.0],
                    "radius": [0.15, 0.4],
                    "theta_orient": [-0.5, 0.5],
                },
                "square": {
                    "center": [1.5, 0.0],
                    "side_length": [0.2, 0.6],
                    "theta_orient": [-0.5, 0.5],
                },
                "pentagon": {
                    "center": [1.5, 0.0],
                    "radius": [0.15, 0.35],
                    "theta_orient": [-0.5, 0.5],
                },
                "hexagon": {
                    "center": [1.5, 0.0],
                    "radius": [0.15, 0.35],
                    "theta_orient": [-0.5, 0.5],
                },
                "line": {
                    "x_range": [1.0, 2.0],
                    "y_range": [-0.5, 0.5],
                    "theta_orient": [-0.5, 0.5],
                },
                "random_smooth": {
                    "workspace_center": [1.5, 0.0],
                    "workspace_radius": 0.35,
                    "num_control_points": [4, 8],
                    "theta_orient": [-0.5, 0.5],
                },
            },
            "train": {
                "batch_size": 64,
                "epochs": 500,
                "lr": 3e-4,
                "weight_decay": 0.0,
                "early_stop_patience": 60,
                "grad_clip": 1.0,
                "num_workers": 0,
                "use_amp": True,
                "scheduler_tmax": 500,
            },
            "model": {
                "latent_dim": 16,
                "condition_dim": 64,
                "cond_conv_channels": [32, 64],
                "cond_kernel_sizes": [5, 3],
                "enc_hidden": 128,
                "enc_layers": 2,
                "enc_dropout": 0.1,
                "dec_hidden": 128,
                "dec_layers": 2,
                "dec_dropout": 0.1,
                "dec_fc_dims": [64],
            },
            "loss": {
                "recon_weight": 1.0,
                "kl_weight": 0.01,
                "energy_weight": 0.001,
                "smoothness_weight": 0.001,
                "endpoint_weight": 2.0,
                "kl_annealing": "cyclical",
                "kl_warmup_epochs": 50,
                "kl_cycle_epochs": 20,
                "free_bits_lambda": 0.1,
            },
            "evaluation": {
                "generative_samples": 10,
                "batch_size": 32,
                "latency_num_conditions": 3,
                "latency_runs": 10,
                "max_tsne_points": 1500,
            },
        }

        def refresh_output_paths():
            global OUTPUT_DIR, DATA_DIR, FIG_DIR, METRICS_DIR, CKPT_DIR, CACHE_DIR
            OUTPUT_DIR = Path(CONFIG["output_dir"])
            DATA_DIR = OUTPUT_DIR / "data"
            FIG_DIR = OUTPUT_DIR / "figures"
            METRICS_DIR = OUTPUT_DIR / "metrics"
            CKPT_DIR = OUTPUT_DIR / "checkpoints"
            CACHE_DIR = OUTPUT_DIR / "generated_trajectories"
            for folder in [OUTPUT_DIR, DATA_DIR, FIG_DIR, METRICS_DIR, CKPT_DIR, CACHE_DIR]:
                folder.mkdir(parents=True, exist_ok=True)

        refresh_output_paths()

        DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {DEVICE}")
        """
    ),
    code(
        """
        def seed_everything(seed: int) -> None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False


        def maybe_sync():
            if torch.cuda.is_available():
                torch.cuda.synchronize()


        def wrap_angle(angle):
            return (angle + np.pi) % (2 * np.pi) - np.pi


        class ThreeLinkManipulator:
            def __init__(self, link_lengths, joint_limits):
                self.link_lengths = np.asarray(link_lengths, dtype=np.float32)
                self.joint_limits = np.asarray(joint_limits, dtype=np.float32)
                self.base_position = np.array([0.0, 0.0], dtype=np.float32)

            def check_joint_limits(self, q):
                q = np.asarray(q)
                return np.all((q >= self.joint_limits[:, 0]) & (q <= self.joint_limits[:, 1]))

            def forward_kinematics(self, q):
                l1, l2, l3 = self.link_lengths
                q1, q2, q3 = q
                x = l1 * np.cos(q1) + l2 * np.cos(q1 + q2) + l3 * np.cos(q1 + q2 + q3)
                y = l1 * np.sin(q1) + l2 * np.sin(q1 + q2) + l3 * np.sin(q1 + q2 + q3)
                theta = q1 + q2 + q3
                return np.array([x, y, theta], dtype=np.float32)

            def forward_kinematics_batch(self, q):
                q = np.asarray(q)
                l1, l2, l3 = self.link_lengths
                q1 = q[..., 0]
                q2 = q[..., 1]
                q3 = q[..., 2]
                x = l1 * np.cos(q1) + l2 * np.cos(q1 + q2) + l3 * np.cos(q1 + q2 + q3)
                y = l1 * np.sin(q1) + l2 * np.sin(q1 + q2) + l3 * np.sin(q1 + q2 + q3)
                theta = q1 + q2 + q3
                return np.stack([x, y, theta], axis=-1).astype(np.float32)

            def inverse_kinematics(self, pose, elbow="both"):
                x, y, theta = pose
                l1, l2, l3 = self.link_lengths
                wx = x - l3 * np.cos(theta)
                wy = y - l3 * np.sin(theta)
                dx = wx - self.base_position[0]
                dy = wy - self.base_position[1]
                D = (dx ** 2 + dy ** 2 - l1 ** 2 - l2 ** 2) / (2 * l1 * l2)
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


        class TrajectoryGenerator:
            def __init__(self, robot, num_waypoints=100):
                self.robot = robot
                self.num_waypoints = num_waypoints

            def circle(self, center=(1.5, 0.0), radius=0.3, theta_orient=0.0):
                t = np.linspace(0, 2 * np.pi, self.num_waypoints, endpoint=False)
                x = center[0] + radius * np.cos(t)
                y = center[1] + radius * np.sin(t)
                theta = np.full_like(t, theta_orient)
                return np.column_stack([x, y, theta]).astype(np.float32)

            def square(self, center=(1.5, 0.0), side_length=0.4, theta_orient=0.0):
                half = side_length / 2.0
                corners = np.array([
                    [center[0] - half, center[1] - half],
                    [center[0] + half, center[1] - half],
                    [center[0] + half, center[1] + half],
                    [center[0] - half, center[1] + half],
                ], dtype=np.float32)
                return self._polygon_waypoints(corners, theta_orient)

            def pentagon(self, center=(1.5, 0.0), radius=0.3, theta_orient=0.0):
                corners = self._regular_polygon_corners(center, radius, 5)
                return self._polygon_waypoints(corners, theta_orient)

            def hexagon(self, center=(1.5, 0.0), radius=0.3, theta_orient=0.0):
                corners = self._regular_polygon_corners(center, radius, 6)
                return self._polygon_waypoints(corners, theta_orient)

            def line(self, start=(1.2, -0.3), end=(1.8, 0.3), theta_orient=0.0):
                t = np.linspace(0, 1, self.num_waypoints)
                x = start[0] + t * (end[0] - start[0])
                y = start[1] + t * (end[1] - start[1])
                theta = np.full_like(t, theta_orient)
                return np.column_stack([x, y, theta]).astype(np.float32)

            def random_smooth(self, num_control_points=6, workspace_center=(1.5, 0.0), workspace_radius=0.4, theta_orient=0.0, rng=None):
                if rng is None:
                    rng = np.random.default_rng()
                angles = np.sort(rng.uniform(0, 2 * np.pi, num_control_points))
                radii = rng.uniform(0.1, workspace_radius, num_control_points)
                cx = workspace_center[0] + radii * np.cos(angles)
                cy = workspace_center[1] + radii * np.sin(angles)
                cx = np.append(cx, cx[0])
                cy = np.append(cy, cy[0])
                param = np.linspace(0, 1, len(cx))
                cs_x = CubicSpline(param, cx, bc_type="periodic")
                cs_y = CubicSpline(param, cy, bc_type="periodic")
                t_eval = np.linspace(0, 1, self.num_waypoints, endpoint=False)
                x = cs_x(t_eval)
                y = cs_y(t_eval)
                theta = np.full_like(x, theta_orient)
                return np.column_stack([x, y, theta]).astype(np.float32)

            def resolve_joint_trajectory(self, waypoints):
                q_seq = np.zeros((len(waypoints), 3), dtype=np.float32)
                for i, pose in enumerate(waypoints):
                    solutions = self.robot.inverse_kinematics(pose, elbow="both")
                    if len(solutions) == 0:
                        return None
                    if i == 0:
                        energies = [np.sum(s ** 2) for s in solutions]
                        q_seq[0] = solutions[int(np.argmin(energies))]
                    else:
                        distances = [np.linalg.norm(s - q_seq[i - 1]) for s in solutions]
                        q_seq[i] = solutions[int(np.argmin(distances))]
                return q_seq

            def generate(self, shape_type, rng=None, **kwargs):
                generators = {
                    "circle": self.circle,
                    "square": self.square,
                    "pentagon": self.pentagon,
                    "hexagon": self.hexagon,
                    "line": self.line,
                    "random_smooth": self.random_smooth,
                }
                if rng is not None and shape_type == "random_smooth":
                    kwargs["rng"] = rng
                waypoints = generators[shape_type](**kwargs)
                q_sequence = self.resolve_joint_trajectory(waypoints)
                if q_sequence is None:
                    return None
                return {
                    "shape_type": shape_type,
                    "waypoints": waypoints.astype(np.float32),
                    "q_sequence": q_sequence.astype(np.float32),
                }

            def _regular_polygon_corners(self, center, radius, n_sides):
                angles = np.linspace(0, 2 * np.pi, n_sides, endpoint=False)
                cx = center[0] + radius * np.cos(angles)
                cy = center[1] + radius * np.sin(angles)
                return np.column_stack([cx, cy]).astype(np.float32)

            def _polygon_waypoints(self, corners, theta_orient):
                closed = np.vstack([corners, corners[0:1]])
                edge_lengths = np.linalg.norm(np.diff(closed, axis=0), axis=1)
                cum_lengths = np.concatenate([[0.0], np.cumsum(edge_lengths)])
                total_length = cum_lengths[-1]
                t_samples = np.linspace(0, total_length, self.num_waypoints, endpoint=False)
                x = np.interp(t_samples, cum_lengths, closed[:, 0])
                y = np.interp(t_samples, cum_lengths, closed[:, 1])
                theta = np.full_like(x, theta_orient)
                return np.column_stack([x, y, theta]).astype(np.float32)


        def sample_shape_kwargs(shape_type, cfg, rng):
            params = cfg["shape_params"][shape_type]
            if shape_type == "circle":
                return {
                    "center": tuple(params["center"]),
                    "radius": float(rng.uniform(*params["radius"])),
                    "theta_orient": float(rng.uniform(*params["theta_orient"])),
                }
            if shape_type == "square":
                return {
                    "center": tuple(params["center"]),
                    "side_length": float(rng.uniform(*params["side_length"])),
                    "theta_orient": float(rng.uniform(*params["theta_orient"])),
                }
            if shape_type in {"pentagon", "hexagon"}:
                return {
                    "center": tuple(params["center"]),
                    "radius": float(rng.uniform(*params["radius"])),
                    "theta_orient": float(rng.uniform(*params["theta_orient"])),
                }
            if shape_type == "line":
                start_x = float(rng.uniform(*params["x_range"]))
                end_x = float(rng.uniform(*params["x_range"]))
                start_y = float(rng.uniform(*params["y_range"]))
                end_y = float(rng.uniform(*params["y_range"]))
                return {
                    "start": (start_x, start_y),
                    "end": (end_x, end_y),
                    "theta_orient": float(rng.uniform(*params["theta_orient"])),
                }
            if shape_type == "random_smooth":
                return {
                    "num_control_points": int(rng.integers(params["num_control_points"][0], params["num_control_points"][1] + 1)),
                    "workspace_center": tuple(params["workspace_center"]),
                    "workspace_radius": float(params["workspace_radius"]),
                    "theta_orient": float(rng.uniform(*params["theta_orient"])),
                }
            raise ValueError(f"Unknown shape type: {shape_type}")


        def stack_records(records):
            return {
                "waypoints": np.stack([rec["waypoints"] for rec in records]).astype(np.float32),
                "q_sequence": np.stack([rec["q_sequence"] for rec in records]).astype(np.float32),
                "shape_type": np.array([rec["shape_type"] for rec in records]),
            }


        def fit_normalizers(train_split):
            stats = {}
            for key in ["waypoints", "q_sequence"]:
                mean = train_split[key].mean(axis=(0, 1), keepdims=True)
                std = train_split[key].std(axis=(0, 1), keepdims=True)
                std = np.where(std < 1e-6, 1.0, std)
                stats[key] = {
                    "mean": mean.astype(np.float32),
                    "std": std.astype(np.float32),
                }
            return stats


        def normalize_array(x, stats, key):
            return ((x - stats[key]["mean"]) / stats[key]["std"]).astype(np.float32)


        def inverse_normalize_array(x, stats, key):
            return (x * stats[key]["std"] + stats[key]["mean"]).astype(np.float32)


        class TrajectoryDataset(Dataset):
            def __init__(self, split_dict, stats):
                self.waypoints = normalize_array(split_dict["waypoints"], stats, "waypoints")
                self.q_sequence = normalize_array(split_dict["q_sequence"], stats, "q_sequence")
                self.shape_type = split_dict["shape_type"]

            def __len__(self):
                return len(self.waypoints)

            def __getitem__(self, idx):
                return (
                    torch.from_numpy(self.waypoints[idx]),
                    torch.from_numpy(self.q_sequence[idx]),
                )


        class ConditionEncoder(nn.Module):
            def __init__(self, input_dim=3, conv_channels=None, kernel_sizes=None, condition_dim=64, dropout=0.1):
                super().__init__()
                conv_channels = conv_channels or [32, 64]
                kernel_sizes = kernel_sizes or [5, 3]
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
                x = waypoints.permute(0, 2, 1)
                x = self.conv(x)
                x = self.pool(x).squeeze(-1)
                return self.proj(x)


        class BiGRUEncoder(nn.Module):
            def __init__(self, input_dim=3, condition_dim=64, hidden=128, layers=2, dropout=0.1, latent_dim=16):
                super().__init__()
                self.cond_proj = nn.Linear(condition_dim, input_dim)
                self.gru = nn.GRU(
                    input_size=input_dim * 2,
                    hidden_size=hidden,
                    num_layers=layers,
                    batch_first=True,
                    bidirectional=True,
                    dropout=dropout if layers > 1 else 0.0,
                )
                self.mu_proj = nn.Linear(hidden * 2, latent_dim)
                self.logvar_proj = nn.Linear(hidden * 2, latent_dim)

            def forward(self, trajectory, cond):
                batch_size, steps, _ = trajectory.shape
                cond_expanded = self.cond_proj(cond).unsqueeze(1).expand(batch_size, steps, -1)
                x = torch.cat([trajectory, cond_expanded], dim=-1)
                _, h = self.gru(x)
                h_fwd = h[-2]
                h_bwd = h[-1]
                h_cat = torch.cat([h_fwd, h_bwd], dim=-1)
                return self.mu_proj(h_cat), self.logvar_proj(h_cat)


        class GRUDecoder(nn.Module):
            def __init__(self, latent_dim=16, condition_dim=64, output_dim=3, output_steps=100, hidden=128, layers=2, dropout=0.1, fc_dims=None):
                super().__init__()
                self.output_steps = output_steps
                fc_dims = fc_dims or [64]
                self.input_proj = nn.Linear(latent_dim + condition_dim, hidden)
                self.gru = nn.GRU(
                    input_size=hidden,
                    hidden_size=hidden,
                    num_layers=layers,
                    batch_first=True,
                    dropout=dropout if layers > 1 else 0.0,
                )
                out_layers = []
                fc_in = hidden
                for dim in fc_dims:
                    out_layers.extend([nn.Linear(fc_in, dim), nn.ReLU(inplace=True), nn.Dropout(dropout)])
                    fc_in = dim
                out_layers.append(nn.Linear(fc_in, output_dim))
                self.output_proj = nn.Sequential(*out_layers)

            def forward(self, z, cond):
                batch_size = z.shape[0]
                zc = torch.cat([z, cond], dim=-1)
                inp = self.input_proj(zc).unsqueeze(1).expand(batch_size, self.output_steps, -1)
                out, _ = self.gru(inp)
                return self.output_proj(out)


        class ConditionalVAE(nn.Module):
            def __init__(self, cfg):
                super().__init__()
                mcfg = cfg["model"]
                self.latent_dim = mcfg["latent_dim"]
                self.cond_encoder = ConditionEncoder(
                    input_dim=3,
                    conv_channels=mcfg["cond_conv_channels"],
                    kernel_sizes=mcfg["cond_kernel_sizes"],
                    condition_dim=mcfg["condition_dim"],
                    dropout=mcfg["enc_dropout"],
                )
                self.encoder = BiGRUEncoder(
                    input_dim=3,
                    condition_dim=mcfg["condition_dim"],
                    hidden=mcfg["enc_hidden"],
                    layers=mcfg["enc_layers"],
                    dropout=mcfg["enc_dropout"],
                    latent_dim=mcfg["latent_dim"],
                )
                self.decoder = GRUDecoder(
                    latent_dim=mcfg["latent_dim"],
                    condition_dim=mcfg["condition_dim"],
                    output_dim=3,
                    output_steps=cfg["num_waypoints"],
                    hidden=mcfg["dec_hidden"],
                    layers=mcfg["dec_layers"],
                    dropout=mcfg["dec_dropout"],
                    fc_dims=mcfg["dec_fc_dims"],
                )

            def reparameterize(self, mu, log_var):
                std = torch.exp(0.5 * log_var)
                eps = torch.randn_like(std)
                return mu + eps * std

            def forward(self, waypoints, trajectory):
                cond = self.cond_encoder(waypoints)
                mu, log_var = self.encoder(trajectory, cond)
                z = self.reparameterize(mu, log_var)
                recon = self.decoder(z, cond)
                return recon, mu, log_var

            @torch.no_grad()
            def generate(self, waypoints, num_samples=1):
                self.eval()
                cond = self.cond_encoder(waypoints)
                batch_size = cond.shape[0]
                if num_samples > 1:
                    cond = cond.unsqueeze(1).expand(batch_size, num_samples, -1).reshape(batch_size * num_samples, -1)
                z = torch.randn(cond.shape[0], self.latent_dim, device=cond.device)
                return self.decoder(z, cond)


        class VAELoss(nn.Module):
            def __init__(self, cfg):
                super().__init__()
                lcfg = cfg["loss"]
                self.recon_weight = lcfg["recon_weight"]
                self.kl_weight_target = lcfg["kl_weight"]
                self.energy_weight = lcfg["energy_weight"]
                self.smoothness_weight = lcfg["smoothness_weight"]
                self.endpoint_weight = lcfg["endpoint_weight"]
                self.kl_annealing = lcfg["kl_annealing"]
                self.kl_warmup_epochs = lcfg["kl_warmup_epochs"]
                self.kl_cycle_epochs = lcfg["kl_cycle_epochs"]
                self.free_bits_lambda = lcfg["free_bits_lambda"]
                self.current_epoch = 0

            def set_epoch(self, epoch):
                self.current_epoch = epoch

            def get_kl_weight(self):
                epoch = self.current_epoch
                if self.kl_annealing == "monotonic":
                    return self.kl_weight_target * min(1.0, epoch / max(self.kl_warmup_epochs, 1))
                if self.kl_annealing == "cyclical":
                    cycle_pos = epoch % self.kl_cycle_epochs
                    ramp = min(1.0, cycle_pos / max(self.kl_cycle_epochs * 0.5, 1))
                    warmup = min(1.0, epoch / max(self.kl_warmup_epochs, 1))
                    return self.kl_weight_target * ramp * warmup
                return self.kl_weight_target

            def recon_loss(self, pred, target):
                mse = (pred - target) ** 2
                weights = torch.ones(pred.shape[1], device=pred.device)
                weights[0] = self.endpoint_weight
                weights[-1] = self.endpoint_weight
                weights = weights / weights.mean()
                mse = mse * weights.unsqueeze(0).unsqueeze(-1)
                return mse.mean()

            def kl_divergence(self, mu, log_var):
                kl_per_dim = -0.5 * (1 + log_var - mu.pow(2) - log_var.exp())
                if self.kl_annealing == "free_bits":
                    kl_per_dim = torch.clamp(kl_per_dim, min=self.free_bits_lambda)
                return kl_per_dim.sum(dim=-1).mean()

            def forward(self, pred, target, mu, log_var):
                recon = self.recon_loss(pred, target)
                kl = self.kl_divergence(mu, log_var)
                kl_weight = self.get_kl_weight()
                vel = pred[:, 1:, :] - pred[:, :-1, :]
                energy = (vel ** 2).mean()
                acc = vel[:, 1:, :] - vel[:, :-1, :]
                jerk = acc[:, 1:, :] - acc[:, :-1, :]
                smoothness = (jerk ** 2).mean()
                total = (
                    self.recon_weight * recon
                    + kl_weight * kl
                    + self.energy_weight * energy
                    + self.smoothness_weight * smoothness
                )
                return total, {
                    "total": float(total.item()),
                    "recon": float(recon.item()),
                    "kl": float(kl.item()),
                    "kl_weight": float(kl_weight),
                    "energy": float(energy.item()),
                    "smoothness": float(smoothness.item()),
                }


        def split_shape_records(records, cfg, rng):
            indices = np.arange(len(records))
            rng.shuffle(indices)
            n_total = len(indices)
            n_train = int(n_total * cfg["split_ratios"]["train"])
            n_val = int(n_total * cfg["split_ratios"]["val"])
            train_idx = indices[:n_train]
            val_idx = indices[n_train:n_train + n_val]
            test_idx = indices[n_train + n_val:]
            return (
                [records[i] for i in train_idx],
                [records[i] for i in val_idx],
                [records[i] for i in test_idx],
            )


        def plot_dataset_distribution(split_dicts, output_path):
            rows = []
            for split_name, split in split_dicts.items():
                unique, counts = np.unique(split["shape_type"], return_counts=True)
                for shape, count in zip(unique, counts):
                    rows.append({"split": split_name, "shape_type": shape, "count": int(count)})
            df = pd.DataFrame(rows)
            plt.figure(figsize=(10, 5))
            sns.barplot(data=df, x="shape_type", y="count", hue="split")
            plt.title("Dataset Distribution by Shape and Split", fontweight="bold")
            plt.tight_layout()
            plt.savefig(output_path, dpi=300)
            plt.close()


        def plot_shape_grid(example_records, output_path):
            shapes = list(example_records.keys())
            fig, axes = plt.subplots(2, 3, figsize=(12, 7))
            axes = axes.ravel()
            for ax, shape in zip(axes, shapes):
                waypoints = example_records[shape]["waypoints"]
                ax.plot(waypoints[:, 0], waypoints[:, 1], linewidth=2.0)
                ax.set_title(shape)
                ax.set_aspect("equal")
                ax.grid(True, alpha=0.25)
            plt.suptitle("Task-Space Trajectory Shapes", fontweight="bold")
            plt.tight_layout()
            plt.savefig(output_path, dpi=300)
            plt.close()


        seed_everything(CONFIG["seed"])
        robot = ThreeLinkManipulator(CONFIG["link_lengths"], CONFIG["joint_limits"])
        generator = TrajectoryGenerator(robot, num_waypoints=CONFIG["num_waypoints"])
        """
    ),
    code(
        """
        def generate_or_load_dataset(cfg, generator, output_dir):
            dataset_path = output_dir / "dataset_splits.npz"
            stats_path = output_dir / "normalization_stats.npz"

            if dataset_path.exists() and stats_path.exists() and not cfg["force_regenerate_data"]:
                loaded = np.load(dataset_path, allow_pickle=True)
                splits = {
                    "train": {
                        "waypoints": loaded["train_waypoints"],
                        "q_sequence": loaded["train_q_sequence"],
                        "shape_type": loaded["train_shape_type"],
                    },
                    "val": {
                        "waypoints": loaded["val_waypoints"],
                        "q_sequence": loaded["val_q_sequence"],
                        "shape_type": loaded["val_shape_type"],
                    },
                    "test": {
                        "waypoints": loaded["test_waypoints"],
                        "q_sequence": loaded["test_q_sequence"],
                        "shape_type": loaded["test_shape_type"],
                    },
                }
                stats_npz = np.load(stats_path)
                stats = {
                    "waypoints": {
                        "mean": stats_npz["waypoints_mean"],
                        "std": stats_npz["waypoints_std"],
                    },
                    "q_sequence": {
                        "mean": stats_npz["q_sequence_mean"],
                        "std": stats_npz["q_sequence_std"],
                    },
                }
                print("Loaded cached dataset.")
                return splits, stats

            rng = np.random.default_rng(cfg["seed"])
            train_records = []
            val_records = []
            test_records = []
            example_records = {}

            for shape_type, n_samples in cfg["samples_per_shape"].items():
                shape_records = []
                pbar = tqdm(total=n_samples, desc=f"Generating {shape_type}", leave=False)
                while len(shape_records) < n_samples:
                    kwargs = sample_shape_kwargs(shape_type, cfg, rng)
                    sample = generator.generate(shape_type, rng=rng, **kwargs)
                    if sample is not None:
                        shape_records.append(sample)
                        if shape_type not in example_records:
                            example_records[shape_type] = sample
                        pbar.update(1)
                pbar.close()
                tr, va, te = split_shape_records(shape_records, cfg, rng)
                train_records.extend(tr)
                val_records.extend(va)
                test_records.extend(te)

            for shape_type, n_samples in cfg["test_only_shapes"].items():
                pbar = tqdm(total=n_samples, desc=f"Generating {shape_type} (test only)", leave=False)
                while n_samples > 0:
                    kwargs = sample_shape_kwargs(shape_type, cfg, rng)
                    sample = generator.generate(shape_type, rng=rng, **kwargs)
                    if sample is not None:
                        test_records.append(sample)
                        if shape_type not in example_records:
                            example_records[shape_type] = sample
                        n_samples -= 1
                        pbar.update(1)
                pbar.close()

            rng.shuffle(train_records)
            rng.shuffle(val_records)
            rng.shuffle(test_records)

            splits = {
                "train": stack_records(train_records),
                "val": stack_records(val_records),
                "test": stack_records(test_records),
            }
            stats = fit_normalizers(splits["train"])

            np.savez_compressed(
                dataset_path,
                train_waypoints=splits["train"]["waypoints"],
                train_q_sequence=splits["train"]["q_sequence"],
                train_shape_type=splits["train"]["shape_type"],
                val_waypoints=splits["val"]["waypoints"],
                val_q_sequence=splits["val"]["q_sequence"],
                val_shape_type=splits["val"]["shape_type"],
                test_waypoints=splits["test"]["waypoints"],
                test_q_sequence=splits["test"]["q_sequence"],
                test_shape_type=splits["test"]["shape_type"],
            )
            np.savez_compressed(
                stats_path,
                waypoints_mean=stats["waypoints"]["mean"],
                waypoints_std=stats["waypoints"]["std"],
                q_sequence_mean=stats["q_sequence"]["mean"],
                q_sequence_std=stats["q_sequence"]["std"],
            )

            plot_dataset_distribution(splits, FIG_DIR / "dataset_distribution.png")
            plot_shape_grid(example_records, FIG_DIR / "shape_trajectories_grid.png")
            print("Generated dataset from scratch.")
            return splits, stats


        splits, stats = generate_or_load_dataset(CONFIG, generator, DATA_DIR)
        print({name: len(split["shape_type"]) for name, split in splits.items()})
        """
    ),
    code(
        """
        train_dataset = TrajectoryDataset(splits["train"], stats)
        val_dataset = TrajectoryDataset(splits["val"], stats)
        test_dataset = TrajectoryDataset(splits["test"], stats)

        # Notebook runtimes such as Colab/Jupyter can be fragile with
        # multiprocessing DataLoader workers, especially after cell re-runs.
        # Force single-process loading here for maximum portability.
        loader_num_workers = 0
        loader_pin_memory = bool(DEVICE == "cuda")

        train_loader = DataLoader(
            train_dataset,
            batch_size=CONFIG["train"]["batch_size"],
            shuffle=True,
            num_workers=loader_num_workers,
            pin_memory=loader_pin_memory,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=CONFIG["train"]["batch_size"],
            shuffle=False,
            num_workers=loader_num_workers,
            pin_memory=loader_pin_memory,
        )

        model = ConditionalVAE(CONFIG).to(DEVICE)
        loss_fn = VAELoss(CONFIG)
        optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG["train"]["lr"], weight_decay=CONFIG["train"]["weight_decay"])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG["train"]["scheduler_tmax"])
        scaler = GradScaler("cuda", enabled=(DEVICE == "cuda" and CONFIG["train"]["use_amp"]))

        param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"cVAE parameters: {param_count:,}")


        def run_epoch(model, loader, loss_fn, optimizer=None):
            train_mode = optimizer is not None
            model.train(train_mode)
            metrics_accum = defaultdict(float)
            n_batches = 0

            for waypoints, q_target in tqdm(loader, leave=False):
                waypoints = waypoints.to(DEVICE, dtype=torch.float32, non_blocking=True)
                q_target = q_target.to(DEVICE, dtype=torch.float32, non_blocking=True)

                if train_mode:
                    optimizer.zero_grad(set_to_none=True)

                with autocast("cuda", enabled=(DEVICE == "cuda" and CONFIG["train"]["use_amp"])):
                    recon, mu, log_var = model(waypoints, q_target)
                    loss, loss_dict = loss_fn(recon, q_target, mu, log_var)

                if train_mode:
                    scaler.scale(loss).backward()
                    if CONFIG["train"]["grad_clip"] is not None:
                        scaler.unscale_(optimizer)
                        nn.utils.clip_grad_norm_(model.parameters(), CONFIG["train"]["grad_clip"])
                    scaler.step(optimizer)
                    scaler.update()

                for key, value in loss_dict.items():
                    metrics_accum[key] += float(value)
                n_batches += 1

            return {key: value / max(n_batches, 1) for key, value in metrics_accum.items()}


        def train_or_load_model(model, loss_fn, optimizer, scheduler, train_loader, val_loader, cfg):
            ckpt_path = CKPT_DIR / "cvae_best.pt"
            history_path = METRICS_DIR / "cvae_history.json"

            if ckpt_path.exists() and history_path.exists() and not cfg["force_retrain"]:
                checkpoint = torch.load(ckpt_path, map_location=DEVICE)
                model.load_state_dict(checkpoint["model_state_dict"])
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                history = json.loads(history_path.read_text())
                print("Loaded cached trained model.")
                return history

            history = {
                "train_total": [],
                "train_recon": [],
                "train_kl": [],
                "train_energy": [],
                "train_smoothness": [],
                "val_total": [],
                "val_recon": [],
                "val_kl": [],
                "val_energy": [],
                "val_smoothness": [],
                "kl_weight": [],
            }

            best_val_recon = float("inf")
            patience_counter = 0

            for epoch in range(1, cfg["train"]["epochs"] + 1):
                loss_fn.set_epoch(epoch)
                train_metrics = run_epoch(model, train_loader, loss_fn, optimizer=optimizer)
                with torch.no_grad():
                    val_metrics = run_epoch(model, val_loader, loss_fn, optimizer=None)

                for key in ["total", "recon", "kl", "energy", "smoothness"]:
                    history[f"train_{key}"].append(train_metrics[key])
                    history[f"val_{key}"].append(val_metrics[key])
                history["kl_weight"].append(train_metrics["kl_weight"])

                scheduler.step()

                if epoch == 1 or epoch % 10 == 0:
                    print(
                        f"Epoch {epoch:03d} | "
                        f"train_total={train_metrics['total']:.6f} | "
                        f"train_recon={train_metrics['recon']:.6f} | "
                        f"val_recon={val_metrics['recon']:.6f} | "
                        f"kl_w={train_metrics['kl_weight']:.4f}"
                    )

                if val_metrics["recon"] < best_val_recon:
                    best_val_recon = val_metrics["recon"]
                    patience_counter = 0
                    torch.save(
                        {
                            "epoch": epoch,
                            "best_val_recon": best_val_recon,
                            "model_state_dict": model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                        },
                        ckpt_path,
                    )
                else:
                    patience_counter += 1

                if patience_counter >= cfg["train"]["early_stop_patience"]:
                    print(f"Early stopping at epoch {epoch}")
                    break

            checkpoint = torch.load(ckpt_path, map_location=DEVICE)
            model.load_state_dict(checkpoint["model_state_dict"])
            history_path.write_text(json.dumps(history, indent=2))
            return history


        history = train_or_load_model(model, loss_fn, optimizer, scheduler, train_loader, val_loader, CONFIG)
        """
    ),
    code(
        """
        def plot_training_curves(history, output_path):
            epochs = np.arange(1, len(history["train_total"]) + 1)
            fig, axes = plt.subplots(2, 2, figsize=(14, 9))
            axes = axes.ravel()

            axes[0].plot(epochs, history["train_total"], label="Train")
            axes[0].plot(epochs, history["val_total"], label="Val")
            axes[0].set_title("Total Loss")
            axes[0].legend()

            axes[1].plot(epochs, history["train_recon"], label="Train")
            axes[1].plot(epochs, history["val_recon"], label="Val")
            axes[1].set_title("Reconstruction Loss")
            axes[1].legend()

            axes[2].plot(epochs, history["train_kl"], label="Train KL")
            axes[2].plot(epochs, history["val_kl"], label="Val KL")
            axes[2].set_title("KL Divergence")
            axes[2].legend()

            axes[3].plot(epochs, history["kl_weight"], label="KL Weight", color="tab:red")
            axes[3].set_title("KL Annealing Weight")
            axes[3].legend()

            for ax in axes:
                ax.grid(True, alpha=0.25)
                ax.set_xlabel("Epoch")
            plt.suptitle("cVAE Training Curves", fontweight="bold")
            plt.tight_layout()
            plt.savefig(output_path, dpi=300)
            plt.close()


        @torch.no_grad()
        def plot_reconstruction_examples(model, dataset, stats, output_path, n_examples=4):
            model.eval()
            indices = np.linspace(0, len(dataset) - 1, n_examples, dtype=int)
            fig, axes = plt.subplots(n_examples, 3, figsize=(14, 3.5 * n_examples))
            if n_examples == 1:
                axes = np.expand_dims(axes, axis=0)

            for row, idx in enumerate(indices):
                wp_norm, q_norm = dataset[idx]
                wp = inverse_normalize_array(wp_norm.numpy()[None], stats, "waypoints")[0]
                q_true = inverse_normalize_array(q_norm.numpy()[None], stats, "q_sequence")[0]
                recon, _, _ = model(
                    wp_norm.unsqueeze(0).to(DEVICE, dtype=torch.float32),
                    q_norm.unsqueeze(0).to(DEVICE, dtype=torch.float32),
                )
                q_pred = inverse_normalize_array(recon.detach().cpu().numpy(), stats, "q_sequence")[0]

                ee_true = robot.forward_kinematics_batch(q_true)
                ee_pred = robot.forward_kinematics_batch(q_pred)

                axes[row, 0].plot(wp[:, 0], wp[:, 1], "--", color="black", label="Waypoints")
                axes[row, 0].plot(ee_true[:, 0], ee_true[:, 1], label="GT")
                axes[row, 0].plot(ee_pred[:, 0], ee_pred[:, 1], label="Recon")
                axes[row, 0].set_title(f"Task Space Example {row + 1}")
                axes[row, 0].set_aspect("equal")
                axes[row, 0].legend(fontsize=8)

                for joint_idx in range(3):
                    axes[row, 1].plot(q_true[:, joint_idx], label=f"GT q{joint_idx + 1}" if row == 0 else None)
                    axes[row, 1].plot(q_pred[:, joint_idx], linestyle="--", label=f"Pred q{joint_idx + 1}" if row == 0 else None)
                axes[row, 1].set_title("Joint Trajectory Reconstruction")

                error = np.linalg.norm(q_pred - q_true, axis=1)
                axes[row, 2].plot(error, color="tab:red")
                axes[row, 2].set_title("Per-Step Joint Error")

            plt.tight_layout()
            plt.savefig(output_path, dpi=300)
            plt.close()


        @torch.no_grad()
        def plot_generation_examples(model, dataset, stats, output_path, num_conditions=3, num_samples=5):
            model.eval()
            indices = np.linspace(0, len(dataset) - 1, num_conditions, dtype=int)
            fig, axes = plt.subplots(1, num_conditions, figsize=(5 * num_conditions, 4))
            if num_conditions == 1:
                axes = [axes]
            for ax, idx in zip(axes, indices):
                wp_norm, q_norm = dataset[idx]
                wp = inverse_normalize_array(wp_norm.numpy()[None], stats, "waypoints")[0]
                q_true = inverse_normalize_array(q_norm.numpy()[None], stats, "q_sequence")[0]
                samples = model.generate(
                    wp_norm.unsqueeze(0).to(DEVICE, dtype=torch.float32),
                    num_samples=num_samples,
                ).detach().cpu().numpy()
                samples = inverse_normalize_array(samples, stats, "q_sequence")
                ax.plot(wp[:, 0], wp[:, 1], "--", color="black", linewidth=1.0, label="Waypoints")
                ee_true = robot.forward_kinematics_batch(q_true)
                ax.plot(ee_true[:, 0], ee_true[:, 1], color="tab:blue", linewidth=2.0, label="GT")
                for sample_idx, sample in enumerate(samples):
                    ee = robot.forward_kinematics_batch(sample)
                    ax.plot(ee[:, 0], ee[:, 1], alpha=0.7, label="Generated" if sample_idx == 0 else None)
                ax.set_aspect("equal")
                ax.set_title(f"Condition {idx}")
                ax.legend(fontsize=8)
            plt.tight_layout()
            plt.savefig(output_path, dpi=300)
            plt.close()


        @torch.no_grad()
        def plot_latent_tsne(model, dataset, stats, output_path, max_points=1500):
            model.eval()
            n_points = min(len(dataset), max_points)
            if n_points < 3:
                return
            indices = np.linspace(0, len(dataset) - 1, n_points, dtype=int)
            latents = []
            labels = []
            for idx in tqdm(indices, desc="Collecting latents", leave=False):
                wp_norm, q_norm = dataset[idx]
                wp_t = wp_norm.unsqueeze(0).to(DEVICE, dtype=torch.float32)
                q_t = q_norm.unsqueeze(0).to(DEVICE, dtype=torch.float32)
                cond = model.cond_encoder(wp_t)
                mu, _ = model.encoder(q_t, cond)
                latents.append(mu.detach().cpu().numpy()[0])
                labels.append(dataset.shape_type[idx])
            latents = np.asarray(latents)
            perplexity = max(2, min(30, n_points - 1))
            tsne = TSNE(
                n_components=2,
                random_state=CONFIG["seed"],
                init="pca",
                learning_rate="auto",
                perplexity=perplexity,
            )
            embedded = tsne.fit_transform(latents)
            df = pd.DataFrame({"x": embedded[:, 0], "y": embedded[:, 1], "shape_type": labels})
            plt.figure(figsize=(8, 6))
            sns.scatterplot(data=df, x="x", y="y", hue="shape_type", s=24, linewidth=0, alpha=0.8)
            plt.title("Latent Space t-SNE", fontweight="bold")
            plt.tight_layout()
            plt.savefig(output_path, dpi=300)
            plt.close()


        plot_training_curves(history, FIG_DIR / "training_curves.png")
        plot_reconstruction_examples(model, val_dataset, stats, FIG_DIR / "reconstruction_samples.png")
        plot_generation_examples(model, test_dataset, stats, FIG_DIR / "generation_samples.png")
        plot_latent_tsne(model, test_dataset, stats, FIG_DIR / "latent_tsne.png", max_points=CONFIG["evaluation"]["max_tsne_points"])
        print("Saved cVAE training and qualitative figures.")
        """
    ),
    code(
        """
        def benchmark_generation_latency(model, test_dataset, num_conditions=3, n_runs=10):
            unique_shapes = list(dict.fromkeys(test_dataset.shape_type.tolist()))
            chosen_indices = []
            for shape in unique_shapes:
                idx = int(np.where(test_dataset.shape_type == shape)[0][0])
                chosen_indices.append(idx)
                if len(chosen_indices) >= num_conditions:
                    break
            if not chosen_indices:
                chosen_indices = list(range(min(num_conditions, len(test_dataset))))

            latencies = []
            for idx in chosen_indices:
                wp_norm, _ = test_dataset[idx]
                wp_t = wp_norm.unsqueeze(0).to(DEVICE, dtype=torch.float32)
                run_latencies = []
                for _ in range(n_runs):
                    maybe_sync()
                    start = time.perf_counter()
                    _ = model.generate(wp_t, num_samples=1)
                    maybe_sync()
                    run_latencies.append((time.perf_counter() - start) * 1000.0)
                latencies.append(float(np.median(run_latencies)))
            return float(np.mean(latencies)), {"condition_indices": chosen_indices, "per_condition_ms": latencies}


        def compute_diversity(preds):
            # preds: (B, K, T, 3)
            pairwise = preds[:, :, None, :, :] - preds[:, None, :, :, :]
            pairwise_rmse = np.sqrt(np.mean(pairwise ** 2, axis=(3, 4)))
            k = preds.shape[1]
            triu = np.triu_indices(k, 1)
            return pairwise_rmse[:, triu[0], triu[1]].mean(axis=1)


        def evaluate_or_load(model, test_dataset, splits, stats, cfg):
            summary_path = METRICS_DIR / "evaluation_summary.json"
            if summary_path.exists() and not cfg["force_reevaluate"]:
                summary = json.loads(summary_path.read_text())
                print("Loaded cached evaluation summary.")
                return summary

            model.eval()
            batch_size = cfg["evaluation"]["batch_size"]
            k_samples = cfg["evaluation"]["generative_samples"]
            loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

            all_samples = []
            all_oracle = []
            all_targets = []
            all_waypoints = []
            all_shapes = []
            shape_rows = defaultdict(list)
            overall_rows = []
            overlay_payload = None

            latency_ms, latency_meta = benchmark_generation_latency(
                model,
                test_dataset,
                num_conditions=cfg["evaluation"]["latency_num_conditions"],
                n_runs=cfg["evaluation"]["latency_runs"],
            )

            start_idx = 0
            for waypoints_norm, q_norm in tqdm(loader, desc="Full evaluation"):
                batch = waypoints_norm.shape[0]
                waypoints_norm = waypoints_norm.to(DEVICE, dtype=torch.float32)
                preds = model.generate(waypoints_norm, num_samples=k_samples)
                preds = preds.detach().cpu().numpy().reshape(batch, k_samples, cfg["num_waypoints"], 3)
                targets = q_norm.numpy()
                waypoints_np = waypoints_norm.detach().cpu().numpy()

                preds_denorm = inverse_normalize_array(preds, stats, "q_sequence")
                targets_denorm = inverse_normalize_array(targets, stats, "q_sequence")
                waypoints_denorm = inverse_normalize_array(waypoints_np, stats, "waypoints")

                joint_rmse = np.sqrt(np.mean((preds_denorm - targets_denorm[:, None, :, :]) ** 2, axis=(2, 3)))
                mean_joint_rmse = joint_rmse.mean(axis=1)
                oracle_idx = np.argmin(joint_rmse, axis=1)
                oracle_preds = preds_denorm[np.arange(batch), oracle_idx]

                ee_oracle = robot.forward_kinematics_batch(oracle_preds)
                ee_target = robot.forward_kinematics_batch(targets_denorm)

                final_pos_err = np.linalg.norm(ee_oracle[:, -1, :2] - ee_target[:, -1, :2], axis=1)
                final_ori_err = np.abs(wrap_angle(ee_oracle[:, -1, 2] - ee_target[:, -1, 2]))
                path_err = np.linalg.norm(ee_oracle[:, :, :2] - ee_target[:, :, :2], axis=2).mean(axis=1)

                vel = np.diff(oracle_preds, axis=1)
                acc = np.diff(vel, axis=1)
                jerk = np.diff(acc, axis=1)
                energy = np.mean(np.sum(vel ** 2, axis=-1), axis=1)
                smoothness = np.mean(np.sum(jerk ** 2, axis=-1), axis=1)

                limits = robot.joint_limits.astype(np.float32)
                lower = limits[:, 0][None, None, :]
                upper = limits[:, 1][None, None, :]
                violations = ((oracle_preds < lower) | (oracle_preds > upper)).mean(axis=(1, 2))
                diversity = compute_diversity(preds_denorm)

                batch_shapes = splits["test"]["shape_type"][start_idx:start_idx + batch]
                for i in range(batch):
                    row = {
                        "joint_rmse": float(joint_rmse[i, oracle_idx[i]]),
                        "ee_position_error": float(final_pos_err[i]),
                        "ee_orientation_error": float(final_ori_err[i]),
                        "path_tracking_error": float(path_err[i]),
                        "energy_proxy": float(energy[i]),
                        "smoothness_jerk": float(smoothness[i]),
                        "constraint_violation_rate": float(violations[i]),
                        "diversity_score": float(diversity[i]),
                        "inference_time_ms": float(latency_ms),
                        "oracle_joint_rmse": float(joint_rmse[i, oracle_idx[i]]),
                        "mean_joint_rmse": float(mean_joint_rmse[i]),
                    }
                    shape = str(batch_shapes[i])
                    overall_rows.append(row)
                    shape_rows[shape].append(row)
                    all_samples.append(preds_denorm[i].astype(np.float32))
                    all_oracle.append(oracle_preds[i].astype(np.float32))
                    all_targets.append(targets_denorm[i].astype(np.float32))
                    all_waypoints.append(waypoints_denorm[i].astype(np.float32))
                    all_shapes.append(shape)

                    if overlay_payload is None:
                        overlay_payload = {
                            "waypoints": waypoints_denorm[i].tolist(),
                            "target": targets_denorm[i].tolist(),
                            "oracle_prediction": oracle_preds[i].tolist(),
                            "all_samples": preds_denorm[i].tolist(),
                            "shape_type": shape,
                        }

                start_idx += batch

            def aggregate(rows):
                keys = rows[0].keys()
                return {key: float(np.mean([row[key] for row in rows])) for key in keys}

            summary = {
                "config": cfg,
                "device": DEVICE,
                "counts": {
                    "train": int(len(splits["train"]["shape_type"])),
                    "val": int(len(splits["val"]["shape_type"])),
                    "test": int(len(splits["test"]["shape_type"])),
                    "test_shape_counts": {
                        str(shape): int(count)
                        for shape, count in zip(*np.unique(splits["test"]["shape_type"], return_counts=True))
                    },
                },
                "overall": aggregate(overall_rows),
                "per_shape": {shape: aggregate(rows) for shape, rows in shape_rows.items()},
                "latency_benchmark": latency_meta,
                "overlay": overlay_payload,
            }

            summary_path.write_text(json.dumps(summary, indent=2))
            (METRICS_DIR / "evaluation_rows.csv").write_text(pd.DataFrame(overall_rows).to_csv(index=False))

            np.savez_compressed(
                CACHE_DIR / "cvae_full_k_samples.npz",
                q_predicted=np.stack(all_samples),
                q_oracle_predicted=np.stack(all_oracle),
                q_reference=np.stack(all_targets),
                waypoints=np.stack(all_waypoints),
                shape_type=np.array(all_shapes),
            )

            return summary


        summary = evaluate_or_load(model, test_dataset, splits, stats, CONFIG)
        summary["overall"]
        """
    ),
    code(
        """
        def write_summary_markdown(summary, output_path):
            overall = summary["overall"]
            lines = [
                "# cVAE Evaluation Summary",
                "",
                f"- Device: {summary['device']}",
                f"- Train samples: {summary['counts']['train']}",
                f"- Val samples: {summary['counts']['val']}",
                f"- Test samples: {summary['counts']['test']}",
                "",
                "## Overall Metrics",
                "",
                "| Metric | Value |",
                "| --- | ---: |",
            ]
            for key, value in overall.items():
                lines.append(f"| {key} | {value:.6f} |")

            lines.extend(["", "## Per-Shape Metrics", ""])
            for shape, metrics in summary["per_shape"].items():
                lines.append(f"### {shape}")
                lines.append("")
                lines.append("| Metric | Value |")
                lines.append("| --- | ---: |")
                for key, value in metrics.items():
                    lines.append(f"| {key} | {value:.6f} |")
                lines.append("")

            output_path.write_text("\\n".join(lines))


        def plot_overlay(summary, output_path):
            payload = summary["overlay"]
            wp = np.asarray(payload["waypoints"])
            target = np.asarray(payload["target"])
            oracle = np.asarray(payload["oracle_prediction"])
            samples = np.asarray(payload["all_samples"])

            plt.figure(figsize=(7, 5))
            plt.plot(wp[:, 0], wp[:, 1], "--", color="black", linewidth=1.0, label="Waypoints")
            target_ee = robot.forward_kinematics_batch(target)
            oracle_ee = robot.forward_kinematics_batch(oracle)
            plt.plot(target_ee[:, 0], target_ee[:, 1], linewidth=2.0, label="Ground Truth")
            for idx, sample in enumerate(samples[:5]):
                ee = robot.forward_kinematics_batch(sample)
                plt.plot(ee[:, 0], ee[:, 1], alpha=0.55, label="Generated samples" if idx == 0 else None)
            plt.plot(oracle_ee[:, 0], oracle_ee[:, 1], linewidth=2.0, linestyle="--", label="Oracle sample")
            plt.legend()
            plt.gca().set_aspect("equal")
            plt.title(f"Generated Trajectory Overlay ({payload['shape_type']})", fontweight="bold")
            plt.tight_layout()
            plt.savefig(output_path, dpi=300)
            plt.close()


        def plot_evaluation_dashboard(summary, output_path):
            per_shape_df = pd.DataFrame(summary["per_shape"]).T.reset_index().rename(columns={"index": "shape_type"})
            fig, axes = plt.subplots(2, 2, figsize=(14, 9))
            axes = axes.ravel()

            overall_items = ["joint_rmse", "ee_position_error", "path_tracking_error", "diversity_score", "inference_time_ms"]
            overall_vals = [summary["overall"][key] for key in overall_items]
            axes[0].bar(overall_items, overall_vals, color=["#b03a2e", "#2874a6", "#117864", "#8e44ad", "#d68910"])
            axes[0].tick_params(axis="x", rotation=30)
            axes[0].set_title("Overall Metrics")

            sns.barplot(data=per_shape_df, x="shape_type", y="joint_rmse", ax=axes[1], color="#1f77b4")
            axes[1].tick_params(axis="x", rotation=30)
            axes[1].set_title("Per-Shape Oracle Joint RMSE")

            sns.barplot(data=per_shape_df, x="shape_type", y="diversity_score", ax=axes[2], color="#2ca02c")
            axes[2].tick_params(axis="x", rotation=30)
            axes[2].set_title("Per-Shape Diversity")

            sns.barplot(data=per_shape_df, x="shape_type", y="mean_joint_rmse", ax=axes[3], color="#d62728")
            axes[3].tick_params(axis="x", rotation=30)
            axes[3].set_title("Per-Shape Mean Joint RMSE")

            for ax in axes:
                ax.grid(True, alpha=0.25, axis="y")
            plt.suptitle("Full cVAE Evaluation Dashboard", fontweight="bold")
            plt.tight_layout()
            plt.savefig(output_path, dpi=300)
            plt.close()


        write_summary_markdown(summary, METRICS_DIR / "evaluation_summary.md")
        plot_overlay(summary, FIG_DIR / "evaluation_overlay.png")
        plot_evaluation_dashboard(summary, FIG_DIR / "evaluation_dashboard.png")

        print(json.dumps(summary["overall"], indent=2))
        """
    ),
    code(
        """
        zip_path = OUTPUT_DIR.with_suffix(".zip")
        if zip_path.exists():
            zip_path.unlink()

        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for path in OUTPUT_DIR.rglob("*"):
                if path.is_file():
                    zf.write(path, path.relative_to(OUTPUT_DIR.parent))

        print(f"Artifacts folder: {OUTPUT_DIR.resolve()}")
        print(f"Zip bundle: {zip_path.resolve()}")
        print("Generated files:")
        for path in sorted(OUTPUT_DIR.rglob("*")):
            if path.is_file():
                print("-", path.relative_to(OUTPUT_DIR))
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


output_path = Path("notebooks/standalone_cvae_full_pipeline.ipynb")
output_path.parent.mkdir(parents=True, exist_ok=True)
output_path.write_text(json.dumps(notebook, indent=2), encoding="utf-8")
print(f"Wrote {output_path}")
