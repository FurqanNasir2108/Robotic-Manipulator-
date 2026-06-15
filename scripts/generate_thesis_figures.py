"""
Generate thesis-quality figures for Chapter 3 (Robot Model & Data) and Chapter 4 (Baselines).

Usage:
    python scripts/generate_thesis_figures.py
"""

import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.simulation.manipulator import ThreeLinkManipulator
from src.simulation.trajectory_generator import TrajectoryGenerator
from src.data.dataset import load_dataset

# Use publication-quality settings
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 14,
    'legend.fontsize': 11,
    'figure.dpi': 300,
    'savefig.dpi': 300,
})

SIM_DIR = 'figures/simulation'
BASELINE_DIR = 'figures/baselines'
DIFFUSION_DIR = 'figures/diffusion'
os.makedirs(SIM_DIR, exist_ok=True)
os.makedirs(BASELINE_DIR, exist_ok=True)
os.makedirs(DIFFUSION_DIR, exist_ok=True)


def plot_arm_diagram():
    """Generate a clear 3-link arm diagram with labeled joints and links."""
    robot = ThreeLinkManipulator()
    q = [0.5, 0.6, -0.3]
    l1, l2, l3 = robot.link_lengths
    q1, q2, q3 = q

    bx, by = robot.base_position
    j1x = bx + l1 * np.cos(q1)
    j1y = by + l1 * np.sin(q1)
    j2x = j1x + l2 * np.cos(q1 + q2)
    j2y = j1y + l2 * np.sin(q1 + q2)
    eex = j2x + l3 * np.cos(q1 + q2 + q3)
    eey = j2y + l3 * np.sin(q1 + q2 + q3)

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    # Draw links
    links_x = [bx, j1x, j2x, eex]
    links_y = [by, j1y, j2y, eey]
    ax.plot(links_x, links_y, 'o-', color='#2c3e50', linewidth=4, markersize=12, zorder=5)

    # Base
    ax.plot(bx, by, 's', color='#e74c3c', markersize=16, zorder=6)

    # End effector
    ax.plot(eex, eey, 'D', color='#27ae60', markersize=12, zorder=6)

    # Label joints
    offset = 0.08
    ax.annotate('Base\n$(0, 0)$', (bx, by), (bx - 0.3, by - 0.25),
                fontsize=11, ha='center', fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='gray'))
    ax.annotate('Joint 1\n$q_1$', (j1x, j1y), (j1x + 0.15, j1y + 0.15),
                fontsize=11, fontweight='bold')
    ax.annotate('Joint 2\n$q_2$', (j2x, j2y), (j2x + 0.15, j2y + 0.12),
                fontsize=11, fontweight='bold')
    ax.annotate('End Effector\n$(x, y, \\theta)$', (eex, eey), (eex + 0.1, eey - 0.2),
                fontsize=11, fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='gray'))

    # Label links
    mid1x, mid1y = (bx + j1x) / 2, (by + j1y) / 2
    mid2x, mid2y = (j1x + j2x) / 2, (j1y + j2y) / 2
    mid3x, mid3y = (j2x + eex) / 2, (j2y + eey) / 2
    ax.annotate('$l_1 = 1.0$', (mid1x, mid1y), (mid1x - 0.25, mid1y + 0.12),
                fontsize=11, color='#2980b9', fontweight='bold')
    ax.annotate('$l_2 = 1.0$', (mid2x, mid2y), (mid2x - 0.25, mid2y + 0.12),
                fontsize=11, color='#2980b9', fontweight='bold')
    ax.annotate('$l_3 = 0.5$', (mid3x, mid3y), (mid3x + 0.05, mid3y + 0.12),
                fontsize=11, color='#2980b9', fontweight='bold')

    # Draw joint angle arcs
    for angle, center, ref_angle in [
        (q1, (bx, by), 0),
        (q2, (j1x, j1y), q1),
        (q3, (j2x, j2y), q1 + q2),
    ]:
        arc = patches.Arc(center, 0.3, 0.3,
                          angle=0,
                          theta1=np.degrees(ref_angle),
                          theta2=np.degrees(ref_angle + angle),
                          color='#e67e22', linewidth=2, linestyle='--')
        ax.add_patch(arc)

    ax.set_xlim(-0.5, 2.8)
    ax.set_ylim(-0.5, 2.0)
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_title('3-Link Planar Manipulator')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    plt.savefig(os.path.join(SIM_DIR, 'arm_diagram.png'))
    plt.close()
    print("  Saved arm_diagram.png")


def plot_workspace():
    """Generate workspace boundary by sampling random joint configs."""
    robot = ThreeLinkManipulator()
    rng = np.random.default_rng(42)
    N = 50000
    q1 = rng.uniform(-np.pi, np.pi, N)
    q2 = rng.uniform(-np.pi, np.pi, N)
    q3 = rng.uniform(-np.pi, np.pi, N)

    poses = np.array([robot.forward_kinematics([q1[i], q2[i], q3[i]]) for i in range(N)])

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.scatter(poses[:, 0], poses[:, 1], s=0.5, alpha=0.3, c='steelblue')
    ax.plot(0, 0, 's', color='red', markersize=10, label='Base', zorder=5)

    # Draw max reach circle
    max_reach = sum(robot.link_lengths)
    min_reach = abs(robot.link_lengths[0] - robot.link_lengths[1]) - robot.link_lengths[2]
    circle_max = plt.Circle((0, 0), max_reach, fill=False, color='red',
                             linestyle='--', linewidth=1.5, label=f'Max reach = {max_reach}')
    ax.add_patch(circle_max)
    if min_reach > 0:
        circle_min = plt.Circle((0, 0), min_reach, fill=False, color='orange',
                                 linestyle='--', linewidth=1.5, label=f'Min reach = {min_reach:.2f}')
        ax.add_patch(circle_min)

    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_title('Workspace of 3-Link Planar Manipulator')
    ax.set_aspect('equal')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    plt.savefig(os.path.join(SIM_DIR, 'workspace.png'))
    plt.close()
    print("  Saved workspace.png")


def plot_all_shapes_grid():
    """Generate a grid of all 6 trajectory shapes in task space."""
    gen = TrajectoryGenerator(num_waypoints=100)
    shapes = {
        'Circle': gen.circle(center=(1.5, 0.0), radius=0.3),
        'Square': gen.square(center=(1.5, 0.0), side_length=0.4),
        'Pentagon': gen.pentagon(center=(1.5, 0.0), radius=0.3),
        'Hexagon': gen.hexagon(center=(1.5, 0.0), radius=0.3),
        'Line': gen.line(start=(1.2, -0.3), end=(1.8, 0.3)),
        'Random Smooth': gen.random_smooth(rng=np.random.default_rng(42)),
    }

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    for ax, (name, wp) in zip(axes.flat, shapes.items()):
        ax.plot(wp[:, 0], wp[:, 1], 'b-', linewidth=1.5)
        ax.plot(wp[0, 0], wp[0, 1], 'go', markersize=8, label='Start')
        ax.plot(wp[-1, 0], wp[-1, 1], 'ro', markersize=8, label='End')
        ax.set_title(name, fontsize=13, fontweight='bold')
        ax.set_xlabel('x (m)')
        ax.set_ylabel('y (m)')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)
    plt.suptitle('Trajectory Shape Types', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(SIM_DIR, 'shape_trajectories_grid.png'))
    plt.close()
    print("  Saved shape_trajectories_grid.png")


def plot_ik_branches():
    """Show elbow-up vs elbow-down IK solutions for the same pose."""
    robot = ThreeLinkManipulator()
    q_orig = [0.5, 0.8, -0.3]
    pose = robot.forward_kinematics(q_orig)

    sols_up = robot.inverse_kinematics(pose, elbow='up')
    sols_down = robot.inverse_kinematics(pose, elbow='down')

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    titles = ['Elbow-Up', 'Elbow-Down']
    solutions = [sols_up, sols_down]
    colors = ['#2980b9', '#e74c3c']

    for ax, title, sols, color in zip(axes, titles, solutions, colors):
        if sols:
            q = sols[0]
            l1, l2, l3 = robot.link_lengths
            bx, by = robot.base_position
            j1x = bx + l1 * np.cos(q[0])
            j1y = by + l1 * np.sin(q[0])
            j2x = j1x + l2 * np.cos(q[0] + q[1])
            j2y = j1y + l2 * np.sin(q[0] + q[1])
            eex = j2x + l3 * np.cos(q[0] + q[1] + q[2])
            eey = j2y + l3 * np.sin(q[0] + q[1] + q[2])
            ax.plot([bx, j1x, j2x, eex], [by, j1y, j2y, eey],
                    'o-', color=color, linewidth=3, markersize=10)
            ax.plot(eex, eey, 'D', color='#27ae60', markersize=12, zorder=6)
            ax.set_title(f'{title}\n$q = [{q[0]:.2f}, {q[1]:.2f}, {q[2]:.2f}]$')
        else:
            ax.set_title(f'{title}\nNo solution')
        ax.plot(pose[0], pose[1], '*', color='gold', markersize=15, zorder=7, label='Target')
        ax.set_xlabel('x (m)')
        ax.set_ylabel('y (m)')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_xlim(-0.5, 2.5)
        ax.set_ylim(-0.5, 2.0)

    plt.suptitle('Inverse Kinematics: Elbow-Up vs Elbow-Down', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(SIM_DIR, 'ik_branches.png'))
    plt.close()
    print("  Saved ik_branches.png")


def plot_dataset_distribution():
    """Plot dataset statistics: samples per shape, joint angle distributions."""
    train = load_dataset('data/processed/train.npz')
    shapes = train['shape_type']
    unique, counts = np.unique(shapes, return_counts=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Samples per shape
    colors = plt.cm.Set2(np.linspace(0, 1, len(unique)))
    axes[0].bar(unique, counts, color=colors)
    axes[0].set_xlabel('Shape Type')
    axes[0].set_ylabel('Number of Samples')
    axes[0].set_title('Training Set: Samples per Shape')
    for i, (u, c) in enumerate(zip(unique, counts)):
        axes[0].text(i, c + 50, str(c), ha='center', fontweight='bold')

    # Joint angle distribution
    q_seq = train['q_sequence']
    q_flat = q_seq.reshape(-1, 3)
    labels = ['$q_1$', '$q_2$', '$q_3$']
    axes[1].hist(q_flat[:, 0], bins=80, alpha=0.6, label=labels[0], color='#e74c3c')
    axes[1].hist(q_flat[:, 1], bins=80, alpha=0.6, label=labels[1], color='#2980b9')
    axes[1].hist(q_flat[:, 2], bins=80, alpha=0.6, label=labels[2], color='#27ae60')
    axes[1].set_xlabel('Joint Angle (rad)')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Training Set: Joint Angle Distribution')
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(SIM_DIR, 'dataset_distribution.png'))
    plt.close()
    print("  Saved dataset_distribution.png")


def plot_baseline_training_curves():
    """Plot training curves for all baselines."""
    metrics_dir = 'results/metrics/baselines'
    models = ['cnn', 'cnn_lstm', 'cnn_gru']
    labels = ['CNN', 'CNN+LSTM', 'CNN+GRU']
    colors = ['#e74c3c', '#2980b9', '#27ae60']

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for model, label, color in zip(models, labels, colors):
        hist_path = os.path.join(metrics_dir, f'{model}_history.json')
        if not os.path.exists(hist_path):
            print(f"  Skipping {model} (no history file)")
            continue
        with open(hist_path, 'r') as f:
            hist = json.load(f)
        epochs = range(1, len(hist['train_losses']) + 1)
        axes[0].plot(epochs, hist['train_losses'], color=color, label=label, linewidth=1.5)
        axes[1].plot(epochs, hist['val_losses'], color=color, label=label, linewidth=1.5)

    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Training Loss')
    axes[0].set_title('Training Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Validation Loss')
    axes[1].set_title('Validation Loss')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.suptitle('Baseline Model Training Curves', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(BASELINE_DIR, 'training_curves.png'))
    plt.close()
    print("  Saved training_curves.png")


def plot_baseline_comparison_bar():
    """Plot bar chart comparing best val loss of baselines."""
    metrics_dir = 'results/metrics/baselines'
    models = ['cnn', 'cnn_lstm', 'cnn_gru']
    labels = ['CNN', 'CNN+LSTM', 'CNN+GRU']
    colors = ['#e74c3c', '#2980b9', '#27ae60']

    best_losses = []
    for model in models:
        hist_path = os.path.join(metrics_dir, f'{model}_history.json')
        if os.path.exists(hist_path):
            with open(hist_path, 'r') as f:
                hist = json.load(f)
            best_losses.append(min(hist['val_losses']))
        else:
            best_losses.append(0)

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    bars = ax.bar(labels, best_losses, color=colors, edgecolor='black', linewidth=0.5)
    for bar, val in zip(bars, best_losses):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.0005,
                f'{val:.4f}', ha='center', fontweight='bold', fontsize=11)
    ax.set_ylabel('Best Validation Loss')
    ax.set_title('Baseline Model Comparison')
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(BASELINE_DIR, 'comparison_bar.png'))
    plt.close()
    print("  Saved comparison_bar.png")


VAE_DIR = 'figures/vae'
os.makedirs(VAE_DIR, exist_ok=True)


def plot_vae_training_curves():
    """Plot cVAE training curves: reconstruction, KL, total loss."""
    hist_path = 'results/metrics/cvae/cvae_history.json'
    if not os.path.exists(hist_path):
        print("  Skipping VAE training curves (no history file)")
        return
    with open(hist_path, 'r') as f:
        hist = json.load(f)

    epochs = range(1, len(hist['train_total']) + 1)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Total loss
    axes[0, 0].plot(epochs, hist['train_total'], label='Train', color='#e74c3c', linewidth=1.2)
    axes[0, 0].plot(epochs, hist['val_total'], label='Val', color='#2980b9', linewidth=1.2)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Total Loss')
    axes[0, 0].set_title('Total Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Reconstruction loss
    axes[0, 1].plot(epochs, hist['train_recon'], label='Train', color='#e74c3c', linewidth=1.2)
    axes[0, 1].plot(epochs, hist['val_recon'], label='Val', color='#2980b9', linewidth=1.2)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Reconstruction Loss')
    axes[0, 1].set_title('Reconstruction Loss (MSE)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # KL divergence
    axes[1, 0].plot(epochs, hist['train_kl'], label='Train KL', color='#8e44ad', linewidth=1.2)
    axes[1, 0].plot(epochs, hist['val_kl'], label='Val KL', color='#27ae60', linewidth=1.2)
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('KL Divergence')
    axes[1, 0].set_title('KL Divergence')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # KL weight (annealing schedule)
    axes[1, 1].plot(epochs, hist['kl_weight'], color='#e67e22', linewidth=1.5)
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('KL Weight')
    axes[1, 1].set_title('KL Annealing Schedule (Cyclical)')
    axes[1, 1].grid(True, alpha=0.3)

    plt.suptitle('Conditional VAE Training Curves', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(VAE_DIR, 'training_curves.png'))
    plt.close()
    print("  Saved training_curves.png")


def plot_vae_reconstruction_samples():
    """Plot VAE reconstruction vs ground truth on test samples."""
    import torch
    from src.data.normalization import Normalizer
    from src.data.loaders import get_dataloaders
    from src.models.cvae import ConditionalVAE
    from src.utils.config import load_config

    cfg = load_config('configs/vae.yaml')
    ckpt_path = 'results/checkpoints/cvae/cvae_best.pth'
    if not os.path.exists(ckpt_path):
        print("  Skipping VAE reconstruction (no checkpoint)")
        return

    device = 'cpu'
    norm_path = os.path.join(cfg['data']['metadata_dir'], 'normalization_stats.json')
    normalizer = Normalizer.load(norm_path)
    loaders = get_dataloaders(cfg['data']['dir'], normalizer=normalizer,
                              batch_size=8, input_mode=cfg['data']['input_mode'])

    # Build model and load weights
    m = cfg['model']
    model = ConditionalVAE(
        latent_dim=m['latent_dim'], encoder_type=m['encoder_type'],
        decoder_type=m['decoder_type'], condition_dim=m['condition_dim'],
        input_dim=m['input_dim'], output_dim=m['output_dim'],
        input_steps=m['input_steps'], output_steps=m['output_steps'],
        cond_conv_channels=m.get('cond_conv_channels'),
        cond_kernel_sizes=m.get('cond_kernel_sizes'),
        enc_hidden=m.get('enc_hidden', 128), enc_layers=m.get('enc_layers', 2),
        enc_dropout=m.get('enc_dropout', 0.1), dec_hidden=m.get('dec_hidden', 128),
        dec_layers=m.get('dec_layers', 2), dec_dropout=m.get('dec_dropout', 0.1),
        dec_fc_dims=m.get('dec_fc_dims'),
    )
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    # Get a batch from test set
    test_loader = loaders.get('test', loaders.get('val'))
    condition, target = next(iter(test_loader))
    condition = condition.to(device, dtype=torch.float32)
    target = target.to(device, dtype=torch.float32)

    with torch.no_grad():
        recon, mu, log_var = model(condition, target)

    # Plot 4 samples
    fig, axes = plt.subplots(2, 4, figsize=(20, 8))
    joint_labels = ['$q_1$', '$q_2$', '$q_3$']
    for i in range(4):
        gt = target[i].cpu().numpy()
        rc = recon[i].cpu().numpy()
        t = np.arange(gt.shape[0])

        for j in range(3):
            axes[0, i].plot(t, gt[:, j], '-', linewidth=1.5, label=f'GT {joint_labels[j]}')
            axes[0, i].plot(t, rc[:, j], '--', linewidth=1.5, label=f'Recon {joint_labels[j]}')
        axes[0, i].set_title(f'Sample {i+1}')
        axes[0, i].set_xlabel('Timestep')
        axes[0, i].set_ylabel('Joint Angle (normalized)')
        axes[0, i].grid(True, alpha=0.3)
        if i == 0:
            axes[0, i].legend(fontsize=7)

        # Error plot
        err = np.abs(gt - rc)
        for j in range(3):
            axes[1, i].plot(t, err[:, j], linewidth=1.2, label=joint_labels[j])
        axes[1, i].set_title(f'Abs Error — Sample {i+1}')
        axes[1, i].set_xlabel('Timestep')
        axes[1, i].set_ylabel('|GT − Recon|')
        axes[1, i].grid(True, alpha=0.3)
        if i == 0:
            axes[1, i].legend(fontsize=8)

    plt.suptitle('cVAE Reconstruction Quality', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(VAE_DIR, 'reconstruction_samples.png'))
    plt.close()
    print("  Saved reconstruction_samples.png")


def plot_vae_latent_space():
    """Plot t-SNE of VAE latent space colored by shape type."""
    import torch
    from sklearn.manifold import TSNE
    from src.data.normalization import Normalizer
    from src.data.loaders import TrajectoryDataset
    from src.models.cvae import ConditionalVAE
    from src.utils.config import load_config

    cfg = load_config('configs/vae.yaml')
    ckpt_path = 'results/checkpoints/cvae/cvae_best.pth'
    if not os.path.exists(ckpt_path):
        print("  Skipping VAE latent space (no checkpoint)")
        return

    device = 'cpu'
    norm_path = os.path.join(cfg['data']['metadata_dir'], 'normalization_stats.json')
    normalizer = Normalizer.load(norm_path)

    # Load test dataset directly to access shape labels
    test_ds = TrajectoryDataset('data/processed/test.npz', normalizer=normalizer,
                                input_mode=cfg['data']['input_mode'])

    m = cfg['model']
    model = ConditionalVAE(
        latent_dim=m['latent_dim'], encoder_type=m['encoder_type'],
        decoder_type=m['decoder_type'], condition_dim=m['condition_dim'],
        input_dim=m['input_dim'], output_dim=m['output_dim'],
        input_steps=m['input_steps'], output_steps=m['output_steps'],
        cond_conv_channels=m.get('cond_conv_channels'),
        cond_kernel_sizes=m.get('cond_kernel_sizes'),
        enc_hidden=m.get('enc_hidden', 128), enc_layers=m.get('enc_layers', 2),
        enc_dropout=m.get('enc_dropout', 0.1), dec_hidden=m.get('dec_hidden', 128),
        dec_layers=m.get('dec_layers', 2), dec_dropout=m.get('dec_dropout', 0.1),
        dec_fc_dims=m.get('dec_fc_dims'),
    )
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    # Encode all test samples
    N = min(len(test_ds), 2000)
    latents = []
    shapes = test_ds.shape_type[:N]
    for i in range(N):
        cond, tgt = test_ds[i]
        cond = cond.unsqueeze(0).to(device, dtype=torch.float32)
        tgt = tgt.unsqueeze(0).to(device, dtype=torch.float32)
        with torch.no_grad():
            c = model.cond_encoder(cond)
            mu, _ = model.encode(tgt, c)
        latents.append(mu.squeeze(0).numpy())

    latents = np.array(latents)
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    z_2d = tsne.fit_transform(latents)

    unique_shapes = sorted(set(shapes))
    colors = plt.cm.Set1(np.linspace(0, 0.8, len(unique_shapes)))
    shape_to_color = {s: c for s, c in zip(unique_shapes, colors)}

    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    for shape in unique_shapes:
        mask = shapes == shape
        ax.scatter(z_2d[mask, 0], z_2d[mask, 1], s=15, alpha=0.7,
                   color=shape_to_color[shape], label=shape)
    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')
    ax.set_title('cVAE Latent Space (t-SNE) Colored by Shape Type')
    ax.legend(markerscale=3)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(VAE_DIR, 'latent_tsne.png'))
    plt.close()
    print("  Saved latent_tsne.png")


def plot_vae_generation_samples():
    """Plot generated trajectories from prior z ~ N(0, I)."""
    import torch
    from src.data.normalization import Normalizer
    from src.data.loaders import get_dataloaders
    from src.models.cvae import ConditionalVAE
    from src.utils.config import load_config
    from src.simulation.manipulator import ThreeLinkManipulator

    cfg = load_config('configs/vae.yaml')
    ckpt_path = 'results/checkpoints/cvae/cvae_best.pth'
    if not os.path.exists(ckpt_path):
        print("  Skipping VAE generation samples (no checkpoint)")
        return

    device = 'cpu'
    norm_path = os.path.join(cfg['data']['metadata_dir'], 'normalization_stats.json')
    normalizer = Normalizer.load(norm_path)
    loaders = get_dataloaders(cfg['data']['dir'], normalizer=normalizer,
                              batch_size=4, input_mode=cfg['data']['input_mode'])

    m = cfg['model']
    model = ConditionalVAE(
        latent_dim=m['latent_dim'], encoder_type=m['encoder_type'],
        decoder_type=m['decoder_type'], condition_dim=m['condition_dim'],
        input_dim=m['input_dim'], output_dim=m['output_dim'],
        input_steps=m['input_steps'], output_steps=m['output_steps'],
        cond_conv_channels=m.get('cond_conv_channels'),
        cond_kernel_sizes=m.get('cond_kernel_sizes'),
        enc_hidden=m.get('enc_hidden', 128), enc_layers=m.get('enc_layers', 2),
        enc_dropout=m.get('enc_dropout', 0.1), dec_hidden=m.get('dec_hidden', 128),
        dec_layers=m.get('dec_layers', 2), dec_dropout=m.get('dec_dropout', 0.1),
        dec_fc_dims=m.get('dec_fc_dims'),
    )
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    # Get conditions and generate
    test_loader = loaders.get('test', loaders.get('val'))
    condition, target = next(iter(test_loader))
    condition = condition.to(device, dtype=torch.float32)

    with torch.no_grad():
        generated = model.generate(condition, num_samples=3)  # (4*3, T, 3)

    robot = ThreeLinkManipulator()
    gen_np = generated.cpu().numpy()
    # Inverse-normalize
    gen_orig = normalizer.inverse_transform(gen_np, 'q_sequence')

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    for i in range(4):
        # 3 samples per condition
        for s in range(3):
            idx = i * 3 + s
            q_seq = gen_orig[idx]
            # FK to task space
            ee_traj = np.array([robot.forward_kinematics(q_seq[t]) for t in range(len(q_seq))])
            axes[i].plot(ee_traj[:, 0], ee_traj[:, 1], '-', linewidth=1.2,
                         alpha=0.7, label=f'Sample {s+1}')
        axes[i].set_title(f'Condition {i+1}')
        axes[i].set_xlabel('x (m)')
        axes[i].set_ylabel('y (m)')
        axes[i].set_aspect('equal')
        axes[i].grid(True, alpha=0.3)
        axes[i].legend(fontsize=8)

    plt.suptitle('cVAE Generated Trajectories (3 Samples per Condition)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(VAE_DIR, 'generation_samples.png'))
    plt.close()
    print("  Saved generation_samples.png")


def _load_diffusion_artifacts(device='cpu'):
    """Load diffusion config, normalizer, test loader, and EMA model."""
    import torch
    from src.data.normalization import Normalizer
    from src.data.loaders import get_dataloaders
    from src.models.diffusion import build_diffusion_model
    from src.utils.config import load_config

    cfg = load_config('configs/diffusion.yaml')
    ckpt_path = 'results/checkpoints/diffusion/diffusion_best.pth'
    hist_path = 'results/metrics/diffusion/diffusion_history.json'

    if not os.path.exists(ckpt_path):
        print("  Skipping diffusion figures (no checkpoint)")
        return None

    norm_path = os.path.join(cfg['data']['metadata_dir'], 'normalization_stats.json')
    normalizer = Normalizer.load(norm_path)
    loaders = get_dataloaders(
        cfg['data']['dir'],
        normalizer=normalizer,
        batch_size=4,
        input_mode=cfg['data']['input_mode'],
    )

    model, ema = build_diffusion_model(cfg)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    ema.load_state_dict(ckpt['ema_state_dict'])
    model = ema.shadow.to(device)
    model.eval()

    return {
        'cfg': cfg,
        'history_path': hist_path,
        'model': model,
        'normalizer': normalizer,
        'loaders': loaders,
    }


def plot_diffusion_training_curves():
    """Plot diffusion training and validation noise-prediction loss."""
    bundle = _load_diffusion_artifacts(device='cpu')
    if bundle is None:
        return

    hist_path = bundle['history_path']
    if not os.path.exists(hist_path):
        print("  Skipping diffusion training curves (no history file)")
        return

    with open(hist_path, 'r') as f:
        hist = json.load(f)

    epochs = range(1, len(hist['train_losses']) + 1)
    best_epoch = int(np.argmin(hist['val_losses'])) + 1
    best_val = float(np.min(hist['val_losses']))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(epochs, hist['train_losses'], color='#c0392b', linewidth=1.4)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Train Noise MSE')
    axes[0].set_title('Diffusion Training Loss')
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, hist['val_losses'], color='#1f618d', linewidth=1.4)
    axes[1].axvline(best_epoch, color='#117a65', linestyle='--', linewidth=1.0,
                    label=f'Best epoch = {best_epoch}')
    axes[1].scatter([best_epoch], [best_val], color='#117a65', s=30, zorder=5)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Validation Noise MSE')
    axes[1].set_title('Diffusion Validation Loss')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.suptitle('Conditional Diffusion Training Curves', fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(DIFFUSION_DIR, 'training_curves.png'))
    plt.close()
    print("  Saved training_curves.png")


def plot_diffusion_generation_samples():
    """Plot DDIM-generated task-space trajectories for several conditions."""
    import torch

    bundle = _load_diffusion_artifacts(device='cpu')
    if bundle is None:
        return

    model = bundle['model']
    normalizer = bundle['normalizer']
    loaders = bundle['loaders']
    cfg = bundle['cfg']
    robot = ThreeLinkManipulator()

    test_loader = loaders.get('test', loaders.get('val'))
    condition, _ = next(iter(test_loader))
    condition = condition.to('cpu', dtype=torch.float32)

    with torch.no_grad():
        generated = model.generate(
            condition,
            num_samples=3,
            method='ddim',
            num_steps=cfg['sampling'].get('ddim_steps', 50),
            guidance_scale=cfg['sampling'].get('guidance_scale', 2.0),
            seq_len=cfg['sampling'].get('seq_len', 100),
        )

    gen_np = generated.cpu().numpy()
    gen_orig = normalizer.inverse_transform(gen_np, 'q_sequence')

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    for i in range(4):
        for sample_idx in range(3):
            idx = i * 3 + sample_idx
            q_seq = gen_orig[idx]
            ee_traj = np.array([robot.forward_kinematics(q_seq[t]) for t in range(len(q_seq))])
            axes[i].plot(ee_traj[:, 0], ee_traj[:, 1], linewidth=1.3, alpha=0.8,
                         label=f'Sample {sample_idx + 1}')
        wp = normalizer.inverse_transform(condition[i].cpu().numpy(), 'waypoints')
        axes[i].plot(wp[:, 0], wp[:, 1], '--', color='black', linewidth=1.0, alpha=0.7,
                     label='Condition')
        axes[i].set_title(f'Condition {i + 1}')
        axes[i].set_xlabel('x (m)')
        axes[i].set_ylabel('y (m)')
        axes[i].set_aspect('equal')
        axes[i].grid(True, alpha=0.3)
        axes[i].legend(fontsize=8)

    plt.suptitle('Diffusion-Generated Trajectories (DDIM, 3 Samples per Condition)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(DIFFUSION_DIR, 'generation_samples.png'))
    plt.close()
    print("  Saved generation_samples.png")


def plot_diffusion_denoising_progression():
    """Visualize denoising from pure noise to the final trajectory for one condition."""
    import torch

    bundle = _load_diffusion_artifacts(device='cpu')
    if bundle is None:
        return

    model = bundle['model']
    normalizer = bundle['normalizer']
    loaders = bundle['loaders']
    cfg = bundle['cfg']
    robot = ThreeLinkManipulator()

    test_loader = loaders.get('test', loaders.get('val'))
    condition, _ = next(iter(test_loader))
    waypoints = condition[:1].to('cpu', dtype=torch.float32)

    cond = model.condition_encoder(waypoints)
    seq_len = cfg['sampling'].get('seq_len', 100)
    x = torch.randn(1, model.joint_dim, seq_len)
    timesteps = [999, 750, 500, 250, 0]
    captures = []

    for step in reversed(range(model.num_timesteps)):
        t = torch.full((1,), step, dtype=torch.long)
        noise_pred = model.denoiser(x, t, cond)
        guidance_scale = cfg['sampling'].get('guidance_scale', 2.0)
        if guidance_scale != 1.0:
            noise_uncond = model.denoiser(x, t, torch.zeros_like(cond))
            noise_pred = noise_uncond + guidance_scale * (noise_pred - noise_uncond)

        mean = model.sqrt_recip_alpha[step] * (
            x - model.betas[step] / model.sqrt_one_minus_alpha_bar[step] * noise_pred
        )
        if step > 0:
            x = mean + torch.sqrt(model.posterior_variance[step]) * torch.randn_like(x)
        else:
            x = mean

        if step in timesteps:
            captures.append((step, x.clone().permute(0, 2, 1).cpu().numpy()[0]))

    fig, axes = plt.subplots(1, len(captures), figsize=(22, 4.5))
    for ax, (step, q_seq) in zip(axes, captures):
        q_orig = normalizer.inverse_transform(q_seq, 'q_sequence')
        ee_traj = np.array([robot.forward_kinematics(q_orig[t]) for t in range(len(q_orig))])
        wp = normalizer.inverse_transform(waypoints[0].cpu().numpy(), 'waypoints')
        ax.plot(wp[:, 0], wp[:, 1], '--', color='black', linewidth=1.0, alpha=0.7)
        ax.plot(ee_traj[:, 0], ee_traj[:, 1], color='#b03a2e', linewidth=1.4)
        ax.set_title(f't = {step}')
        ax.set_xlabel('x (m)')
        ax.set_ylabel('y (m)')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)

    plt.suptitle('Diffusion Denoising Progression (Noise to Trajectory)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(DIFFUSION_DIR, 'denoising_progression.png'))
    plt.close()
    print("  Saved denoising_progression.png")


if __name__ == '__main__':
    print("Generating thesis figures...")
    print("\n--- Chapter 3: Simulation ---")
    plot_arm_diagram()
    plot_workspace()
    plot_all_shapes_grid()
    plot_ik_branches()
    plot_dataset_distribution()
    print("\n--- Chapter 4: Baselines ---")
    plot_baseline_training_curves()
    plot_baseline_comparison_bar()
    print("\n--- Chapter 5: Conditional VAE ---")
    plot_vae_training_curves()
    plot_vae_reconstruction_samples()
    plot_vae_latent_space()
    plot_vae_generation_samples()
    print("\n--- Chapter 6: Diffusion ---")
    plot_diffusion_training_curves()
    plot_diffusion_generation_samples()
    plot_diffusion_denoising_progression()
    print("\nDone!")
