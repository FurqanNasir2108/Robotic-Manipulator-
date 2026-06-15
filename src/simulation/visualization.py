"""
Visualization utilities for 3-link manipulator trajectories.
Produces plots of task-space paths, manipulator configurations, and joint-space sequences.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from src.simulation.manipulator import ThreeLinkManipulator


def plot_task_space_trajectory(waypoints, title='Task-Space Trajectory', ax=None, save_path=None):
    """
    Plot a trajectory in task space (x, y).

    Parameters
    ----------
    waypoints : ndarray of shape (N, 3)
        (x, y, theta) waypoints.
    title : str
    ax : matplotlib.axes.Axes or None
    save_path : str or None
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.plot(waypoints[:, 0], waypoints[:, 1], 'b-', linewidth=1.5)
    ax.plot(waypoints[0, 0], waypoints[0, 1], 'go', markersize=8, label='Start')
    ax.plot(waypoints[-1, 0], waypoints[-1, 1], 'ro', markersize=8, label='End')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title(title)
    ax.set_aspect('equal')
    ax.legend()
    ax.grid(True, alpha=0.3)
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    return ax


def plot_manipulator_config(robot, q, ax=None, color='b', alpha=1.0):
    """
    Draw the 3-link manipulator in a given joint configuration.

    Parameters
    ----------
    robot : ThreeLinkManipulator
    q : array-like of shape (3,)
    ax : matplotlib.axes.Axes or None
    color : str
    alpha : float
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    l1, l2, l3 = robot.link_lengths
    q1, q2, q3 = q
    bx, by = robot.base_position
    j1x = bx + l1 * np.cos(q1)
    j1y = by + l1 * np.sin(q1)
    j2x = j1x + l2 * np.cos(q1 + q2)
    j2y = j1y + l2 * np.sin(q1 + q2)
    eex = j2x + l3 * np.cos(q1 + q2 + q3)
    eey = j2y + l3 * np.sin(q1 + q2 + q3)
    xs = [bx, j1x, j2x, eex]
    ys = [by, j1y, j2y, eey]
    ax.plot(xs, ys, 'o-', color=color, linewidth=2, markersize=6, alpha=alpha)
    return ax


def plot_joint_trajectory(q_sequence, title='Joint-Space Trajectory', save_path=None):
    """
    Plot joint angles over time.

    Parameters
    ----------
    q_sequence : ndarray of shape (N, 3)
    title : str
    save_path : str or None
    """
    fig, axes = plt.subplots(3, 1, figsize=(8, 6), sharex=True)
    labels = ['q1', 'q2', 'q3']
    for i, ax in enumerate(axes):
        ax.plot(q_sequence[:, i], linewidth=1.5)
        ax.set_ylabel(labels[i])
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel('Waypoint index')
    fig.suptitle(title)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig


def plot_sample_trajectories(generator, shapes=None, save_dir='figures/simulation'):
    """
    Generate and plot one sample trajectory per shape type.

    Parameters
    ----------
    generator : TrajectoryGenerator
    shapes : list of str or None
        If None, uses all standard shapes.
    save_dir : str
    """
    if shapes is None:
        shapes = ['circle', 'square', 'pentagon', 'hexagon', 'line', 'random_smooth']

    os.makedirs(save_dir, exist_ok=True)

    for shape in shapes:
        kwargs = {}
        if shape == 'random_smooth':
            kwargs['rng'] = np.random.default_rng(42)
        result = generator.generate(shape, **kwargs)
        if result is None:
            print(f"Warning: Could not generate feasible trajectory for '{shape}'")
            continue

        # Task-space plot
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        plot_task_space_trajectory(result['waypoints'], title=f'{shape} - Task Space', ax=axes[0])

        # Overlay manipulator at a few configs
        robot = generator.robot
        step = max(1, len(result['q_sequence']) // 8)
        for i in range(0, len(result['q_sequence']), step):
            plot_manipulator_config(robot, result['q_sequence'][i], ax=axes[1],
                                    color=plt.cm.viridis(i / len(result['q_sequence'])),
                                    alpha=0.5)
        axes[1].set_title(f'{shape} - Manipulator Configs')
        axes[1].set_aspect('equal')
        axes[1].grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'{shape}_trajectory.png'), dpi=150, bbox_inches='tight')
        plt.close(fig)

        # Joint-space plot
        plot_joint_trajectory(result['q_sequence'], title=f'{shape} - Joint Space',
                              save_path=os.path.join(save_dir, f'{shape}_joints.png'))
        plt.close('all')

    print(f"Sample trajectory plots saved to {save_dir}")