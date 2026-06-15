"""Utility functions for trajectory analysis: loading, derivatives, smoothing."""

from __future__ import annotations

import json
import os
from typing import Callable

import numpy as np


def _to_numpy(array_like):
    """Convert torch or numpy inputs to numpy arrays."""
    if hasattr(array_like, "detach"):
        return array_like.detach().cpu().numpy()
    return np.asarray(array_like, dtype=np.float64)


def load_generated_trajectories(model_name: str, results_dir: str = "results/generated_trajectories") -> dict:
    """Load cached generated trajectories for a model.

    Parameters
    ----------
    model_name : str
        Name of the model (e.g. 'cvae', 'cnn').
    results_dir : str
        Directory containing per-model .npz files.

    Returns
    -------
    dict
        Dictionary with keys: q_predicted, q_reference, waypoints, shape_types, metadata.
    """
    path = os.path.join(results_dir, f"{model_name}.npz")
    if not os.path.exists(path):
        raise FileNotFoundError(f"No generated trajectories found at {path}")
    data = np.load(path, allow_pickle=True)
    result = {k: data[k] for k in data.files}
    return result


def compute_task_space_trajectory(q_trajectory: np.ndarray, fk_fn: Callable) -> np.ndarray:
    """Convert joint-space trajectory to task-space using forward kinematics.

    Parameters
    ----------
    q_trajectory : np.ndarray
        Joint trajectory of shape (T, 3).
    fk_fn : callable
        Forward kinematics function: q -> [x, y, theta].

    Returns
    -------
    np.ndarray
        Task-space trajectory of shape (T, 3).
    """
    q_trajectory = _to_numpy(q_trajectory)
    return np.array([fk_fn(q) for q in q_trajectory])


def finite_difference(signal: np.ndarray, dt: float, order: int = 1) -> np.ndarray:
    """Compute finite-difference derivative of a signal.

    Parameters
    ----------
    signal : np.ndarray
        Signal of shape (T, D).
    dt : float
        Time step.
    order : int
        Derivative order (1=velocity, 2=acceleration, 3=jerk).

    Returns
    -------
    np.ndarray
        Derivative of shape (T-order, D).
    """
    signal = _to_numpy(signal)
    result = signal
    for _ in range(order):
        result = np.diff(result, axis=0) / dt
    return result


def smooth_trajectory(trajectory: np.ndarray, window: int = 5, method: str = "savgol") -> np.ndarray:
    """Smooth a trajectory before differentiation.

    Parameters
    ----------
    trajectory : np.ndarray
        Trajectory of shape (T, D).
    window : int
        Smoothing window size (must be odd for savgol).
    method : str
        'savgol' for Savitzky-Golay or 'moving_average'.

    Returns
    -------
    np.ndarray
        Smoothed trajectory of same shape.
    """
    trajectory = _to_numpy(trajectory)
    if window < 3:
        return trajectory

    if method == "savgol":
        from scipy.signal import savgol_filter
        if window % 2 == 0:
            window += 1
        polyorder = min(3, window - 1)
        return savgol_filter(trajectory, window, polyorder, axis=0)

    if method == "moving_average":
        kernel = np.ones(window) / window
        smoothed = np.zeros_like(trajectory)
        for d in range(trajectory.shape[1]):
            smoothed[:, d] = np.convolve(trajectory[:, d], kernel, mode="same")
        return smoothed

    raise ValueError(f"Unknown smoothing method: {method}")


def validate_trajectory_format(sample: dict) -> bool:
    """Validate that a trajectory sample has the required fields.

    Parameters
    ----------
    sample : dict
        Trajectory sample dictionary.

    Returns
    -------
    bool
        True if valid.
    """
    required = ["q_predicted", "q_reference", "model_name"]
    for key in required:
        if key not in sample:
            return False
    q_pred = _to_numpy(sample["q_predicted"])
    q_ref = _to_numpy(sample["q_reference"])
    if q_pred.ndim < 2 or q_ref.ndim < 2:
        return False
    if q_pred.shape[-1] != q_ref.shape[-1]:
        return False
    return True


def load_analysis_config(config_path: str = "configs/trajectory_analysis.yaml") -> dict:
    """Load trajectory analysis configuration."""
    import yaml
    with open(config_path, "r") as f:
        return yaml.safe_load(f)
