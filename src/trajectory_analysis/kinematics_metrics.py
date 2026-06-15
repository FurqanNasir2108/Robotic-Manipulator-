"""Kinematics-based accuracy metrics for trajectory analysis."""

from __future__ import annotations

from typing import Callable

import numpy as np

from src.evaluate.metrics import ee_orientation_error, ee_position_error, joint_rmse
from src.trajectory_analysis.utils import _to_numpy, compute_task_space_trajectory, finite_difference


def final_ee_position_error(q_pred: np.ndarray, q_ref: np.ndarray, fk_fn: Callable) -> float:
    """End-effector position error at the final time step only.

    Parameters
    ----------
    q_pred : np.ndarray of shape (T, 3)
    q_ref : np.ndarray of shape (T, 3)
    fk_fn : callable
        Forward kinematics: q -> [x, y, theta].

    Returns
    -------
    float
        L2 position error at t=T.
    """
    q_pred = _to_numpy(q_pred)
    q_ref = _to_numpy(q_ref)
    pred_pose = fk_fn(q_pred[-1])
    ref_pose = fk_fn(q_ref[-1])
    return float(np.linalg.norm(pred_pose[:2] - ref_pose[:2]))


def final_ee_orientation_error(q_pred: np.ndarray, q_ref: np.ndarray, fk_fn: Callable) -> float:
    """End-effector orientation error at the final time step only.

    Parameters
    ----------
    q_pred : np.ndarray of shape (T, 3)
    q_ref : np.ndarray of shape (T, 3)
    fk_fn : callable

    Returns
    -------
    float
        Absolute angular error at t=T (radians).
    """
    q_pred = _to_numpy(q_pred)
    q_ref = _to_numpy(q_ref)
    pred_theta = fk_fn(q_pred[-1])[2]
    ref_theta = fk_fn(q_ref[-1])[2]
    error = (pred_theta - ref_theta + np.pi) % (2 * np.pi) - np.pi
    return float(np.abs(error))


def path_rmse_taskspace(q_pred: np.ndarray, q_ref: np.ndarray, fk_fn: Callable) -> float:
    """RMSE of end-effector positions along the full trajectory.

    Parameters
    ----------
    q_pred, q_ref : np.ndarray of shape (T, 3)
    fk_fn : callable

    Returns
    -------
    float
        Task-space path RMSE (meters).
    """
    x_pred = compute_task_space_trajectory(q_pred, fk_fn)
    x_ref = compute_task_space_trajectory(q_ref, fk_fn)
    return float(np.sqrt(np.mean((x_pred[:, :2] - x_ref[:, :2]) ** 2)))


def max_path_deviation(q_pred: np.ndarray, q_ref: np.ndarray, fk_fn: Callable) -> float:
    """Maximum end-effector position deviation along the trajectory.

    Parameters
    ----------
    q_pred, q_ref : np.ndarray of shape (T, 3)
    fk_fn : callable

    Returns
    -------
    float
        Maximum L2 position deviation (meters).
    """
    x_pred = compute_task_space_trajectory(q_pred, fk_fn)
    x_ref = compute_task_space_trajectory(q_ref, fk_fn)
    deviations = np.linalg.norm(x_pred[:, :2] - x_ref[:, :2], axis=1)
    return float(np.max(deviations))


def joint_angle_rmse(q_pred: np.ndarray, q_ref: np.ndarray) -> float:
    """Joint angle RMSE across the full trajectory. Delegates to evaluate.metrics."""
    return joint_rmse(_to_numpy(q_pred), _to_numpy(q_ref))


def joint_velocity_rmse(q_pred: np.ndarray, q_ref: np.ndarray, dt: float) -> float:
    """RMSE of joint velocities (finite difference).

    Parameters
    ----------
    q_pred, q_ref : np.ndarray of shape (T, 3)
    dt : float

    Returns
    -------
    float
    """
    v_pred = finite_difference(q_pred, dt, order=1)
    v_ref = finite_difference(q_ref, dt, order=1)
    return float(np.sqrt(np.mean((v_pred - v_ref) ** 2)))


def joint_acceleration_rmse(q_pred: np.ndarray, q_ref: np.ndarray, dt: float) -> float:
    """RMSE of joint accelerations (finite difference).

    Parameters
    ----------
    q_pred, q_ref : np.ndarray of shape (T, 3)
    dt : float

    Returns
    -------
    float
    """
    a_pred = finite_difference(q_pred, dt, order=2)
    a_ref = finite_difference(q_ref, dt, order=2)
    return float(np.sqrt(np.mean((a_pred - a_ref) ** 2)))
