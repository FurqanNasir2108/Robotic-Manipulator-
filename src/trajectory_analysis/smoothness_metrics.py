"""Smoothness metrics: velocity, acceleration, jerk profiles and integrals."""

from __future__ import annotations

import numpy as np

from src.trajectory_analysis.utils import _to_numpy, finite_difference


def velocity_profile(trajectory: np.ndarray, dt: float) -> np.ndarray:
    """Joint velocity profile via finite difference.

    Returns
    -------
    np.ndarray of shape (T-1, 3)
    """
    return finite_difference(trajectory, dt, order=1)


def acceleration_profile(trajectory: np.ndarray, dt: float) -> np.ndarray:
    """Joint acceleration profile via finite difference.

    Returns
    -------
    np.ndarray of shape (T-2, 3)
    """
    return finite_difference(trajectory, dt, order=2)


def jerk_profile(trajectory: np.ndarray, dt: float) -> np.ndarray:
    """Joint jerk profile via finite difference.

    Returns
    -------
    np.ndarray of shape (T-3, 3)
    """
    return finite_difference(trajectory, dt, order=3)


def integrated_squared_acceleration(trajectory: np.ndarray, dt: float) -> float:
    """Integrated squared acceleration (ISA) — lower is smoother.

    ISA = dt * sum( ||a(t)||^2 )
    """
    trajectory = _to_numpy(trajectory)
    accel = finite_difference(trajectory, dt, order=2)
    if accel.size == 0:
        return 0.0
    return float(dt * np.sum(accel ** 2))


def integrated_squared_jerk(trajectory: np.ndarray, dt: float) -> float:
    """Integrated squared jerk (ISJ) — lower is smoother.

    ISJ = dt * sum( ||j(t)||^2 )
    """
    trajectory = _to_numpy(trajectory)
    jerk = finite_difference(trajectory, dt, order=3)
    if jerk.size == 0:
        return 0.0
    return float(dt * np.sum(jerk ** 2))


def peak_acceleration(trajectory: np.ndarray, dt: float) -> float:
    """Maximum acceleration magnitude across all joints and time steps."""
    trajectory = _to_numpy(trajectory)
    accel = finite_difference(trajectory, dt, order=2)
    if accel.size == 0:
        return 0.0
    return float(np.max(np.abs(accel)))


def peak_jerk(trajectory: np.ndarray, dt: float) -> float:
    """Maximum jerk magnitude across all joints and time steps."""
    trajectory = _to_numpy(trajectory)
    jerk = finite_difference(trajectory, dt, order=3)
    if jerk.size == 0:
        return 0.0
    return float(np.max(np.abs(jerk)))


def smoothness_summary(trajectory: np.ndarray, dt: float) -> dict:
    """Compute all smoothness metrics in one call.

    Returns
    -------
    dict
        Keys: isa, isj, peak_accel, peak_jerk.
    """
    return {
        "isa": integrated_squared_acceleration(trajectory, dt),
        "isj": integrated_squared_jerk(trajectory, dt),
        "peak_accel": peak_acceleration(trajectory, dt),
        "peak_jerk": peak_jerk(trajectory, dt),
    }
