"""Feasibility checks: joint, velocity, acceleration, torque limit violations."""

from __future__ import annotations

import numpy as np

from src.evaluate.metrics import constraint_violation_rate
from src.trajectory_analysis.dynamics_metrics import inverse_dynamics_proxy
from src.trajectory_analysis.utils import _to_numpy, finite_difference


def joint_limit_violation_rate(trajectory: np.ndarray, joint_limits: np.ndarray) -> float:
    """Fraction of joint values violating joint position limits.

    Delegates to evaluate.metrics.constraint_violation_rate.
    """
    return constraint_violation_rate(trajectory, joint_limits)


def velocity_limit_violation_rate(trajectory: np.ndarray, dt: float,
                                  velocity_limits: np.ndarray) -> float:
    """Fraction of joint velocities violating velocity limits.

    Parameters
    ----------
    trajectory : np.ndarray of shape (T, 3)
    dt : float
    velocity_limits : np.ndarray of shape (3,) or (3, 2)
        If (3,): symmetric limits [-v, +v]. If (3,2): [lower, upper].

    Returns
    -------
    float
        Violation rate in [0, 1].
    """
    trajectory = _to_numpy(trajectory)
    vel = finite_difference(trajectory, dt, order=1)
    velocity_limits = _to_numpy(velocity_limits)

    if velocity_limits.ndim == 1:
        violations = np.abs(vel) > velocity_limits
    else:
        violations = (vel < velocity_limits[:, 0]) | (vel > velocity_limits[:, 1])

    return float(np.mean(violations))


def acceleration_limit_violation_rate(trajectory: np.ndarray, dt: float,
                                      acceleration_limits: np.ndarray) -> float:
    """Fraction of joint accelerations violating acceleration limits.

    Parameters
    ----------
    trajectory : np.ndarray of shape (T, 3)
    dt : float
    acceleration_limits : np.ndarray of shape (3,) or (3, 2)

    Returns
    -------
    float
        Violation rate in [0, 1].
    """
    trajectory = _to_numpy(trajectory)
    acc = finite_difference(trajectory, dt, order=2)
    acceleration_limits = _to_numpy(acceleration_limits)

    if acceleration_limits.ndim == 1:
        violations = np.abs(acc) > acceleration_limits
    else:
        violations = (acc < acceleration_limits[:, 0]) | (acc > acceleration_limits[:, 1])

    return float(np.mean(violations))


def torque_limit_violation_rate(trajectory: np.ndarray, dt: float,
                                torque_limits: np.ndarray,
                                link_masses: list, link_lengths: list,
                                link_com_distances: list, link_inertias: list,
                                gravity: float = 0.0) -> float:
    """Fraction of time steps where computed torque exceeds torque limits.

    Parameters
    ----------
    torque_limits : np.ndarray of shape (3,) or (3, 2)

    Returns
    -------
    float
        Violation rate in [0, 1].
    """
    trajectory = _to_numpy(trajectory)
    vel = finite_difference(trajectory, dt, order=1)
    acc = finite_difference(trajectory, dt, order=2)
    torque_limits = _to_numpy(torque_limits)
    n = len(acc)
    if n == 0:
        return 0.0

    violations = 0
    total = 0
    for i in range(n):
        tau = inverse_dynamics_proxy(
            trajectory[i + 1], vel[i], acc[i],
            link_masses, link_lengths, link_com_distances,
            link_inertias, gravity,
        )
        if torque_limits.ndim == 1:
            v = np.abs(tau) > torque_limits
        else:
            v = (tau < torque_limits[:, 0]) | (tau > torque_limits[:, 1])
        violations += int(np.sum(v))
        total += len(tau)

    return float(violations / max(total, 1))


def workspace_reachability_rate(trajectory: np.ndarray, fk_fn, workspace_radius: float) -> float:
    """Fraction of trajectory points whose EE position is within workspace bounds.

    Parameters
    ----------
    trajectory : np.ndarray of shape (T, 3)
    fk_fn : callable
    workspace_radius : float
        Maximum reachable distance from base.

    Returns
    -------
    float
        Fraction in [0, 1] of points inside workspace.
    """
    trajectory = _to_numpy(trajectory)
    inside = 0
    for q in trajectory:
        pose = fk_fn(q)
        dist = np.sqrt(pose[0] ** 2 + pose[1] ** 2)
        if dist <= workspace_radius:
            inside += 1
    return float(inside / len(trajectory))


def trajectory_success_rate(feasibility_results: list[dict], threshold: float = 0.0) -> float:
    """Fraction of trajectories with zero (or below threshold) total violations.

    Parameters
    ----------
    feasibility_results : list of dict
        Each dict from feasibility_summary.
    threshold : float
        Maximum acceptable total violation rate.

    Returns
    -------
    float
    """
    if not feasibility_results:
        return 0.0
    passed = sum(
        1 for r in feasibility_results
        if r.get("total_violation_rate", 1.0) <= threshold
    )
    return float(passed / len(feasibility_results))


def feasibility_summary(trajectory: np.ndarray, dt: float,
                        joint_limits: np.ndarray,
                        velocity_limits: np.ndarray,
                        acceleration_limits: np.ndarray) -> dict:
    """Compute all feasibility metrics for a single trajectory.

    Returns
    -------
    dict
        Keys: joint_violation, velocity_violation, acceleration_violation, total_violation_rate.
    """
    jv = joint_limit_violation_rate(trajectory, joint_limits)
    vv = velocity_limit_violation_rate(trajectory, dt, velocity_limits)
    av = acceleration_limit_violation_rate(trajectory, dt, acceleration_limits)
    total = (jv + vv + av) / 3.0
    return {
        "joint_violation": jv,
        "velocity_violation": vv,
        "acceleration_violation": av,
        "total_violation_rate": total,
    }
