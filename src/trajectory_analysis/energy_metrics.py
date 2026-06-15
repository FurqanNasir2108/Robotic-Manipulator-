"""Energy metrics: velocity cost, torque cost, power, work."""

from __future__ import annotations

import numpy as np

from src.evaluate.metrics import energy_proxy
from src.trajectory_analysis.dynamics_metrics import inverse_dynamics_proxy
from src.trajectory_analysis.utils import _to_numpy, finite_difference


def velocity_squared_cost(trajectory: np.ndarray, dt: float) -> float:
    """Sum of squared joint velocities (energy proxy).

    Delegates to evaluate.metrics.energy_proxy with method='velocity'.
    """
    return energy_proxy(trajectory, method="velocity")


def torque_squared_cost(trajectory: np.ndarray, dt: float,
                        link_masses: list, link_lengths: list,
                        link_com_distances: list, link_inertias: list,
                        gravity: float = 0.0) -> float:
    """Sum of squared torques along the trajectory.

    Parameters
    ----------
    trajectory : np.ndarray of shape (T, 3)
    dt : float

    Returns
    -------
    float
        Mean squared torque magnitude.
    """
    trajectory = _to_numpy(trajectory)
    vel = finite_difference(trajectory, dt, order=1)
    acc = finite_difference(trajectory, dt, order=2)
    n = len(acc)
    if n == 0:
        return 0.0

    total = 0.0
    for i in range(n):
        tau = inverse_dynamics_proxy(
            trajectory[i + 1], vel[i], acc[i],
            link_masses, link_lengths, link_com_distances,
            link_inertias, gravity,
        )
        total += float(np.sum(tau ** 2))
    return total / n


def mechanical_power(trajectory: np.ndarray, dt: float,
                     link_masses: list, link_lengths: list,
                     link_com_distances: list, link_inertias: list,
                     gravity: float = 0.0) -> np.ndarray:
    """Mechanical power P(t) = tau(t) . qd(t) along the trajectory.

    Returns
    -------
    np.ndarray of shape (T-2,)
        Power at each time step.
    """
    trajectory = _to_numpy(trajectory)
    vel = finite_difference(trajectory, dt, order=1)
    acc = finite_difference(trajectory, dt, order=2)
    n = len(acc)
    power = np.zeros(n)
    for i in range(n):
        tau = inverse_dynamics_proxy(
            trajectory[i + 1], vel[i], acc[i],
            link_masses, link_lengths, link_com_distances,
            link_inertias, gravity,
        )
        power[i] = float(np.dot(tau, vel[i]))
    return power


def positive_mechanical_work(trajectory: np.ndarray, dt: float,
                             link_masses: list, link_lengths: list,
                             link_com_distances: list, link_inertias: list,
                             gravity: float = 0.0) -> float:
    """Total positive mechanical work (energy input to the system).

    W+ = dt * sum( max(0, P(t)) )
    """
    power = mechanical_power(trajectory, dt, link_masses, link_lengths,
                             link_com_distances, link_inertias, gravity)
    return float(dt * np.sum(np.maximum(power, 0.0)))


def peak_power(trajectory: np.ndarray, dt: float,
               link_masses: list, link_lengths: list,
               link_com_distances: list, link_inertias: list,
               gravity: float = 0.0) -> float:
    """Maximum absolute mechanical power along the trajectory."""
    power = mechanical_power(trajectory, dt, link_masses, link_lengths,
                             link_com_distances, link_inertias, gravity)
    if len(power) == 0:
        return 0.0
    return float(np.max(np.abs(power)))


def energy_per_trajectory(trajectory: np.ndarray, dt: float,
                          link_masses: list, link_lengths: list,
                          link_com_distances: list, link_inertias: list,
                          gravity: float = 0.0) -> float:
    """Total absolute energy expenditure.

    E = dt * sum( |P(t)| )
    """
    power = mechanical_power(trajectory, dt, link_masses, link_lengths,
                             link_com_distances, link_inertias, gravity)
    return float(dt * np.sum(np.abs(power)))
