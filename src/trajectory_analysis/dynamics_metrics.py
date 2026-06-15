"""Dynamics metrics: inverse dynamics proxy, gravity torque, inertia."""

from __future__ import annotations

import numpy as np

from src.trajectory_analysis.utils import _to_numpy


def compute_inertia_matrix(q: np.ndarray, link_masses: list, link_lengths: list,
                           link_com_distances: list, link_inertias: list) -> np.ndarray:
    """Compute 3x3 inertia matrix M(q) for a 3-link planar manipulator.

    Uses standard rigid-body formulation for planar arms.

    Parameters
    ----------
    q : np.ndarray of shape (3,)
    link_masses : list of 3 floats
    link_lengths : list of 3 floats
    link_com_distances : list of 3 floats
    link_inertias : list of 3 floats (about CoM)

    Returns
    -------
    np.ndarray of shape (3, 3)
    """
    m1, m2, m3 = link_masses
    l1, l2, l3 = link_lengths
    lc1, lc2, lc3 = link_com_distances
    I1, I2, I3 = link_inertias
    q1, q2, q3 = q

    c2 = np.cos(q2)
    c3 = np.cos(q3)
    c23 = np.cos(q2 + q3)

    # Diagonal terms
    d11 = (I1 + I2 + I3
           + m1 * lc1**2 + m2 * (l1**2 + lc2**2 + 2 * l1 * lc2 * c2)
           + m3 * (l1**2 + l2**2 + lc3**2 + 2 * l1 * l2 * c2
                   + 2 * l2 * lc3 * c3 + 2 * l1 * lc3 * c23))
    d22 = (I2 + I3
           + m2 * lc2**2 + m3 * (l2**2 + lc3**2 + 2 * l2 * lc3 * c3))
    d33 = I3 + m3 * lc3**2

    # Off-diagonal terms
    d12 = (I2 + I3
           + m2 * (lc2**2 + l1 * lc2 * c2)
           + m3 * (l2**2 + lc3**2 + l1 * l2 * c2
                   + 2 * l2 * lc3 * c3 + l1 * lc3 * c23))
    d13 = I3 + m3 * (lc3**2 + l2 * lc3 * c3 + l1 * lc3 * c23)
    d23 = I3 + m3 * (lc3**2 + l2 * lc3 * c3)

    M = np.array([
        [d11, d12, d13],
        [d12, d22, d23],
        [d13, d23, d33],
    ])
    return M


def compute_gravity_torque(q: np.ndarray, link_masses: list, link_lengths: list,
                           link_com_distances: list, gravity: float = 0.0) -> np.ndarray:
    """Compute gravity torque vector g(q) for a 3-link planar manipulator.

    Parameters
    ----------
    q : np.ndarray of shape (3,)
    gravity : float
        Gravitational acceleration. 0.0 for horizontal planar arm.

    Returns
    -------
    np.ndarray of shape (3,)
    """
    if abs(gravity) < 1e-12:
        return np.zeros(3)

    m1, m2, m3 = link_masses
    l1, l2, _ = link_lengths
    lc1, lc2, lc3 = link_com_distances
    q1, q2, q3 = q
    g = gravity

    g1 = (m1 * lc1 + m2 * l1 + m3 * l1) * g * np.cos(q1) \
         + (m2 * lc2 + m3 * l2) * g * np.cos(q1 + q2) \
         + m3 * lc3 * g * np.cos(q1 + q2 + q3)
    g2 = (m2 * lc2 + m3 * l2) * g * np.cos(q1 + q2) \
         + m3 * lc3 * g * np.cos(q1 + q2 + q3)
    g3 = m3 * lc3 * g * np.cos(q1 + q2 + q3)

    return np.array([g1, g2, g3])


def inverse_dynamics_proxy(q: np.ndarray, qd: np.ndarray, qdd: np.ndarray,
                           link_masses: list, link_lengths: list,
                           link_com_distances: list, link_inertias: list,
                           gravity: float = 0.0) -> np.ndarray:
    """Compute torque via simplified inverse dynamics: tau = M(q)*qdd + g(q).

    Coriolis/centrifugal terms are omitted for simplicity (proxy only).

    Parameters
    ----------
    q : np.ndarray of shape (3,)
    qd : np.ndarray of shape (3,)
        Joint velocities (unused in this simplified version).
    qdd : np.ndarray of shape (3,)
        Joint accelerations.

    Returns
    -------
    np.ndarray of shape (3,)
        Approximate joint torques.
    """
    M = compute_inertia_matrix(q, link_masses, link_lengths,
                               link_com_distances, link_inertias)
    g_vec = compute_gravity_torque(q, link_masses, link_lengths,
                                   link_com_distances, gravity)
    return M @ qdd + g_vec
