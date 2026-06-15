"""
Analytical IK baseline for trajectory prediction.
Uses the manipulator's IK solver as the ground-truth reference.
Selects the elbow branch that minimizes energy cost along the trajectory.
"""

import numpy as np
from src.simulation.manipulator import ThreeLinkManipulator


class AnalyticalIKBaseline:
    """
    Analytical inverse kinematics baseline.
    Given task-space waypoints, computes joint-space trajectories using IK
    with closest-solution tracking to ensure continuity.

    Parameters
    ----------
    config_path : str or None
        Path to simulation.yaml.
    """

    def __init__(self, config_path=None):
        self.robot = ThreeLinkManipulator(config_path=config_path)

    def predict(self, waypoints):
        """
        Predict joint trajectory from task-space waypoints.

        Parameters
        ----------
        waypoints : ndarray of shape (N, 3)
            Task-space waypoints (x, y, theta).

        Returns
        -------
        q_sequence : ndarray of shape (N, 3) or None
            Joint-space trajectory, or None if unreachable.
        """
        N = len(waypoints)
        q_seq = np.zeros((N, 3))

        for i in range(N):
            solutions = self.robot.inverse_kinematics(waypoints[i], elbow='both')
            if len(solutions) == 0:
                return None
            if i == 0:
                energies = [np.sum(s ** 2) for s in solutions]
                q_seq[0] = solutions[np.argmin(energies)]
            else:
                dists = [np.linalg.norm(s - q_seq[i - 1]) for s in solutions]
                q_seq[i] = solutions[np.argmin(dists)]

        return q_seq

    def predict_batch(self, waypoints_batch):
        """
        Predict joint trajectories for a batch of waypoint sequences.

        Parameters
        ----------
        waypoints_batch : ndarray of shape (B, N, 3)

        Returns
        -------
        list of ndarray or None
        """
        return [self.predict(wp) for wp in waypoints_batch]