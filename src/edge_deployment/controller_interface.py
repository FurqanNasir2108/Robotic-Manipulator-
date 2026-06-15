"""Controller interface for sending trajectories to simulator/controller."""

from __future__ import annotations

import numpy as np

from src.simulation.manipulator import ThreeLinkManipulator


class SimulatorController:
    """Send generated trajectories to the manipulator simulator and validate.

    Parameters
    ----------
    manipulator : ThreeLinkManipulator or None
    """

    def __init__(self, manipulator: ThreeLinkManipulator | None = None):
        if manipulator is None:
            manipulator = ThreeLinkManipulator()
        self.manipulator = manipulator

    def execute_trajectory(self, trajectory: np.ndarray) -> dict:
        """Execute a trajectory on the simulator and return execution log.

        Parameters
        ----------
        trajectory : np.ndarray of shape (T, 3)

        Returns
        -------
        dict
            Keys: success (bool), n_steps, ee_positions, joint_limit_violations.
        """
        trajectory = np.asarray(trajectory)
        ee_positions = []
        violations = 0

        for q in trajectory:
            pose = self.manipulator.forward_kinematics(q)
            ee_positions.append(pose.tolist())
            if not self.manipulator.check_joint_limits(q):
                violations += 1

        return {
            "success": violations == 0,
            "n_steps": len(trajectory),
            "ee_positions": ee_positions,
            "joint_limit_violations": violations,
            "violation_rate": float(violations / max(len(trajectory), 1)),
        }

    def validate_execution(self, planned: np.ndarray, executed: np.ndarray) -> dict:
        """Compare planned vs executed trajectory.

        Parameters
        ----------
        planned : np.ndarray of shape (T, 3)
        executed : np.ndarray of shape (T, 3)

        Returns
        -------
        dict
            Keys: joint_rmse, max_deviation.
        """
        planned = np.asarray(planned)
        executed = np.asarray(executed)
        rmse = float(np.sqrt(np.mean((planned - executed) ** 2)))
        max_dev = float(np.max(np.abs(planned - executed)))
        return {"joint_rmse": rmse, "max_deviation": max_dev}
