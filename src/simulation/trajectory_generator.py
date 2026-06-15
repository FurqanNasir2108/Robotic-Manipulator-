"""
Trajectory generators for 6 shape types: circle, square, pentagon, hexagon,
straight line, and random smooth curves (Bezier-based).

Each generator returns N waypoints in task-space (x, y, theta) and resolves
joint-space sequences via IK with closest-solution tracking.
"""

import numpy as np
from scipy.interpolate import CubicSpline
from src.simulation.manipulator import ThreeLinkManipulator


class TrajectoryGenerator:
    """
    Generates trajectories in both task-space and joint-space for a 3-link manipulator.

    Parameters
    ----------
    config_path : str or None
        Path to simulation.yaml. If None, uses the default.
    num_waypoints : int
        Number of evenly-spaced waypoints per trajectory.
    """

    def __init__(self, config_path=None, num_waypoints=100):
        self.robot = ThreeLinkManipulator(config_path=config_path)
        self.num_waypoints = num_waypoints

    # ------------------------------------------------------------------
    # Shape generators (return task-space waypoints)
    # ------------------------------------------------------------------

    def circle(self, center=(1.5, 0.0), radius=0.3, theta_orient=0.0):
        """
        Generate a circular trajectory in task space.

        Parameters
        ----------
        center : tuple
            (cx, cy) center of the circle.
        radius : float
            Radius of the circle.
        theta_orient : float
            Fixed end-effector orientation along the path.

        Returns
        -------
        waypoints : ndarray of shape (N, 3)
            Task-space waypoints (x, y, theta).
        """
        t = np.linspace(0, 2 * np.pi, self.num_waypoints, endpoint=False)
        x = center[0] + radius * np.cos(t)
        y = center[1] + radius * np.sin(t)
        theta = np.full_like(t, theta_orient)
        return np.column_stack([x, y, theta])

    def square(self, center=(1.5, 0.0), side_length=0.4, theta_orient=0.0):
        """
        Generate a square trajectory in task space.

        Parameters
        ----------
        center : tuple
            (cx, cy) center of the square.
        side_length : float
            Side length of the square.
        theta_orient : float
            Fixed end-effector orientation along the path.

        Returns
        -------
        waypoints : ndarray of shape (N, 3)
        """
        half = side_length / 2.0
        corners = np.array([
            [center[0] - half, center[1] - half],
            [center[0] + half, center[1] - half],
            [center[0] + half, center[1] + half],
            [center[0] - half, center[1] + half],
        ])
        return self._polygon_waypoints(corners, theta_orient)

    def pentagon(self, center=(1.5, 0.0), radius=0.3, theta_orient=0.0):
        """
        Generate a regular pentagon trajectory in task space.

        Parameters
        ----------
        center : tuple
            (cx, cy) center of the pentagon.
        radius : float
            Circumradius of the pentagon.
        theta_orient : float
            Fixed end-effector orientation along the path.

        Returns
        -------
        waypoints : ndarray of shape (N, 3)
        """
        corners = self._regular_polygon_corners(center, radius, 5)
        return self._polygon_waypoints(corners, theta_orient)

    def hexagon(self, center=(1.5, 0.0), radius=0.3, theta_orient=0.0):
        """
        Generate a regular hexagon trajectory in task space.

        Parameters
        ----------
        center : tuple
            (cx, cy) center of the hexagon.
        radius : float
            Circumradius of the hexagon.
        theta_orient : float
            Fixed end-effector orientation along the path.

        Returns
        -------
        waypoints : ndarray of shape (N, 3)
        """
        corners = self._regular_polygon_corners(center, radius, 6)
        return self._polygon_waypoints(corners, theta_orient)

    def line(self, start=(1.2, -0.3), end=(1.8, 0.3), theta_orient=0.0):
        """
        Generate a straight-line trajectory in task space.

        Parameters
        ----------
        start : tuple
            (x, y) start point.
        end : tuple
            (x, y) end point.
        theta_orient : float
            Fixed end-effector orientation along the path.

        Returns
        -------
        waypoints : ndarray of shape (N, 3)
        """
        t = np.linspace(0, 1, self.num_waypoints)
        x = start[0] + t * (end[0] - start[0])
        y = start[1] + t * (end[1] - start[1])
        theta = np.full_like(t, theta_orient)
        return np.column_stack([x, y, theta])

    def random_smooth(self, num_control_points=6, workspace_center=(1.5, 0.0),
                      workspace_radius=0.4, theta_orient=0.0, rng=None):
        """
        Generate a random smooth curve trajectory using cubic-spline interpolation
        through randomly sampled control points.

        Parameters
        ----------
        num_control_points : int
            Number of random control points.
        workspace_center : tuple
            (cx, cy) center around which control points are sampled.
        workspace_radius : float
            Radius of the sampling region.
        theta_orient : float
            Fixed end-effector orientation.
        rng : np.random.Generator or None
            Random generator for reproducibility.

        Returns
        -------
        waypoints : ndarray of shape (N, 3)
        """
        if rng is None:
            rng = np.random.default_rng()
        angles = np.sort(rng.uniform(0, 2 * np.pi, num_control_points))
        radii = rng.uniform(0.1, workspace_radius, num_control_points)
        cx = workspace_center[0] + radii * np.cos(angles)
        cy = workspace_center[1] + radii * np.sin(angles)
        # Close the loop
        cx = np.append(cx, cx[0])
        cy = np.append(cy, cy[0])
        param = np.linspace(0, 1, len(cx))
        cs_x = CubicSpline(param, cx, bc_type='periodic')
        cs_y = CubicSpline(param, cy, bc_type='periodic')
        t_eval = np.linspace(0, 1, self.num_waypoints, endpoint=False)
        x = cs_x(t_eval)
        y = cs_y(t_eval)
        theta = np.full_like(x, theta_orient)
        return np.column_stack([x, y, theta])

    # ------------------------------------------------------------------
    # IK resolution along trajectory
    # ------------------------------------------------------------------

    def resolve_joint_trajectory(self, waypoints):
        """
        Convert task-space waypoints to joint-space using IK with
        closest-solution tracking to prevent discontinuities.

        For the first waypoint, the lowest-energy solution is selected.
        Subsequent waypoints select the IK solution closest to the previous one.

        Parameters
        ----------
        waypoints : ndarray of shape (N, 3)
            Task-space waypoints (x, y, theta).

        Returns
        -------
        q_sequence : ndarray of shape (N, 3) or None
            Joint-space trajectory, or None if any waypoint is unreachable.
        """
        N = len(waypoints)
        q_seq = np.zeros((N, 3))
        for i in range(N):
            pose = waypoints[i]
            # Get all solutions (both elbow branches)
            solutions = self.robot.inverse_kinematics(pose, elbow='both')
            if len(solutions) == 0:
                return None  # Unreachable waypoint
            if i == 0:
                # Pick lowest energy solution (sum of squared angles)
                energies = [np.sum(s ** 2) for s in solutions]
                q_seq[0] = solutions[np.argmin(energies)]
            else:
                # Pick solution closest to previous to avoid discontinuities
                dists = [np.linalg.norm(s - q_seq[i - 1]) for s in solutions]
                q_seq[i] = solutions[np.argmin(dists)]
        return q_seq

    def generate(self, shape_type, **kwargs):
        """
        Generate a full trajectory (task-space + joint-space) for a given shape type.

        Parameters
        ----------
        shape_type : str
            One of 'circle', 'square', 'pentagon', 'hexagon', 'line', 'random_smooth'.
        **kwargs
            Keyword arguments forwarded to the shape generator.

        Returns
        -------
        dict with keys:
            'shape_type', 'waypoints', 'q_sequence', 'start_pose', 'goal_pose',
            'dq_sequence', 'ddq_sequence', 'energy_cost', 'smoothness_cost'
            or None if trajectory is infeasible.
        """
        generators = {
            'circle': self.circle,
            'square': self.square,
            'pentagon': self.pentagon,
            'hexagon': self.hexagon,
            'line': self.line,
            'random_smooth': self.random_smooth,
        }
        if shape_type not in generators:
            raise ValueError(f"Unknown shape_type '{shape_type}'. Choose from {list(generators.keys())}")

        waypoints = generators[shape_type](**kwargs)
        q_sequence = self.resolve_joint_trajectory(waypoints)
        if q_sequence is None:
            return None

        # Compute joint velocities and accelerations via finite differences
        dt = 1.0 / (self.num_waypoints - 1) if self.num_waypoints > 1 else 1.0
        dq_sequence = np.gradient(q_sequence, dt, axis=0)
        ddq_sequence = np.gradient(dq_sequence, dt, axis=0)

        # Energy cost: sum of squared joint velocities
        energy_cost = float(np.sum(dq_sequence ** 2) * dt)
        # Smoothness cost: integral of squared jerk
        jerk = np.gradient(ddq_sequence, dt, axis=0)
        smoothness_cost = float(np.sum(jerk ** 2) * dt)

        return {
            'shape_type': shape_type,
            'waypoints': waypoints,
            'q_sequence': q_sequence,
            'start_pose': waypoints[0],
            'goal_pose': waypoints[-1],
            'x_sequence': waypoints[:, 0],
            'y_sequence': waypoints[:, 1],
            'theta_sequence': waypoints[:, 2],
            'dq_sequence': dq_sequence,
            'ddq_sequence': ddq_sequence,
            'energy_cost': energy_cost,
            'smoothness_cost': smoothness_cost,
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _regular_polygon_corners(self, center, radius, n_sides):
        """Generate vertices of a regular polygon."""
        angles = np.linspace(0, 2 * np.pi, n_sides, endpoint=False)
        cx = center[0] + radius * np.cos(angles)
        cy = center[1] + radius * np.sin(angles)
        return np.column_stack([cx, cy])

    def _polygon_waypoints(self, corners, theta_orient):
        """
        Distribute N waypoints evenly along the edges of a polygon defined by corners.
        """
        n_corners = len(corners)
        # Close the polygon
        closed = np.vstack([corners, corners[0:1]])
        # Compute cumulative edge lengths
        edge_lengths = np.linalg.norm(np.diff(closed, axis=0), axis=1)
        cum_lengths = np.concatenate([[0], np.cumsum(edge_lengths)])
        total_length = cum_lengths[-1]
        # Sample evenly along total perimeter
        t_samples = np.linspace(0, total_length, self.num_waypoints, endpoint=False)
        x = np.interp(t_samples, cum_lengths, closed[:, 0])
        y = np.interp(t_samples, cum_lengths, closed[:, 1])
        theta = np.full_like(x, theta_orient)
        return np.column_stack([x, y, theta])