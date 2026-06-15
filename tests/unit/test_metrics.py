import time

import numpy as np

from src.evaluate.metrics import (
	constraint_violation_rate,
	diversity_score,
	ee_orientation_error,
	ee_position_error,
	energy_proxy,
	inference_time,
	joint_rmse,
	path_tracking_error,
	smoothness_jerk,
)
from src.simulation.manipulator import ThreeLinkManipulator


class SlowIdentityModel:
	def __call__(self, condition):
		time.sleep(0.001)
		return condition


def _make_linear_trajectory(scale=1.0, steps=10):
	t = np.linspace(0.0, 1.0, steps)
	return np.stack([scale * t, scale * 2.0 * t, scale * -t], axis=1)


class TestMetrics:
	def test_joint_rmse_zero_for_identical_trajectories(self):
		traj = _make_linear_trajectory()
		assert joint_rmse(traj, traj) == 0.0

	def test_joint_rmse_positive_for_offset_trajectory(self):
		traj = _make_linear_trajectory()
		pred = traj + 0.5
		assert joint_rmse(pred, traj) > 0.0

	def test_ee_position_and_orientation_error_zero_for_identical(self):
		robot = ThreeLinkManipulator()
		traj = np.array([
			[0.1, 0.2, 0.0],
			[0.2, 0.1, -0.1],
			[0.3, 0.0, -0.2],
		])
		assert ee_position_error(traj, traj, robot.forward_kinematics) == 0.0
		assert ee_orientation_error(traj, traj, robot.forward_kinematics) == 0.0

	def test_path_tracking_error_zero_for_identical_and_positive_otherwise(self):
		traj = _make_linear_trajectory()
		shifted = traj + 0.1
		assert path_tracking_error(traj, traj, method='dtw') == 0.0
		assert path_tracking_error(traj, shifted, method='dtw') > 0.0
		assert path_tracking_error(traj, shifted, method='frechet') > 0.0

	def test_energy_proxy_velocity_and_torque(self):
		linear = _make_linear_trajectory()
		curved = linear.copy()
		curved[:, 0] = curved[:, 0] ** 2
		assert energy_proxy(linear, method='velocity') > 0.0
		assert energy_proxy(curved, method='torque') > 0.0

	def test_smoothness_jerk_zero_for_linear_trajectory(self):
		linear = _make_linear_trajectory(steps=20)
		assert smoothness_jerk(linear, dt=0.1) == 0.0

	def test_smoothness_jerk_positive_for_nonlinear_trajectory(self):
		t = np.linspace(0.0, 1.0, 20)
		nonlinear = np.stack([t**3, np.sin(t), np.cos(t)], axis=1)
		assert smoothness_jerk(nonlinear, dt=0.1) > 0.0

	def test_inference_time_returns_positive_milliseconds(self):
		model = SlowIdentityModel()
		latency = inference_time(model, np.ones((4, 3)), n_runs=5)
		assert latency > 0.5

	def test_diversity_score_zero_for_identical_samples(self):
		traj = _make_linear_trajectory()
		samples = np.stack([traj, traj, traj], axis=0)
		assert diversity_score(samples, method='dtw') == 0.0

	def test_diversity_score_positive_for_different_samples(self):
		traj = _make_linear_trajectory()
		samples = np.stack([traj, traj + 0.1, traj + 0.2], axis=0)
		assert diversity_score(samples, method='dtw') > 0.0

	def test_constraint_violation_rate(self):
		traj = np.array([
			[0.0, 0.0, 0.0],
			[2.0, 0.0, 0.0],
			[0.0, -3.0, 0.0],
		])
		limits = np.array([
			[-1.0, 1.0],
			[-1.0, 1.0],
			[-1.0, 1.0],
		])
		rate = constraint_violation_rate(traj, limits)
		assert 0.0 < rate < 1.0