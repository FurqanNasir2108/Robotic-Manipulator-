"""Evaluation metrics for trajectory prediction and generation."""

from __future__ import annotations

import time
from typing import Callable

import numpy as np


def _to_numpy(array_like):
	"""Convert torch or numpy inputs to numpy arrays."""
	if hasattr(array_like, "detach"):
		return array_like.detach().cpu().numpy()
	return np.asarray(array_like)


def _wrap_angle(angle):
	"""Wrap angles to [-pi, pi]."""
	return (angle + np.pi) % (2 * np.pi) - np.pi


def _trajectory_point_dist(a, b):
	return float(np.linalg.norm(np.asarray(a) - np.asarray(b)))


def _dtw_distance(path_a, path_b):
	"""Compute DTW distance between two trajectories of vectors."""
	path_a = _to_numpy(path_a)
	path_b = _to_numpy(path_b)
	n_steps_a = len(path_a)
	n_steps_b = len(path_b)
	cost = np.full((n_steps_a + 1, n_steps_b + 1), np.inf, dtype=np.float64)
	cost[0, 0] = 0.0

	for i in range(1, n_steps_a + 1):
		for j in range(1, n_steps_b + 1):
			dist = _trajectory_point_dist(path_a[i - 1], path_b[j - 1])
			cost[i, j] = dist + min(cost[i - 1, j], cost[i, j - 1], cost[i - 1, j - 1])

	return float(cost[n_steps_a, n_steps_b] / max(n_steps_a + n_steps_b, 1))


def _discrete_frechet_distance(path_a, path_b):
	"""Compute discrete Frechet distance between two trajectories."""
	path_a = _to_numpy(path_a)
	path_b = _to_numpy(path_b)
	cache = np.full((len(path_a), len(path_b)), -1.0, dtype=np.float64)

	def recurse(i, j):
		if cache[i, j] >= 0:
			return cache[i, j]

		dist = _trajectory_point_dist(path_a[i], path_b[j])
		if i == 0 and j == 0:
			value = dist
		elif i > 0 and j == 0:
			value = max(recurse(i - 1, 0), dist)
		elif i == 0 and j > 0:
			value = max(recurse(0, j - 1), dist)
		else:
			value = max(
				min(recurse(i - 1, j), recurse(i - 1, j - 1), recurse(i, j - 1)),
				dist,
			)
		cache[i, j] = value
		return value

	return float(recurse(len(path_a) - 1, len(path_b) - 1))


def joint_rmse(pred, gt) -> float:
	"""Root-mean-squared error over joint trajectories."""
	pred = _to_numpy(pred)
	gt = _to_numpy(gt)
	return float(np.sqrt(np.mean((pred - gt) ** 2)))


def ee_position_error(pred, gt, fk_fn: Callable) -> float:
	"""Mean end-effector position error in task space."""
	pred = _to_numpy(pred)
	gt = _to_numpy(gt)
	pred_poses = np.asarray([fk_fn(q) for q in pred])
	gt_poses = np.asarray([fk_fn(q) for q in gt])
	return float(np.mean(np.linalg.norm(pred_poses[:, :2] - gt_poses[:, :2], axis=1)))


def ee_orientation_error(pred, gt, fk_fn: Callable) -> float:
	"""Mean wrapped orientation error in task space."""
	pred = _to_numpy(pred)
	gt = _to_numpy(gt)
	pred_theta = np.asarray([fk_fn(q)[2] for q in pred])
	gt_theta = np.asarray([fk_fn(q)[2] for q in gt])
	return float(np.mean(np.abs(_wrap_angle(pred_theta - gt_theta))))


def path_tracking_error(pred, gt, method='dtw') -> float:
	"""Trajectory path mismatch using DTW or discrete Frechet distance."""
	pred = _to_numpy(pred)
	gt = _to_numpy(gt)
	if method == 'dtw':
		return _dtw_distance(pred, gt)
	if method == 'frechet':
		return _discrete_frechet_distance(pred, gt)
	raise ValueError(f"Unknown path tracking method: {method}")


def energy_proxy(trajectory, method='velocity') -> float:
	"""Simple energy proxy based on velocity or acceleration magnitude."""
	trajectory = _to_numpy(trajectory)
	velocity = np.diff(trajectory, axis=0)
	if method == 'velocity':
		return float(np.mean(np.sum(velocity ** 2, axis=-1)))
	if method == 'torque':
		acceleration = np.diff(velocity, axis=0)
		return float(np.mean(np.sum(acceleration ** 2, axis=-1)))
	raise ValueError(f"Unknown energy method: {method}")


def smoothness_jerk(trajectory, dt) -> float:
	"""Mean squared jerk magnitude of a trajectory."""
	trajectory = _to_numpy(trajectory)
	velocity = np.diff(trajectory, axis=0) / dt
	acceleration = np.diff(velocity, axis=0) / dt
	jerk = np.diff(acceleration, axis=0) / dt
	if jerk.size == 0:
		return 0.0
	value = float(np.mean(np.sum(jerk ** 2, axis=-1)))
	if abs(value) < 1e-12:
		return 0.0
	return value


def inference_time(model, condition, n_runs=100) -> float:
	"""Median inference latency in milliseconds."""
	latencies_ms = []

	for _ in range(n_runs):
		start = time.perf_counter()
		output = model(condition)
		if hasattr(output, 'detach'):
			output = output.detach()
		end = time.perf_counter()
		latencies_ms.append((end - start) * 1000.0)

	return float(np.median(latencies_ms))


def diversity_score(samples, method='dtw') -> float:
	"""Mean pairwise diversity across multiple sampled trajectories."""
	samples = _to_numpy(samples)
	n_samples = len(samples)
	if n_samples < 2:
		return 0.0

	distances = []
	for i in range(n_samples):
		for j in range(i + 1, n_samples):
			distances.append(path_tracking_error(samples[i], samples[j], method=method))
	return float(np.mean(distances))


def constraint_violation_rate(trajectory, joint_limits) -> float:
	"""Fraction of joint values that violate joint limits."""
	trajectory = _to_numpy(trajectory)
	joint_limits = _to_numpy(joint_limits)
	lower = joint_limits[:, 0]
	upper = joint_limits[:, 1]
	violations = (trajectory < lower) | (trajectory > upper)
	return float(np.mean(violations))