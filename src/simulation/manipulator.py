
import numpy as np
import yaml
import os

class ThreeLinkManipulator:
	"""
	3-Link Planar Manipulator Kinematics using DH parameters.
	Supports FK, IK (all branches), Jacobian, and joint limit checking.
	"""
	def __init__(self, config_path=None):
		if config_path is None:
			config_path = os.path.join(os.path.dirname(__file__), '..', '..', 'configs', 'simulation.yaml')
		config_path = os.path.normpath(config_path)
		with open(config_path, 'r') as f:
			cfg = yaml.safe_load(f)
		self.link_lengths = np.array(cfg['link_lengths'])
		self.joint_limits = np.array([
			cfg['joint_limits']['q1'],
			cfg['joint_limits']['q2'],
			cfg['joint_limits']['q3']
		])
		self.base_position = np.array(cfg['base_position'])

	def forward_kinematics(self, q):
		"""
		Compute end-effector pose (x, y, theta) from joint angles q = [q1, q2, q3].
		"""
		l1, l2, l3 = self.link_lengths
		q1, q2, q3 = q
		x = self.base_position[0] + l1 * np.cos(q1) + l2 * np.cos(q1 + q2) + l3 * np.cos(q1 + q2 + q3)
		y = self.base_position[1] + l1 * np.sin(q1) + l2 * np.sin(q1 + q2) + l3 * np.sin(q1 + q2 + q3)
		theta = q1 + q2 + q3
		return np.array([x, y, theta])

	def inverse_kinematics(self, pose, elbow='up'):
		"""
		Compute joint angles q = [q1, q2, q3] for a given end-effector pose (x, y, theta).

		Parameters
		----------
		pose : array-like of shape (3,)
			End-effector pose (x, y, theta).
		elbow : str
			'up', 'down', or 'both' to return one or both solution branches.

		Returns
		-------
		list of ndarray
			Each element is a (3,) joint angle solution. May be empty if unreachable.
		"""
		x, y, theta = pose
		l1, l2, l3 = self.link_lengths
		# Compute wrist position
		wx = x - l3 * np.cos(theta)
		wy = y - l3 * np.sin(theta)
		# Compute q2
		dx = wx - self.base_position[0]
		dy = wy - self.base_position[1]
		D = (dx**2 + dy**2 - l1**2 - l2**2) / (2 * l1 * l2)
		if np.abs(D) > 1.0:
			return []  # No solution
		q2_options = [np.arccos(D), -np.arccos(D)]
		if elbow == 'up':
			q2_options = [q2_options[0]]
		elif elbow == 'down':
			q2_options = [q2_options[1]]
		# elbow == 'both' keeps both options
		solutions = []
		for q2 in q2_options:
			# Compute q1
			k1 = l1 + l2 * np.cos(q2)
			k2 = l2 * np.sin(q2)
			q1 = np.arctan2(dy, dx) - np.arctan2(k2, k1)
			# Compute q3
			q3 = theta - q1 - q2
			q = np.array([q1, q2, q3])
			if self.check_joint_limits(q):
				solutions.append(q)
		return solutions

	def jacobian(self, q):
		"""
		Compute the manipulator Jacobian at joint configuration q.
		Returns a 3x3 matrix mapping joint velocities to end-effector velocities.
		"""
		l1, l2, l3 = self.link_lengths
		q1, q2, q3 = q
		s1 = np.sin(q1)
		s12 = np.sin(q1 + q2)
		s123 = np.sin(q1 + q2 + q3)
		c1 = np.cos(q1)
		c12 = np.cos(q1 + q2)
		c123 = np.cos(q1 + q2 + q3)
		J = np.zeros((3, 3))
		# dx/dq
		J[0, 0] = -l1 * s1 - l2 * s12 - l3 * s123
		J[0, 1] = -l2 * s12 - l3 * s123
		J[0, 2] = -l3 * s123
		# dy/dq
		J[1, 0] = l1 * c1 + l2 * c12 + l3 * c123
		J[1, 1] = l2 * c12 + l3 * c123
		J[1, 2] = l3 * c123
		# dtheta/dq
		J[2, :] = 1.0
		return J

	def check_joint_limits(self, q):
		"""
		Check if joint angles q are within the specified joint limits.
		"""
		q = np.asarray(q)
		return np.all((q >= self.joint_limits[:, 0]) & (q <= self.joint_limits[:, 1]))