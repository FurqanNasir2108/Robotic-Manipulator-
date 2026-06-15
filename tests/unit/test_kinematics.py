"""Unit tests for 3-link manipulator kinematics."""

import numpy as np
import pytest
from src.simulation.manipulator import ThreeLinkManipulator


@pytest.fixture
def robot():
    return ThreeLinkManipulator()


class TestForwardKinematics:
    def test_zero_config(self, robot):
        pose = robot.forward_kinematics([0, 0, 0])
        expected_x = sum(robot.link_lengths)
        np.testing.assert_allclose(pose, [expected_x, 0.0, 0.0], atol=1e-10)

    def test_known_config(self, robot):
        q = [np.pi / 4, 0, 0]
        pose = robot.forward_kinematics(q)
        total = sum(robot.link_lengths)
        np.testing.assert_allclose(pose[0], total * np.cos(np.pi / 4), atol=1e-10)
        np.testing.assert_allclose(pose[1], total * np.sin(np.pi / 4), atol=1e-10)
        np.testing.assert_allclose(pose[2], np.pi / 4, atol=1e-10)


class TestInverseKinematics:
    def test_fk_ik_roundtrip(self, robot):
        q_orig = np.array([0.3, 0.5, -0.2])
        pose = robot.forward_kinematics(q_orig)
        solutions = robot.inverse_kinematics(pose, elbow='both')
        assert len(solutions) > 0
        matched = False
        for sol in solutions:
            if np.allclose(sol, q_orig, atol=1e-8):
                matched = True
                break
        assert matched, "Original joint angles not recovered by IK"

    def test_fk_ik_roundtrip_pose(self, robot):
        q_orig = np.array([0.3, 0.5, -0.2])
        pose = robot.forward_kinematics(q_orig)
        solutions = robot.inverse_kinematics(pose, elbow='both')
        for sol in solutions:
            pose_check = robot.forward_kinematics(sol)
            np.testing.assert_allclose(pose_check[:2], pose[:2], atol=1e-8)

    def test_unreachable_returns_empty(self, robot):
        solutions = robot.inverse_kinematics([100, 100, 0])
        assert solutions == []

    def test_elbow_up_and_down_differ(self, robot):
        pose = robot.forward_kinematics([0.3, 0.5, -0.2])
        up = robot.inverse_kinematics(pose, elbow='up')
        down = robot.inverse_kinematics(pose, elbow='down')
        if len(up) > 0 and len(down) > 0:
            assert not np.allclose(up[0], down[0])


class TestJacobian:
    def test_jacobian_shape(self, robot):
        J = robot.jacobian([0, 0, 0])
        assert J.shape == (3, 3)

    def test_jacobian_orientation_row(self, robot):
        J = robot.jacobian([0.1, 0.2, 0.3])
        np.testing.assert_allclose(J[2, :], [1, 1, 1])

    def test_jacobian_numerical(self, robot):
        q = np.array([0.3, 0.5, -0.2])
        J_analytic = robot.jacobian(q)
        eps = 1e-6
        J_numerical = np.zeros((3, 3))
        for i in range(3):
            q_plus = q.copy()
            q_plus[i] += eps
            q_minus = q.copy()
            q_minus[i] -= eps
            J_numerical[:, i] = (robot.forward_kinematics(q_plus) - robot.forward_kinematics(q_minus)) / (2 * eps)
        np.testing.assert_allclose(J_analytic, J_numerical, atol=1e-5)


class TestJointLimits:
    def test_within_limits(self, robot):
        assert robot.check_joint_limits([0, 0, 0])

    def test_outside_limits(self, robot):
        assert not robot.check_joint_limits([10, 0, 0])