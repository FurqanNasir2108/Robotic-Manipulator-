"""Unit tests for trajectory generators."""

import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.simulation.trajectory_generator import TrajectoryGenerator


@pytest.fixture
def gen():
    return TrajectoryGenerator(num_waypoints=50)


class TestShapeGenerators:
    """Test that each shape generator produces correct output."""

    def test_circle_shape(self, gen):
        wp = gen.circle()
        assert wp.shape == (50, 3)

    def test_square_shape(self, gen):
        wp = gen.square()
        assert wp.shape == (50, 3)

    def test_pentagon_shape(self, gen):
        wp = gen.pentagon()
        assert wp.shape == (50, 3)

    def test_hexagon_shape(self, gen):
        wp = gen.hexagon()
        assert wp.shape == (50, 3)

    def test_line_shape(self, gen):
        wp = gen.line()
        assert wp.shape == (50, 3)

    def test_random_smooth_shape(self, gen):
        wp = gen.random_smooth(rng=np.random.default_rng(42))
        assert wp.shape == (50, 3)

    def test_circle_center_and_radius(self, gen):
        wp = gen.circle(center=(1.5, 0.0), radius=0.3)
        distances = np.sqrt((wp[:, 0] - 1.5) ** 2 + (wp[:, 1] - 0.0) ** 2)
        np.testing.assert_allclose(distances, 0.3, atol=1e-10)

    def test_line_endpoints(self, gen):
        wp = gen.line(start=(1.2, -0.3), end=(1.8, 0.3))
        np.testing.assert_allclose(wp[0, :2], [1.2, -0.3], atol=1e-10)
        np.testing.assert_allclose(wp[-1, :2], [1.8, 0.3], atol=0.05)


class TestIKResolution:
    """Test IK resolution along trajectories."""

    def test_circle_ik_roundtrip(self, gen):
        result = gen.generate('circle', center=(1.5, 0.0), radius=0.2)
        assert result is not None
        # Verify FK(q) matches the original waypoints
        for i in range(len(result['waypoints'])):
            pose_fk = gen.robot.forward_kinematics(result['q_sequence'][i])
            np.testing.assert_allclose(pose_fk[:2], result['waypoints'][i, :2], atol=1e-8)

    def test_infeasible_trajectory_returns_none(self, gen):
        # A circle far from the workspace should be unreachable
        result = gen.generate('circle', center=(10.0, 10.0), radius=0.1)
        assert result is None

    def test_q_continuity(self, gen):
        result = gen.generate('circle', center=(1.5, 0.0), radius=0.2)
        if result is not None:
            diffs = np.abs(np.diff(result['q_sequence'], axis=0))
            # Joint angle changes should be small between adjacent waypoints
            assert np.all(diffs < 1.0), "Joint trajectory has discontinuities"


class TestGenerateOutput:
    """Test the full generate() output dictionary."""

    def test_output_keys(self, gen):
        result = gen.generate('circle', center=(1.5, 0.0), radius=0.2)
        assert result is not None
        expected_keys = {'shape_type', 'waypoints', 'q_sequence', 'start_pose',
                         'goal_pose', 'x_sequence', 'y_sequence', 'theta_sequence',
                         'dq_sequence', 'ddq_sequence', 'energy_cost', 'smoothness_cost'}
        assert set(result.keys()) == expected_keys

    def test_output_shapes(self, gen):
        result = gen.generate('square', center=(1.5, 0.0), side_length=0.3)
        if result is not None:
            N = gen.num_waypoints
            assert result['waypoints'].shape == (N, 3)
            assert result['q_sequence'].shape == (N, 3)
            assert result['dq_sequence'].shape == (N, 3)
            assert result['ddq_sequence'].shape == (N, 3)
            assert result['x_sequence'].shape == (N,)
            assert result['y_sequence'].shape == (N,)
            assert result['theta_sequence'].shape == (N,)

    def test_energy_cost_positive(self, gen):
        result = gen.generate('line', start=(1.3, 0.0), end=(1.7, 0.0))
        if result is not None:
            assert result['energy_cost'] >= 0

    def test_all_shapes_generate(self, gen):
        shapes = ['circle', 'square', 'pentagon', 'hexagon', 'line']
        for shape in shapes:
            result = gen.generate(shape)
            assert result is not None, f"Failed to generate '{shape}' with defaults"