"""Unit tests for trajectory analysis module."""

import numpy as np
import pytest

from src.simulation.manipulator import ThreeLinkManipulator
from src.trajectory_analysis.kinematics_metrics import (
    final_ee_orientation_error,
    final_ee_position_error,
    joint_acceleration_rmse,
    joint_angle_rmse,
    joint_velocity_rmse,
    max_path_deviation,
    path_rmse_taskspace,
)
from src.trajectory_analysis.smoothness_metrics import (
    integrated_squared_acceleration,
    integrated_squared_jerk,
    peak_acceleration,
    peak_jerk,
    smoothness_summary,
    velocity_profile,
)
from src.trajectory_analysis.energy_metrics import (
    energy_per_trajectory,
    velocity_squared_cost,
)
from src.trajectory_analysis.feasibility_checks import (
    acceleration_limit_violation_rate,
    feasibility_summary,
    joint_limit_violation_rate,
    velocity_limit_violation_rate,
)
from src.trajectory_analysis.diversity_metrics import (
    best_of_k_accuracy,
    feasible_diversity,
    pairwise_trajectory_diversity,
)
from src.trajectory_analysis.statistical_tests import (
    compute_statistics,
    confidence_interval,
    wilcoxon_signed_rank,
)
from src.trajectory_analysis.utils import (
    compute_task_space_trajectory,
    finite_difference,
    smooth_trajectory,
    validate_trajectory_format,
)


@pytest.fixture
def manipulator():
    return ThreeLinkManipulator()


@pytest.fixture
def sample_trajectory():
    """A smooth 100-step trajectory."""
    t = np.linspace(0, 2 * np.pi, 100)
    q = np.column_stack([
        0.5 * np.sin(t),
        0.3 * np.cos(t),
        0.2 * np.sin(2 * t),
    ])
    return q


@pytest.fixture
def reference_trajectory(sample_trajectory):
    return sample_trajectory


class TestUtils:
    def test_finite_difference_velocity(self, sample_trajectory):
        vel = finite_difference(sample_trajectory, dt=0.01, order=1)
        assert vel.shape == (99, 3)

    def test_finite_difference_acceleration(self, sample_trajectory):
        acc = finite_difference(sample_trajectory, dt=0.01, order=2)
        assert acc.shape == (98, 3)

    def test_finite_difference_jerk(self, sample_trajectory):
        jerk = finite_difference(sample_trajectory, dt=0.01, order=3)
        assert jerk.shape == (97, 3)

    def test_smooth_trajectory_savgol(self, sample_trajectory):
        smoothed = smooth_trajectory(sample_trajectory, window=5, method="savgol")
        assert smoothed.shape == sample_trajectory.shape

    def test_smooth_trajectory_moving_average(self, sample_trajectory):
        smoothed = smooth_trajectory(sample_trajectory, window=5, method="moving_average")
        assert smoothed.shape == sample_trajectory.shape

    def test_validate_trajectory_format_valid(self):
        sample = {
            "q_predicted": np.zeros((100, 3)),
            "q_reference": np.zeros((100, 3)),
            "model_name": "test",
        }
        assert validate_trajectory_format(sample) is True

    def test_validate_trajectory_format_missing_key(self):
        sample = {"q_predicted": np.zeros((100, 3))}
        assert validate_trajectory_format(sample) is False

    def test_compute_task_space(self, manipulator, sample_trajectory):
        x = compute_task_space_trajectory(sample_trajectory, manipulator.forward_kinematics)
        assert x.shape == (100, 3)


class TestKinematicsMetrics:
    def test_joint_angle_rmse_identical(self, sample_trajectory):
        assert joint_angle_rmse(sample_trajectory, sample_trajectory) == pytest.approx(0.0, abs=1e-10)

    def test_joint_angle_rmse_nonzero(self, sample_trajectory):
        noisy = sample_trajectory + 0.01 * np.random.randn(*sample_trajectory.shape)
        assert joint_angle_rmse(noisy, sample_trajectory) > 0

    def test_joint_velocity_rmse_identical(self, sample_trajectory):
        assert joint_velocity_rmse(sample_trajectory, sample_trajectory, dt=0.01) == pytest.approx(0.0, abs=1e-10)

    def test_joint_acceleration_rmse_identical(self, sample_trajectory):
        assert joint_acceleration_rmse(sample_trajectory, sample_trajectory, dt=0.01) == pytest.approx(0.0, abs=1e-10)

    def test_final_ee_position_error_identical(self, sample_trajectory, manipulator):
        fk = manipulator.forward_kinematics
        assert final_ee_position_error(sample_trajectory, sample_trajectory, fk) == pytest.approx(0.0, abs=1e-10)

    def test_final_ee_orientation_error_identical(self, sample_trajectory, manipulator):
        fk = manipulator.forward_kinematics
        assert final_ee_orientation_error(sample_trajectory, sample_trajectory, fk) == pytest.approx(0.0, abs=1e-10)

    def test_path_rmse_taskspace_identical(self, sample_trajectory, manipulator):
        fk = manipulator.forward_kinematics
        assert path_rmse_taskspace(sample_trajectory, sample_trajectory, fk) == pytest.approx(0.0, abs=1e-10)

    def test_max_path_deviation_identical(self, sample_trajectory, manipulator):
        fk = manipulator.forward_kinematics
        assert max_path_deviation(sample_trajectory, sample_trajectory, fk) == pytest.approx(0.0, abs=1e-10)


class TestSmoothnessMetrics:
    def test_velocity_profile_shape(self, sample_trajectory):
        vel = velocity_profile(sample_trajectory, dt=0.01)
        assert vel.shape == (99, 3)

    def test_isa_nonnegative(self, sample_trajectory):
        assert integrated_squared_acceleration(sample_trajectory, dt=0.01) >= 0

    def test_isj_nonnegative(self, sample_trajectory):
        assert integrated_squared_jerk(sample_trajectory, dt=0.01) >= 0

    def test_peak_acceleration_nonnegative(self, sample_trajectory):
        assert peak_acceleration(sample_trajectory, dt=0.01) >= 0

    def test_peak_jerk_nonnegative(self, sample_trajectory):
        assert peak_jerk(sample_trajectory, dt=0.01) >= 0

    def test_smoothness_summary_keys(self, sample_trajectory):
        s = smoothness_summary(sample_trajectory, dt=0.01)
        assert "isa" in s
        assert "isj" in s
        assert "peak_accel" in s
        assert "peak_jerk" in s


class TestEnergyMetrics:
    def test_velocity_squared_cost_nonnegative(self, sample_trajectory):
        assert velocity_squared_cost(sample_trajectory, dt=0.01) >= 0

    def test_energy_per_trajectory_nonnegative(self, sample_trajectory):
        e = energy_per_trajectory(
            sample_trajectory, dt=0.01,
            link_masses=[1.0, 1.0, 0.5],
            link_lengths=[1.0, 1.0, 0.5],
            link_com_distances=[0.5, 0.5, 0.25],
            link_inertias=[0.083, 0.083, 0.010],
            gravity=0.0,
        )
        assert e >= 0


class TestFeasibilityChecks:
    def test_joint_limit_no_violations(self, sample_trajectory):
        limits = np.array([[-np.pi, np.pi]] * 3)
        assert joint_limit_violation_rate(sample_trajectory, limits) == pytest.approx(0.0)

    def test_joint_limit_all_violations(self):
        traj = np.ones((10, 3)) * 10  # way out of bounds
        limits = np.array([[-1, 1]] * 3)
        assert joint_limit_violation_rate(traj, limits) == pytest.approx(1.0)

    def test_velocity_limit_violation(self, sample_trajectory):
        rate = velocity_limit_violation_rate(sample_trajectory, dt=0.01, velocity_limits=np.array([100, 100, 100]))
        assert rate == pytest.approx(0.0)

    def test_acceleration_limit_violation(self, sample_trajectory):
        rate = acceleration_limit_violation_rate(sample_trajectory, dt=0.01, acceleration_limits=np.array([1e6, 1e6, 1e6]))
        assert rate == pytest.approx(0.0)

    def test_feasibility_summary_keys(self, sample_trajectory):
        s = feasibility_summary(
            sample_trajectory, dt=0.01,
            joint_limits=np.array([[-np.pi, np.pi]] * 3),
            velocity_limits=np.array([100, 100, 100]),
            acceleration_limits=np.array([1e6, 1e6, 1e6]),
        )
        assert "joint_violation" in s
        assert "velocity_violation" in s
        assert "total_violation_rate" in s


class TestDiversityMetrics:
    def test_diversity_identical_samples(self, sample_trajectory):
        samples = np.stack([sample_trajectory, sample_trajectory])
        assert pairwise_trajectory_diversity(samples) == pytest.approx(0.0)

    def test_diversity_different_samples(self, sample_trajectory):
        other = sample_trajectory + 0.5
        samples = np.stack([sample_trajectory, other])
        assert pairwise_trajectory_diversity(samples) > 0

    def test_best_of_k(self, sample_trajectory):
        noisy1 = sample_trajectory + 0.1 * np.random.randn(*sample_trajectory.shape)
        noisy2 = sample_trajectory + 0.2 * np.random.randn(*sample_trajectory.shape)
        samples = np.stack([noisy1, noisy2])
        result = best_of_k_accuracy(samples, sample_trajectory, [1, 2], joint_angle_rmse)
        assert result[2] <= result[1]  # best-of-2 should be <= best-of-1


class TestStatisticalTests:
    def test_compute_statistics(self):
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        stats = compute_statistics(values)
        assert stats["mean"] == pytest.approx(3.0)
        assert stats["median"] == pytest.approx(3.0)
        assert "ci_lower" in stats
        assert "ci_upper" in stats

    def test_confidence_interval(self):
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        lower, upper = confidence_interval(values)
        assert lower < 3.0 < upper

    def test_wilcoxon_small_sample(self):
        a = np.array([1, 2, 3])
        b = np.array([4, 5, 6])
        result = wilcoxon_signed_rank(a, b)
        assert "p_value" in result
