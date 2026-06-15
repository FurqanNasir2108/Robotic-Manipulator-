"""Integration test for trajectory analysis pipeline."""

import os
import tempfile

import numpy as np
import pytest

from src.simulation.manipulator import ThreeLinkManipulator
from src.trajectory_analysis.kinematics_metrics import joint_angle_rmse, path_rmse_taskspace
from src.trajectory_analysis.smoothness_metrics import smoothness_summary
from src.trajectory_analysis.feasibility_checks import feasibility_summary
from src.trajectory_analysis.trajectory_report import (
    generate_aggregate_json,
    generate_per_trajectory_csv,
)
from src.trajectory_analysis.utils import load_generated_trajectories


@pytest.fixture
def manipulator():
    return ThreeLinkManipulator()


@pytest.fixture
def sample_data():
    """Create sample generated trajectory data."""
    n = 20
    t = np.linspace(0, 2 * np.pi, 100)
    q_ref = np.stack([np.column_stack([
        0.5 * np.sin(t + i * 0.1),
        0.3 * np.cos(t + i * 0.1),
        0.2 * np.sin(2 * t + i * 0.1),
    ]) for i in range(n)])
    q_pred = q_ref + 0.01 * np.random.randn(*q_ref.shape)
    return {
        "q_reference": q_ref,
        "q_predicted": q_pred,
        "shape_type": np.array(["circle"] * n),
    }


class TestTrajectoryAnalysisPipeline:
    def test_full_analysis_pipeline(self, sample_data, manipulator):
        """Test end-to-end: load data → compute metrics → generate report."""
        dt = 0.01
        fk = manipulator.forward_kinematics
        joint_limits = np.array(manipulator.joint_limits)

        all_metrics = []
        for i in range(len(sample_data["q_reference"])):
            q_ref = sample_data["q_reference"][i]
            q_pred = sample_data["q_predicted"][i]

            row = {
                "sample_id": i,
                "model_name": "test_model",
                "shape_type": "circle",
                "joint_angle_rmse": joint_angle_rmse(q_pred, q_ref),
                "path_rmse_taskspace": path_rmse_taskspace(q_pred, q_ref, fk),
            }

            smooth = smoothness_summary(q_pred, dt)
            row.update({f"smooth_{k}": v for k, v in smooth.items()})

            feas = feasibility_summary(
                q_pred, dt, joint_limits,
                velocity_limits=np.array([100, 100, 100]),
                acceleration_limits=np.array([1e6, 1e6, 1e6]),
            )
            row.update({f"feas_{k}": v for k, v in feas.items()})

            all_metrics.append(row)

        assert len(all_metrics) == 20

        # Generate reports
        with tempfile.TemporaryDirectory() as tmpdir:
            generate_per_trajectory_csv(all_metrics, os.path.join(tmpdir, "metrics.csv"))
            generate_aggregate_json(all_metrics, os.path.join(tmpdir, "aggregate.json"))
            assert os.path.exists(os.path.join(tmpdir, "metrics.csv"))
            assert os.path.exists(os.path.join(tmpdir, "aggregate.json"))

    def test_load_cached_trajectories(self):
        """Test loading and saving cached trajectories."""
        with tempfile.TemporaryDirectory() as tmpdir:
            q_ref = np.random.randn(10, 100, 3)
            q_pred = np.random.randn(10, 100, 3)
            np.savez(
                os.path.join(tmpdir, "test_model.npz"),
                q_reference=q_ref,
                q_predicted=q_pred,
                shape_type=np.array(["circle"] * 10),
            )
            data = load_generated_trajectories("test_model", tmpdir)
            assert data["q_reference"].shape == (10, 100, 3)
