"""Trajectory analysis framework for evaluating generated trajectories."""

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
)
from src.trajectory_analysis.energy_metrics import (
    energy_per_trajectory,
    peak_power,
    positive_mechanical_work,
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

__all__ = [
    "final_ee_position_error",
    "final_ee_orientation_error",
    "path_rmse_taskspace",
    "max_path_deviation",
    "joint_angle_rmse",
    "joint_velocity_rmse",
    "joint_acceleration_rmse",
    "integrated_squared_acceleration",
    "integrated_squared_jerk",
    "peak_acceleration",
    "peak_jerk",
    "smoothness_summary",
    "velocity_squared_cost",
    "energy_per_trajectory",
    "positive_mechanical_work",
    "peak_power",
    "joint_limit_violation_rate",
    "velocity_limit_violation_rate",
    "acceleration_limit_violation_rate",
    "feasibility_summary",
    "pairwise_trajectory_diversity",
    "best_of_k_accuracy",
    "feasible_diversity",
]
