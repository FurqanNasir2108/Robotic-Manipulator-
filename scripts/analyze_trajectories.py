"""Run trajectory analysis on all model outputs."""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

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
from src.trajectory_analysis.smoothness_metrics import smoothness_summary
from src.trajectory_analysis.energy_metrics import energy_per_trajectory, velocity_squared_cost
from src.trajectory_analysis.feasibility_checks import feasibility_summary
from src.trajectory_analysis.diversity_metrics import pairwise_trajectory_diversity
from src.trajectory_analysis.trajectory_report import (
    generate_aggregate_json,
    generate_analysis_summary_md,
    generate_comparison_table,
    generate_per_shape_report,
    generate_per_trajectory_csv,
)
from src.trajectory_analysis.utils import load_analysis_config, load_generated_trajectories


def analyze_model(model_name, data, manipulator, config):
    """Analyze trajectories for a single model."""
    dt = config["analysis"]["dt"]
    joint_limits = np.array(manipulator.joint_limits)
    vel_limits = np.array(config["analysis"]["velocity_limits"])
    acc_limits = np.array(config["analysis"]["acceleration_limits"])
    fk = manipulator.forward_kinematics

    dynamics_params = {
        "link_masses": config["analysis"]["link_masses"],
        "link_lengths": manipulator.link_lengths.tolist(),
        "link_com_distances": config["analysis"]["link_com_distances"],
        "link_inertias": config["analysis"]["link_inertias"],
        "gravity": config["analysis"]["gravity"],
    }

    q_ref_all = data["q_reference"]
    q_pred_all = data["q_predicted"]
    q_eval_all = data.get("q_oracle_predicted", q_pred_all)
    shape_types = data.get("shape_type", [None] * len(q_ref_all))

    results = []
    for i in range(len(q_ref_all)):
        q_ref = q_ref_all[i]
        q_pred = q_pred_all[i]
        q_pred_single = q_eval_all[i]

        row = {
            "sample_id": i,
            "model_name": model_name,
            "shape_type": str(shape_types[i]) if shape_types[i] is not None else "unknown",
            "joint_angle_rmse": joint_angle_rmse(q_pred_single, q_ref),
            "joint_velocity_rmse": joint_velocity_rmse(q_pred_single, q_ref, dt),
            "joint_acceleration_rmse": joint_acceleration_rmse(q_pred_single, q_ref, dt),
            "final_ee_pos_error": final_ee_position_error(q_pred_single, q_ref, fk),
            "final_ee_orient_error": final_ee_orientation_error(q_pred_single, q_ref, fk),
            "path_rmse_taskspace": path_rmse_taskspace(q_pred_single, q_ref, fk),
            "max_path_deviation": max_path_deviation(q_pred_single, q_ref, fk),
            "velocity_squared_cost": velocity_squared_cost(q_pred_single, dt),
            "energy_per_trajectory": energy_per_trajectory(q_pred_single, dt, **dynamics_params),
        }

        smooth = smoothness_summary(q_pred_single, dt)
        row.update({f"smooth_{k}": v for k, v in smooth.items()})

        feas = feasibility_summary(q_pred_single, dt, joint_limits, vel_limits, acc_limits)
        row.update({f"feas_{k}": v for k, v in feas.items()})

        # Diversity (generative models only)
        if q_pred.ndim == 3 and q_pred.shape[0] >= 2:
            row["diversity"] = pairwise_trajectory_diversity(q_pred)
        else:
            row["diversity"] = 0.0

        results.append(row)

    return results


def main():
    parser = argparse.ArgumentParser(description="Run trajectory analysis")
    parser.add_argument("--config", default="configs/trajectory_analysis.yaml")
    args = parser.parse_args()

    config = load_analysis_config(args.config)
    manipulator = ThreeLinkManipulator()
    models = config["analysis"]["models"]["include"]
    traj_dir = config["paths"]["generated_trajectories_dir"]
    metrics_dir = config["paths"]["metrics_dir"]
    os.makedirs(metrics_dir, exist_ok=True)

    all_metrics = []
    for model_name in models:
        try:
            data = load_generated_trajectories(model_name, traj_dir)
            results = analyze_model(model_name, data, manipulator, config)
            all_metrics.extend(results)
            print(f"Analyzed {len(results)} trajectories for {model_name}")
        except FileNotFoundError:
            print(f"Skipping {model_name}: no generated trajectories found")

    if all_metrics:
        generate_per_trajectory_csv(all_metrics, os.path.join(metrics_dir, "per_trajectory_metrics.csv"))
        generate_aggregate_json(all_metrics, os.path.join(metrics_dir, "aggregate_metrics.json"))
        generate_per_shape_report(all_metrics, os.path.join(metrics_dir, "per_shape_metrics.json"))
        generate_comparison_table(all_metrics, os.path.join(metrics_dir, "comparison_table.md"))
        generate_analysis_summary_md(all_metrics, os.path.join(metrics_dir, "analysis_summary.md"))
        print(f"Analysis complete. Results saved to {metrics_dir}")
    else:
        print("No trajectories analyzed. Run evaluation first to cache generated trajectories.")


if __name__ == "__main__":
    main()
