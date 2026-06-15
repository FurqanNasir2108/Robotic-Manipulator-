"""Run full 4,500-sample DDIM evaluation for the diffusion model."""

from __future__ import annotations

import copy
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scripts.plot_results import generate_result_plots
from src.evaluate.comparisons import (
    evaluate_model_set,
    load_model_specs,
    load_test_dataset,
    plot_main_comparison,
    plot_trajectory_overlay,
    write_summary_files,
)
from src.utils.config import load_config


def main():
    cfg = load_config("configs/evaluation.yaml")
    cfg = copy.deepcopy(cfg)
    cfg["evaluation"]["max_test_samples"] = None
    cfg["evaluation"]["subset_strategy"] = "stratified"
    cfg["models"]["include"] = ["diffusion_ddim"]
    cfg["models"]["include_diffusion_ddpm"] = False
    cfg["paths"]["summary_json"] = os.path.join(
        "results",
        "metrics",
        "diffusion",
        "diffusion_ddim_full_4500_summary.json",
    )
    cfg["paths"]["summary_md"] = os.path.join(
        "results",
        "metrics",
        "diffusion",
        "diffusion_ddim_full_4500_summary.md",
    )
    cfg["paths"]["comparison_figure"] = os.path.join(
        "results",
        "plots",
        "diffusion_ddim_full_4500_main_comparison.png",
    )
    cfg["paths"]["overlay_figure"] = os.path.join(
        "results",
        "plots",
        "diffusion_ddim_full_4500_overlay.png",
    )
    cfg["paths"]["plots_dir"] = os.path.join(
        "results",
        "plots",
        "diffusion_ddim_full_4500",
    )

    dataset, normalizer = load_test_dataset(
        cfg["paths"]["data_dir"],
        cfg["paths"]["metadata_dir"],
        max_samples=cfg["evaluation"]["max_test_samples"],
        seed=cfg["evaluation"].get("seed", 42),
        subset_strategy=cfg["evaluation"].get("subset_strategy", "stratified"),
    )
    model_specs = load_model_specs(
        device="cpu",
        include=["diffusion_ddim"],
        include_diffusion_ddpm=False,
    )
    summary = evaluate_model_set(model_specs, dataset, normalizer, cfg)

    write_summary_files(summary, cfg["paths"]["summary_json"], cfg["paths"]["summary_md"])
    plot_main_comparison(summary, cfg["paths"]["comparison_figure"])
    plot_trajectory_overlay(summary, cfg["paths"]["overlay_figure"])
    generate_result_plots(summary, cfg["paths"]["plots_dir"])

    print(f"Full diffusion evaluation complete on {summary['num_test_samples']} test samples")
    print(f"Summary JSON: {cfg['paths']['summary_json']}")
    print("Trajectory cache: results/generated_trajectories/diffusion_ddim.npz")


if __name__ == "__main__":
    main()
