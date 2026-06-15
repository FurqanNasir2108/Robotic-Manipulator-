"""Generate thesis/paper-ready trajectory analysis report."""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.trajectory_analysis.trajectory_report import generate_analysis_summary_md
from scripts.plot_trajectory_analysis import load_per_trajectory_metrics


def main():
    parser = argparse.ArgumentParser(description="Generate trajectory analysis report")
    parser.add_argument("--metrics-dir", default="results/metrics/trajectory_analysis")
    parser.add_argument("--output", default="results/metrics/trajectory_analysis/analysis_summary.md")
    args = parser.parse_args()

    csv_path = os.path.join(args.metrics_dir, "per_trajectory_metrics.csv")
    if not os.path.exists(csv_path):
        print("Run analyze_trajectories.py first.")
        return

    metrics = load_per_trajectory_metrics(csv_path)
    generate_analysis_summary_md(metrics, args.output)
    print(f"Report saved to {args.output}")


if __name__ == "__main__":
    main()
