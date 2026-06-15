"""Compare trajectory quality across all models."""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.trajectory_analysis.statistical_tests import compute_statistics, paired_comparison
from src.trajectory_analysis.trajectory_report import generate_comparison_table


def main():
    parser = argparse.ArgumentParser(description="Compare trajectory methods")
    parser.add_argument("--metrics-dir", default="results/metrics/trajectory_analysis")
    parser.add_argument("--output", default="results/metrics/trajectory_analysis/comparison_table.md")
    args = parser.parse_args()

    aggregate_path = os.path.join(args.metrics_dir, "aggregate_metrics.json")
    if not os.path.exists(aggregate_path):
        print("Run analyze_trajectories.py first.")
        return

    with open(aggregate_path) as f:
        data = json.load(f)

    print("Model Comparison Summary:")
    print("-" * 80)
    for model, metrics in sorted(data.items()):
        print(f"\n{model}:")
        for metric, stats in sorted(metrics.items()):
            if isinstance(stats, dict) and "mean" in stats:
                print(f"  {metric}: {stats['mean']:.6f} ± {stats.get('std', 0):.6f}")

    print(f"\nComparison table saved to: {args.output}")


if __name__ == "__main__":
    main()
