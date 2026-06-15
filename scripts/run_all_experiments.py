"""Run the full experiment pipeline end-to-end."""

import subprocess
import sys


def run(script, description):
    print(f"\n{'='*60}")
    print(f"  {description}")
    print(f"{'='*60}\n")
    result = subprocess.run([sys.executable, script], check=False)
    if result.returncode != 0:
        print(f"WARNING: {script} exited with code {result.returncode}")
    return result.returncode


def main():
    steps = [
        # Phase 1: Data generation
        ("scripts/generate_data.py", "Phase 1: Generate simulation data"),
        # Phase 2: Baseline training
        ("scripts/train_baselines.py", "Phase 2: Train baseline models (CNN, CNN+LSTM, CNN+GRU)"),
        # Phase 3: cVAE training
        ("scripts/train_vae.py", "Phase 3: Train conditional VAE"),
        # Phase 4: Diffusion training
        ("scripts/train_diffusion.py", "Phase 4: Train diffusion model"),
        # Phase 5 (formerly): Evaluation (caches trajectories for Phase 6)
        ("scripts/evaluate_models.py", "Phase 5: Evaluate all models"),
        # Phase 6: Trajectory analysis
        ("scripts/analyze_trajectories.py", "Phase 6: Trajectory analysis"),
        ("scripts/compare_trajectory_methods.py", "Phase 6: Compare trajectory methods"),
        ("scripts/generate_trajectory_report.py", "Phase 6: Generate trajectory report"),
        ("scripts/plot_trajectory_analysis.py", "Phase 6: Plot trajectory analysis figures"),
        # Phase 7: Evaluation plots and figures
        ("scripts/plot_results.py", "Phase 7: Plot evaluation results"),
        ("scripts/generate_thesis_figures.py", "Phase 7: Generate thesis figures"),
        # Phase 8: Edge deployment
        ("scripts/export_model_to_onnx.py", "Phase 8: Export models to ONNX"),
        ("scripts/benchmark_edge_inference.py", "Phase 8: Benchmark edge inference"),
        ("scripts/compare_pytorch_vs_onnx.py", "Phase 8: Compare PyTorch vs ONNX"),
        ("scripts/profile_memory_latency.py", "Phase 8: Profile memory and latency"),
    ]

    failed = []
    for script, description in steps:
        code = run(script, description)
        if code != 0:
            failed.append(script)

    print(f"\n{'='*60}")
    print("  Pipeline Complete")
    print(f"{'='*60}")
    if failed:
        print(f"\nFailed steps ({len(failed)}):")
        for s in failed:
            print(f"  - {s}")
    else:
        print("\nAll steps completed successfully.")


if __name__ == "__main__":
    main()