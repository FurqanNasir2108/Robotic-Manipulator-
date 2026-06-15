# Generative Trajectory Modeling for a 3-Link Planar Robotic Manipulator

This repository contains a thesis-grade research codebase for trajectory generation, evaluation, and deployment in a 3-link planar robotic manipulator. It combines analytical inverse kinematics, deterministic neural baselines, conditional generative models, trajectory-quality analysis, and edge-deployment tooling in one reproducible workflow.

## Overview

The project studies the mapping:

- input: end-effector pose and waypoint trajectory `(x, y, theta)`
- output: joint-space trajectory `(q1, q2, q3)`

The implemented modeling stack includes:

- analytical inverse kinematics as a reference method
- deterministic baselines: CNN, CNN+LSTM, CNN+GRU
- conditional VAE for trajectory generation
- conditional diffusion for multi-modal trajectory generation
- trajectory-analysis utilities for feasibility, smoothness, energy, and diversity
- edge-deployment utilities for ONNX export, runtime comparison, and latency profiling

## Repository Status

Based on the latest project status review in [thesis_notes/project_status.md](thesis_notes/project_status.md):

- environment and dependency setup are stabilized
- the latest verified test-suite result recorded in the repo is `101 passed, 1 skipped`
- trajectory dataset generation is implemented
- baseline, cVAE, and diffusion model training code is implemented
- cVAE full 4,500-sample evaluation has been completed
- cross-model full-test comparison, trajectory-analysis artifacts, and edge-deployment artifacts are still partially pending

Current dataset targets:

- train: `20,000`
- val: `2,500`
- test: `4,500`
- held-out hexagon generalization set: `2,000` test-only samples

## Project Structure

```text
.
├── configs/                 # YAML configs for data, training, evaluation, analysis
├── docs/                    # project guides, instruction packs, repo admin notes
├── figures/                 # generated thesis/report figures tracked in git
├── notebooks/               # standalone notebooks for training and evaluation workflows
├── scripts/                 # CLI entry points for data, training, evaluation, export
├── src/
│   ├── data/                # dataset IO, normalization, loaders
│   ├── models/              # analytical IK, CNN baselines, cVAE, diffusion
│   ├── simulation/          # manipulator kinematics and trajectory generation
│   ├── evaluate/            # evaluation and comparison utilities
│   ├── trajectory_analysis/ # quality metrics and report generation
│   ├── edge_deployment/     # ONNX export and inference tooling
│   └── train/               # trainers, schedulers, losses
├── tests/                   # unit, integration, and regression tests
└── thesis_notes/            # chapter drafts, paper drafts, and status tracking
```

## Environment Setup

### Option 1: `venv`

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

### Option 2: Conda

```bash
conda env create -f environment.yml
conda activate three-link-manipulator
```

## Common Workflows

### 1. Generate the dataset

```bash
python scripts/generate_data.py
```

### 2. Train deterministic baselines

```bash
python scripts/train_baselines.py --model all
```

Train a single model:

```bash
python scripts/train_baselines.py --model cnn
python scripts/train_baselines.py --model cnn_lstm
python scripts/train_baselines.py --model cnn_gru
```

### 3. Train generative models

```bash
python scripts/train_vae.py
python scripts/train_diffusion.py
```

### 4. Run model evaluation

```bash
python scripts/evaluate_models.py --device cpu
```

Example subset evaluation:

```bash
python scripts/evaluate_models.py --device cpu --max-samples 128 --subset-strategy stratified
```

### 5. Run trajectory analysis

```bash
python scripts/analyze_trajectories.py
python scripts/plot_trajectory_analysis.py
python scripts/generate_trajectory_report.py
```

### 6. Build evaluation summaries from cached outputs

```bash
python scripts/finalize_full_evaluation_from_caches.py
python scripts/finalize_full_evaluation_fast.py
```

### 7. Export and benchmark for edge deployment

```bash
python scripts/export_model_to_onnx.py
python scripts/benchmark_edge_inference.py
python scripts/compare_pytorch_vs_onnx.py
python scripts/profile_memory_latency.py
```

### 8. Run the end-to-end pipeline

```bash
python scripts/run_all_experiments.py
```

## Testing

Run the test suite with:

```bash
python -m pytest
```

Tests are organized into:

- `tests/unit/`
- `tests/integration/`
- `tests/regression/`

## Outputs

Tracked repository assets mainly include code, configs, docs, notebooks, and publication-style figures. Large generated artifacts are intentionally excluded from git, including:

- dataset files in `data/`
- checkpoints and metrics in `results/`
- standalone experiment output folders
- local virtual environments and cache folders

This keeps the repository lightweight while preserving the code and documentation needed to reproduce the runs locally.

## Documentation and Writing Assets

Useful entry points:

- [docs/instructions/00_main_orchestrator.instructions.md](docs/instructions/00_main_orchestrator.instructions.md)
- [docs/REPORT_GENERATION_GUIDE.md](docs/REPORT_GENERATION_GUIDE.md)
- [docs/PRESENTATION_GUIDE.md](docs/PRESENTATION_GUIDE.md)
- [thesis_notes/project_status.md](thesis_notes/project_status.md)
- [thesis_notes/chapter_drafts/](thesis_notes/chapter_drafts/)

## Current Gaps

The codebase is strong on implementation coverage, but some thesis-grade execution work is still pending:

- uniform full-test evaluation across all models
- full trajectory-analysis artifact generation
- ablation-study completion
- ONNX and deployment benchmarking artifact completion
- final repository and publication packaging

## Repository Metadata

Suggested GitHub repo metadata and branch-protection settings are documented in [docs/GITHUB_REPO_RECOMMENDATIONS.md](docs/GITHUB_REPO_RECOMMENDATIONS.md).

## License

No license file has been added yet. Choose a license explicitly before treating the repository as open for reuse.
