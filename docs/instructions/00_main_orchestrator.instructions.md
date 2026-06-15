# Main Orchestrator Instructions

> **This is the single entry point.** Read this file first, then load the relevant phase file before executing any task.

---

## Project Title

**Generative Trajectory Modeling and Analysis for a 3-Link Planar Robotic Manipulator with Edge Deployment**

---

## Project Goal

Build a complete, reproducible, thesis-grade research codebase that:

1. Simulates a 3-link planar manipulator.
2. Generates trajectory datasets in joint space and task space.
3. Implements prior supervised baselines (CNN, CNN+LSTM, CNN+GRU).
4. Implements conditional VAE and diffusion-based trajectory generation.
5. Analyzes generated trajectories for physical feasibility, smoothness, energy, and diversity.
6. Evaluates all methods on accuracy, energy, smoothness, diversity, and inference time.
7. Studies edge deployment of the best model with ONNX export, quantization, and benchmarking.
8. Produces figures, logs, results, and thesis/paper-ready outputs.

---

## Context to Preserve

The student has already completed:

- 3-link planar manipulator trajectory generation
- CNN, CNN+LSTM, CNN+GRU, multi-input CNN, adaptive kernel CNN experiments
- Shape trajectories: circle, square, pentagon, hexagon
- Input: end-effector pose `(x, y, theta)` → Output: joint angles `(q1, q2, q3)`
- Comparison against analytical IK; focus on energy, accuracy, execution time

The next stage is generative modeling:

- Conditional VAE for trajectory generation
- Diffusion model (with DDIM/DPM-Solver acceleration fallback)
- Optional: Flow Matching as a modern alternative
- Conditioning on waypoints, goal pose, or shape descriptors

---

## Implementation Rules

| Rule | Detail |
|------|--------|
| Language | Python 3.10+ |
| DL Framework | PyTorch (primary), JAX (fallback for diffusion speed) |
| Style | Clean, modular, reproducible |
| Config | YAML-based configs for every component |
| Logging | TensorBoard primary; W&B optional |
| Seeds | All random seeds must be settable and logged |
| Docstrings | Required for every public function/class |
| Comments | Required where logic is non-obvious |

---

## Engineering Priorities (ranked)

1. Reproducibility
2. Clarity
3. Scientific correctness
4. Modularity
5. Extensibility to higher-DOF manipulators

---

## Phase Index — Instruction Files

| Phase | File | Purpose |
|-------|------|---------|
| 1 | `phase_1_foundation_and_simulation.instructions.md` | Repo setup, simulation, data generation |
| 2 | `phase_2_baselines.instructions.md` | Supervised baseline reproduction |
| 3 | `phase_3_conditional_vae.instructions.md` | Conditional VAE implementation |
| 4 | `phase_4_diffusion.instructions.md` | Diffusion model implementation |
| 5 | `phase_5_evaluation.instructions.md` | Evaluation, comparison, ablations |
| 6 | `phase_6_trajectory_analysis.instructions.md` | Trajectory analysis framework |
| 7 | `phase_7_evaluation.instructions.md` | Evaluation + comparison (was Phase 5) |
| 8 | `phase_8_edge_deployment.instructions.md` | Edge deployment and IoT integration |
| 9 | `phase_9_packaging.instructions.md` | Thesis, paper, presentation packaging |
| T | `phase_T_testing.instructions.md` | Unit, integration, and regression testing |

---

## Optimized Repository Structure

```text
project_root/
├── README.md
├── CHANGELOG.md
├── requirements.txt
├── environment.yml
├── setup.py
├── pyproject.toml
├── .gitignore
├── configs/
│   ├── data.yaml
│   ├── simulation.yaml
│   ├── baseline_cnn.yaml
│   ├── baseline_cnn_lstm.yaml
│   ├── baseline_cnn_gru.yaml
│   ├── vae.yaml
│   ├── diffusion.yaml
│   ├── evaluation.yaml
│   ├── trajectory_analysis.yaml
│   └── edge_deployment.yaml
├── data/
│   ├── raw/
│   ├── processed/
│   └── metadata/
├── docs/
│   ├── instructions/              # All instruction files live here
│   │   ├── 00_main_orchestrator.instructions.md
│   │   ├── phase_1_foundation_and_simulation.instructions.md
│   │   ├── phase_2_baselines.instructions.md
│   │   ├── phase_3_conditional_vae.instructions.md
│   │   ├── phase_4_diffusion.instructions.md
│   │   ├── phase_5_evaluation.instructions.md
│   │   ├── phase_6_trajectory_analysis.instructions.md
│   │   ├── phase_7_evaluation.instructions.md
│   │   ├── phase_8_edge_deployment.instructions.md
│   │   ├── phase_9_packaging.instructions.md
│   │   └── phase_T_testing.instructions.md
│   ├── 01_project_overview.md
│   ├── 02_problem_formulation.md
│   ├── 03_data_and_simulation.md
│   ├── 04_baselines.md
│   ├── 05_vae_method.md
│   ├── 06_diffusion_method.md
│   ├── 07_metrics_and_evaluation.md
│   ├── 08_experiments.md
│   ├── 09_thesis_outline.md
│   ├── 10_paper_outline.md
│   ├── 11_visualization_and_reporting.md
│   ├── 12_risk_and_limitations.md
│   ├── 13_phase_wise_tasks.md
│   ├── 14_repo_completion_checklist.md
│   ├── 15_testing_strategy.md
│   ├── PRESENTATION_GUIDE.md
│   └── REPORT_GENERATION_GUIDE.md
├── figures/
│   ├── simulation/
│   ├── baselines/
│   ├── vae/
│   ├── diffusion/
│   ├── comparison/
│   ├── trajectory_analysis/
│   └── edge_deployment/
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_baseline_analysis.ipynb
│   ├── 03_vae_analysis.ipynb
│   └── 04_diffusion_analysis.ipynb
├── results/
│   ├── logs/
│   ├── metrics/
│   ├── checkpoints/
│   ├── generated_trajectories/
│   └── plots/
├── scripts/
│   ├── generate_data.py
│   ├── train_baselines.py
│   ├── train_vae.py
│   ├── train_diffusion.py
│   ├── evaluate_models.py
│   ├── analyze_trajectories.py
│   ├── compare_trajectory_methods.py
│   ├── plot_trajectory_analysis.py
│   ├── generate_trajectory_report.py
│   ├── export_model_to_onnx.py
│   ├── benchmark_edge_inference.py
│   ├── run_edge_inference_loop.py
│   ├── compare_pytorch_vs_onnx.py
│   ├── profile_memory_latency.py
│   ├── plot_results.py
│   └── run_all_experiments.py
├── src/
│   ├── __init__.py
│   ├── simulation/
│   │   ├── __init__.py
│   │   ├── manipulator.py         # 3-link kinematics (pure Python)
│   │   ├── trajectory_generator.py
│   │   ├── physics_engine.py      # PyBullet/MuJoCo wrapper (optional)
│   │   └── visualization.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── dataset.py
│   │   ├── normalization.py
│   │   ├── augmentation.py
│   │   └── loaders.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── analytical_ik.py
│   │   ├── cnn_baseline.py
│   │   ├── cnn_lstm.py
│   │   ├── cnn_gru.py
│   │   ├── cvae.py
│   │   ├── diffusion.py
│   │   └── flow_matching.py       # Alternative generative model
│   ├── train/
│   │   ├── __init__.py
│   │   ├── trainer.py
│   │   ├── losses.py
│   │   └── schedulers.py
│   ├── evaluate/
│   │   ├── __init__.py
│   │   ├── metrics.py
│   │   ├── comparisons.py
│   │   └── ablations.py
│   ├── trajectory_analysis/
│   │   ├── __init__.py
│   │   ├── kinematics_metrics.py
│   │   ├── dynamics_metrics.py
│   │   ├── smoothness_metrics.py
│   │   ├── energy_metrics.py
│   │   ├── diversity_metrics.py
│   │   ├── feasibility_checks.py
│   │   ├── robustness_tests.py
│   │   ├── statistical_tests.py
│   │   ├── trajectory_report.py
│   │   └── utils.py
│   ├── edge_deployment/
│   │   ├── __init__.py
│   │   ├── model_export.py
│   │   ├── onnx_export.py
│   │   ├── tensorrt_export.py
│   │   ├── runtime_inference.py
│   │   ├── edge_benchmark.py
│   │   ├── sensor_interface.py
│   │   ├── communication.py
│   │   ├── controller_interface.py
│   │   ├── device_monitor.py
│   │   ├── optimization_utils.py
│   │   └── config.py
│   ├── visualization/
│   │   ├── __init__.py
│   │   ├── trajectory_plots.py
│   │   ├── training_plots.py
│   │   ├── latent_plots.py
│   │   └── result_figures.py
│   └── utils/
│       ├── __init__.py
│       ├── config.py
│       ├── reproducibility.py
│       ├── math_helpers.py
│       └── io_helpers.py
├── tests/
│   ├── __init__.py
│   ├── conftest.py
│   ├── unit/
│   │   ├── __init__.py
│   │   ├── test_kinematics.py
│   │   ├── test_trajectory_generator.py
│   │   ├── test_dataset.py
│   │   ├── test_normalization.py
│   │   ├── test_metrics.py
│   │   ├── test_losses.py
│   │   ├── test_models.py
│   │   ├── test_trajectory_analysis.py
│   │   └── test_edge_deployment.py
│   ├── integration/
│   │   ├── __init__.py
│   │   ├── test_data_pipeline.py
│   │   ├── test_training_loop.py
│   │   ├── test_evaluation_pipeline.py
│   │   ├── test_trajectory_analysis_pipeline.py
│   │   ├── test_edge_pipeline.py
│   │   └── test_realtime_loop.py
│   └── regression/
│       ├── __init__.py
│       └── test_known_trajectories.py
└── thesis_notes/
    ├── outline.md
    ├── chapter_drafts/
    └── paper_drafts/
```

### Key Structure Changes vs. Original

| Change | Rationale |
|--------|-----------|
| `docs/instructions/` subfolder | Separates Copilot instructions from project docs |
| `figures/` subfolders per method | Prevents a flat dump of hundreds of images |
| `src/simulation/physics_engine.py` | Isolates optional PyBullet/MuJoCo dependency |
| `src/models/flow_matching.py` | Adds modern alternative to diffusion |
| `src/data/augmentation.py` | Data augmentation to reduce overfitting risk |
| `tests/` with unit/integration/regression | Proper test hierarchy |
| `thesis_notes/` restored | Was in main instructions but missing from tree |
| `setup.py` + `pyproject.toml` | Makes `src/` importable as a package |
| `configs/simulation.yaml` | Simulation was not configurable before |
| `CHANGELOG.md` at root | Tracks all project changes |
| `scripts/run_all_experiments.py` | One-command full pipeline execution |

---

## Identified Fallbacks & Alternatives

### Simulation

| Risk | Original Plan | Fallback / Alternative |
|------|--------------|----------------------|
| PyBullet/MuJoCo hard to install | Use PyBullet or MuJoCo | **Primary: Pure-Python analytical kinematics** (no physics engine needed for a planar arm). Use FK/IK with DH parameters directly. Only add PyBullet if dynamics/collision checking is needed. |
| Simulation too slow for large datasets | Generate on-the-fly | Pre-generate and cache datasets to disk as `.npz` or `.h5` files |

### Data

| Risk | Original Plan | Fallback / Alternative |
|------|--------------|----------------------|
| Dataset too simple → overfitting | 6 shape types | Add noise perturbation, trajectory augmentation (time-warping, joint offset), and random smooth Bezier curves |
| Sequence length mismatch | Pad sequences | Also support variable-length via packed sequences or masking |

### VAE

| Risk | Original Plan | Fallback / Alternative |
|------|--------------|----------------------|
| Posterior collapse | Standard KL loss | Use **KL annealing** (cyclical or monotonic warmup), or **free-bits** strategy |
| Blurry/averaged trajectories | Single reconstruction loss | Add **adversarial loss** (VAE-GAN hybrid) or switch to **VQ-VAE** for discrete latent codes |
| Poor reconstruction | LSTM/GRU decoder | Try **Transformer decoder** or **1D-CNN decoder** for parallelism |

### Diffusion

| Risk | Original Plan | Fallback / Alternative |
|------|--------------|----------------------|
| Slow inference (1000 steps) | Reduce steps | Use **DDIM** (50 steps), **DPM-Solver** (10–20 steps), or **consistency distillation** |
| High compute cost | Standard DDPM | Try **Flow Matching** (single-step or few-step generation) as a lighter alternative |
| Poor conditioning | Concatenation-based | Use **classifier-free guidance** or **cross-attention conditioning** |

### Baselines

| Risk | Original Plan | Fallback / Alternative |
|------|--------------|----------------------|
| Baselines too weak for comparison | CNN/LSTM/GRU only | Add a **Transformer regressor** baseline for a stronger deterministic reference |
| Analytical IK fails for some poses | Single IK solution | Implement **all IK branches** (elbow-up/down) and select by energy cost |

### Hardware / Compute

| Risk | Original Plan | Fallback / Alternative |
|------|--------------|----------------------|
| No GPU available | Train on GPU | All models must support **CPU training** with smaller configs; add `device: auto` in configs |
| GPU OOM | Full batch | Use **gradient accumulation** and **mixed-precision (AMP)** training |

### Evaluation

| Risk | Original Plan | Fallback / Alternative |
|------|--------------|----------------------|
| Diversity metric unclear | Not defined | Use **pairwise trajectory distance** (DTW or L2) across N samples for the same condition |
| Energy proxy inaccurate | Sum of squared velocities | Also compute **torque-based proxy** using inverse dynamics if available |

---

## Required Model Families

### Baselines
1. Analytical IK (all solution branches)
2. CNN trajectory regressor
3. CNN + LSTM trajectory regressor
4. CNN + GRU trajectory regressor
5. (Optional) Transformer regressor

### Generative Models
1. Conditional VAE (with KL annealing)
2. Diffusion Model (DDPM + DDIM sampling)
3. (Optional) Flow Matching

---

## Required Evaluation Metrics

1. Joint-space RMSE
2. End-effector position error (L2)
3. End-effector orientation error (absolute angular)
4. Path tracking error (Frechet or DTW-based)
5. Energy proxy (sum of squared velocities + optional torque-based)
6. Smoothness (jerk integral)
7. Inference latency (ms per trajectory)
8. Diversity (pairwise DTW distance across N samples)
9. Constraint violation rate (joint limits, workspace bounds)

---

## Execution Protocol

1. **Read this orchestrator file first.**
2. **Read the matching phase instruction file** before starting that phase.
3. **Run tests** after each phase (see `phase_T_testing.instructions.md`).
4. **Update `CHANGELOG.md`** after every significant change.
5. **Do not skip documentation.** Every module needs docstrings.
6. **Do not overwrite the student's earlier work.** Extend it.
7. **Mark unimplemented features clearly** with `# TODO:` and `raise NotImplementedError`.
8. **Preserve the 3-link planar manipulator** as the primary testbed.

---

## Cross-References

| Document | Purpose |
|----------|---------|
| `CHANGELOG.md` | Track every change to the project |
| `docs/PRESENTATION_GUIDE.md` | Guide for creating project presentations |
| `docs/REPORT_GENERATION_GUIDE.md` | Guide for generating thesis/paper reports |
| `docs/15_testing_strategy.md` | Full testing strategy |
| `docs/14_repo_completion_checklist.md` | Final completeness checklist |
| `docs/instructions/phase_6_trajectory_analysis.instructions.md` | Trajectory analysis framework |
| `docs/instructions/phase_8_edge_deployment.instructions.md` | Edge deployment and IoT integration |
