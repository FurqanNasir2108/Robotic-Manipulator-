# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

---

## [Phase 7A - Remaining Models Full-Eval Notebook] - 2026-06-15

### Added
- Standalone Colab-ready notebook for the remaining-model evaluation phase:
  - `notebooks/standalone_remaining_eval_notebook.ipynb`
- Notebook builder script:
  - `scripts/build_standalone_remaining_eval_notebook.py`
- Notebook workflow covers:
  - same-protocol evaluation for `analytical_ik`, `cnn`, `cnn_lstm`, `cnn_gru`, and `diffusion_ddim`
  - stratified smoke mode for local verification
  - full-run mode for 4,500-sample Colab execution
- optional merge with the integrated full cVAE summary to create a final cross-model comparison package

### Verified
- Local smoke test passed using the project virtual environment on a 6-sample stratified subset
- Smoke-test artifacts written under `standalone_remaining_eval_run_smoke_local/`
- Generated summary includes:
  - `analytical_ik`
  - `cnn`
  - `cnn_lstm`
  - `cnn_gru`
  - `diffusion_ddim`

### Fixed
- Rebuilt the remaining-model notebook to be fully self-contained with no `src/` imports
- Embedded dataset loading, normalization, metrics, baseline models, diffusion model, evaluation, caching, and plotting logic directly inside the notebook
- Re-verified the notebook by executing its code cells locally end-to-end in smoke mode after regeneration

## [Phase 4 — Conditional Diffusion] - 2026-04-30

### Added
- Conditional Diffusion model (`src/models/diffusion.py`)
  - Linear and cosine beta noise schedules
  - 1D U-Net denoiser with sinusoidal time embedding
  - ResBlock1D, DownBlock, UpBlock with skip connections
  - DiffusionConditionEncoder: 1D-CNN on waypoints → condition embedding
  - ConditionalDiffusion: full model with forward process, training loss, DDPM/DDIM sampling
  - Classifier-free guidance (10% condition dropout during training)
  - EMA (Exponential Moving Average) weight tracking
- Diffusion training script (`scripts/train_diffusion.py`)
- Diffusion config (`configs/diffusion.yaml`)
  - 1D U-Net denoiser, channels=[64, 128, 256], T=1000, cosine schedule
  - AdamW lr=2e-4, weight_decay=0.01, cosine scheduler
  - 500 epochs, batch_size=64, early stopping patience=80
  - Classifier-free guidance with 10% dropout, scale=2.0
  - EMA decay=0.9999
- VAE thesis figures (`figures/vae/`)
  - `training_curves.png` — total, reconstruction, KL, annealing schedule
  - `reconstruction_samples.png` — GT vs reconstructed + error plots
  - `latent_tsne.png` — t-SNE of latent space colored by shape
  - `generation_samples.png` — prior-sampled trajectories in task space

### Training Results
- Model: 1,425,606 parameters
- Best val loss: **0.007965**
- Early stopped at epoch 494 (out of 500)
- Steady convergence: train 0.1388→0.0108, val 0.0493→0.0080
- Checkpoints saved at epochs 100, 200, 300, 400 + best
- EMA weights included in all checkpoints

### Fixed
- VAE figures gap: `figures/vae/` was empty, now has 4 publication-quality figures
- All 36 existing tests verified passing before Phase 4 implementation

---

## [Phase 3 — Conditional VAE] - 2026-04-29

### Added
- Conditional VAE model (`src/models/cvae.py`)
  - ConditionEncoder: 1D-CNN over waypoints → condition embedding
  - BiGRUEncoder: Bidirectional GRU → (mu, log_var) latent parameters
  - CNNEncoder: 1D-CNN alternative encoder
  - GRUDecoder: GRU-based trajectory decoder
  - CNNDecoder: 1D-CNN alternative decoder
  - ConditionalVAE: Full model with encode/decode/reparameterize/generate
- VAELoss with KL annealing (`src/train/losses.py`)
  - Reconstruction (MSE) + KL divergence + energy + smoothness penalties
  - Three KL annealing strategies: monotonic warmup, cyclical, free bits
- cVAE training script (`scripts/train_vae.py`)
- cVAE config (`configs/vae.yaml`)
  - BiGRU encoder, GRU decoder, latent_dim=16, condition_dim=64
  - Cyclical KL annealing (warmup=50, cycle=20)
  - 500 epochs, Adam lr=3e-4, cosine scheduler, early stopping patience=60

### Training Results
- Model: 637,350 parameters
- Best val reconstruction loss: **0.001590** (epoch ~226)
- Early stopped at epoch 286
- KL divergence stable at ~0.13–0.17 (no posterior collapse)
- Cyclical annealing worked as designed (KL weight oscillates 0→0.01)
- **3.4× better** than best baseline (CNN+LSTM val=0.0054)

### Thesis Documentation
- Chapter 5 draft: cVAE for Trajectory Generation (`thesis_notes/chapter_drafts/ch5_vae.md`)
  - VAE theory, ELBO, reparameterization, cVAE formulation, architecture details, loss function, KL annealing, results
- Presentation slides Section 5 (Slides 20–25) fully drafted
- Thesis outline updated: Chapter 5 marked complete

---

## [Thesis Writing — Phase 2 Docs] - 2026-04-29

### Added
- Chapter 3 draft: Robot Model and Data Generation (`thesis_notes/chapter_drafts/ch3_robot_model.md`)
  - FK/IK derivations, DH parameters, workspace analysis, trajectory design, dataset construction, normalization
- Chapter 4 draft: Supervised Baseline Models (`thesis_notes/chapter_drafts/ch4_baselines.md`)
  - CNN, CNN+LSTM, CNN+GRU architectures, loss function, training details, results analysis
- Presentation slide outline (`thesis_notes/paper_drafts/presentation_slides.md`)
  - 42-slide structure following PRESENTATION_GUIDE.md; Sections 1–4 and 8 content drafted
- Updated thesis outline (`thesis_notes/outline.md`) with chapter-by-chapter status and figure references
- Thesis figure generation script (`scripts/generate_thesis_figures.py`)
- New figures:
  - `figures/simulation/arm_diagram.png` — Labeled 3-link arm diagram
  - `figures/simulation/workspace.png` — Workspace boundary via random sampling
  - `figures/simulation/shape_trajectories_grid.png` — All 6 shape types in grid
  - `figures/simulation/ik_branches.png` — Elbow-up vs elbow-down comparison
  - `figures/simulation/dataset_distribution.png` — Samples per shape + joint angle histograms
  - `figures/baselines/training_curves.png` — Training/validation loss for all baselines
  - `figures/baselines/comparison_bar.png` — Best validation loss bar chart

---

## [Phase 2 Complete] - 2026-04-29

### Added
- CNN trajectory regressor baseline (`src/models/cnn_baseline.py`)
- CNN+LSTM trajectory regressor baseline (`src/models/cnn_lstm.py`)
- CNN+GRU trajectory regressor baseline (`src/models/cnn_gru.py`)
- Analytical IK baseline (`src/models/analytical_ik.py`)
- Generic trainer with checkpointing, early stopping, TensorBoard, AMP (`src/train/trainer.py`)
- Loss functions: MSE, L1, combined with smoothness penalty (`src/train/losses.py`)
- LR scheduler utilities (`src/train/schedulers.py`)
- PyTorch data loaders for trajectory datasets (`src/data/loaders.py`)
- Config loader utility (`src/utils/config.py`)
- Reproducibility seed utilities (`src/utils/reproducibility.py`)
- Baseline training script (`scripts/train_baselines.py`)
- YAML configs for all baselines (`configs/baseline_cnn.yaml`, `baseline_cnn_lstm.yaml`, `baseline_cnn_gru.yaml`)
- Training history saved to `results/metrics/baselines/`
- Model checkpoints saved to `results/checkpoints/`

### Training Results
- CNN: best_val_loss = 0.0196, early stopped at epoch 161 (140K params)
- CNN+LSTM: best_val_loss = 0.0054, early stopped at epoch 83 (247K params)
- CNN+GRU: best_val_loss = 0.0061, early stopped at epoch 56 (189K params)

### Fixed
- Deprecated `torch.cuda.amp` API updated to `torch.amp`
- Float64/Float32 dtype mismatch in data loader → trainer pipeline

---

## [Phase 1 Complete] - 2026-04-28

### Added
- 3-link manipulator FK/IK implementation with DH parameters (`src/simulation/manipulator.py`)
- IK with elbow-up, elbow-down, and both-branch support
- Jacobian computation for velocity/force mapping
- Trajectory generators for 6 shape types: circle, square, pentagon, hexagon, line, random smooth (`src/simulation/trajectory_generator.py`)
- Dataset construction with schema validation (`src/data/dataset.py`)
- Min-max and z-score normalization with persistent statistics (`src/data/normalization.py`)
- Dataset generation script producing 20K train / 2.5K val / 4.5K test samples (`scripts/generate_data.py`)
- Simulation visualization utilities (`src/simulation/visualization.py`)
- Sample trajectory figures for all 6 shapes in `figures/simulation/`
- Simulation config (`configs/simulation.yaml`)
- Data generation config (`configs/data.yaml`)
- 36 unit tests passing (kinematics, trajectory, dataset, normalization)

### Fixed
- YAML tab indentation in `configs/simulation.yaml`
- Config path resolution for `manipulator.py`

---

## [Unreleased]

### Added
- Project instruction pack with full documentation (`docs/copilot_repo_instruction_pack_3_link_manipulator.md`)
- Modular instruction files split by phase (`docs/instructions/`)
  - `00_main_orchestrator.instructions.md` — Central orchestrator with full context
  - `phase_1_foundation_and_simulation.instructions.md` — Simulation and data setup
  - `phase_2_baselines.instructions.md` — Supervised baseline reproduction
  - `phase_3_conditional_vae.instructions.md` — Conditional VAE implementation
  - `phase_4_diffusion.instructions.md` — Diffusion model implementation
  - `phase_5_evaluation.instructions.md` — Evaluation and comparison
  - `phase_6_packaging.instructions.md` — Thesis/paper packaging
  - `phase_T_testing.instructions.md` — Testing strategy (unit, integration, regression)
- Identified fallbacks and alternatives for all major risk areas
- Optimized folder structure with proper subfolders and `__init__.py` files
- `CHANGELOG.md` for tracking project changes
- `docs/PRESENTATION_GUIDE.md` for slide deck creation
- `docs/REPORT_GENERATION_GUIDE.md` for thesis and paper report generation

### Changed
- Folder structure optimized: added `docs/instructions/`, `figures/` subfolders, `thesis_notes/`, test hierarchy
- Simulation approach changed: pure-Python kinematics as primary (no PyBullet/MuJoCo dependency for basic operation)
- Added Flow Matching as alternative generative model
- VAE instructions now include KL annealing and posterior collapse mitigation
- Diffusion instructions now include DDIM and DPM-Solver fast sampling
- Evaluation pipeline now supports reproducible subset sampling (`stratified`, `random`, `ordered`)
- Evaluation results plotting script implemented with summary-driven comparison figures
- Generative trajectory caches now preserve full K samples alongside oracle predictions

### Fixed
- `thesis_notes/` directory was referenced in main instructions but missing from folder tree
- No `__init__.py` files were specified in original structure
- Testing was completely absent from original plan
- Evaluation latency benchmarking moved outside the per-sample loop
- Edge quantization utilities now expose the compatibility alias expected by integration tests

---

## How to Update This File

After every significant change, add an entry under `[Unreleased]` with:

```markdown
### Added / Changed / Fixed / Removed
- Brief description of what changed and why
```

When a phase is completed, move its entries to a dated version:

```markdown
## [Phase 1 Complete] - YYYY-MM-DD

### Added
- 3-link manipulator FK/IK implementation
- Trajectory generators for 6 shape types
- Dataset generation pipeline
```

### Categories

| Category | When to Use |
|----------|-------------|
| **Added** | New features, files, modules |
| **Changed** | Modifications to existing functionality |
| **Fixed** | Bug fixes |
| **Removed** | Deleted files, deprecated features |
| **Deprecated** | Features marked for future removal |
| **Security** | Vulnerability fixes |
