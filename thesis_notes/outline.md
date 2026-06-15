# Thesis Outline

## Generative Models for Trajectory Generation in a 3-Link Planar Robotic Manipulator

---

## Project Snapshot (June 15, 2026)

- **Environment/testing:** Clean `.venv` workflow is in place; `python -m pytest` is runnable and the latest verified suite result was **101 passed, 1 skipped**.
- **Implemented model stack:** analytical IK, CNN, CNN+LSTM, CNN+GRU, cVAE, and conditional diffusion are all implemented with configs, training scripts, checkpoints, and core figures.
- **Current evaluation state:** the evaluation pipeline is implemented and produces plots/reports. The cVAE now has a completed **4,500-sample full-test evaluation**, while the broader cross-model comparison is still mainly represented by the earlier **6-sample stratified subset**.
- **Trajectory analysis / edge deployment:** core modules and tests exist, but thesis-grade result artifacts for these phases have not been generated yet.

### Chapter 1: Introduction
- Motivation, problem statement, research questions, contributions
- **Status:** Not started
- **Code modules:** —
- **Figures:** —

### Chapter 2: Literature Review
- Classical trajectory generation, learning-based IK, VAEs, diffusion models, evaluation metrics
- **Status:** Not started
- **Code modules:** —
- **Figures:** —

### Chapter 3: Robot Model and Data Generation ✓
- 3-link planar manipulator, DH parameters, FK/IK derivation, workspace analysis
- Trajectory shapes (6 types), dataset construction, normalization
- **Status:** Draft complete → `chapter_drafts/ch3_robot_model.md`
- **Code modules:** `src/simulation/`, `src/data/`, `configs/simulation.yaml`, `configs/data.yaml`
- **Key results:** 27,000 total samples, 6 shapes, z-score normalization
- **Figures:** arm_diagram, workspace, shape_trajectories_grid, ik_branches, dataset_distribution, per-shape trajectory/joint plots

### Chapter 4: Supervised Baseline Models ✓
- CNN, CNN+LSTM, CNN+GRU architectures, training, results
- **Status:** Draft complete → `chapter_drafts/ch4_baselines.md`
- **Code modules:** `src/models/cnn_baseline.py`, `src/models/cnn_lstm.py`, `src/models/cnn_gru.py`, `src/train/`
- **Key results:** CNN val=0.0196, CNN+LSTM val=0.0054, CNN+GRU val=0.0061
- **Figures:** `figures/baselines/training_curves.png`, `figures/baselines/comparison_bar.png`

### Chapter 5: Conditional VAE for Trajectory Generation ✓
- VAE theory, cVAE formulation, architecture, KL annealing, results
- **Status:** Draft complete → `chapter_drafts/ch5_vae.md`
- **Code modules:** `src/models/cvae.py`, `src/train/losses.py` (VAELoss), `scripts/train_vae.py`, `configs/vae.yaml`
- **Key results:** val_recon=0.000774 (best epoch 426), full-test joint RMSE=0.02012, held-out hexagon RMSE=0.01543, zero constraint violations
- **Figures:** `results/plots/cvae_full_4500_training_curves.png`, `results/plots/cvae_full_4500_reconstruction_samples.png`, `results/plots/cvae_full_4500_generation_samples.png`, `results/plots/cvae_full_4500_latent_tsne.png`, `results/plots/cvae_full_4500_evaluation_dashboard.png`

### Chapter 6: Diffusion-Based Trajectory Generation ✓
- Diffusion theory, conditioning, denoiser architecture, sampling strategies, results
- **Status:** Draft complete → `chapter_drafts/ch6_diffusion.md`
- **Code modules:** `src/models/diffusion.py`, `scripts/train_diffusion.py`, `configs/diffusion.yaml`
- **Key results:** best val loss=0.007965 (epoch 414), 1.43M params, 494 logged epochs, DDIM 75-step sampling with classifier-free guidance
- **Figures:** `figures/diffusion/training_curves.png`, `figures/diffusion/generation_samples.png`, `figures/diffusion/denoising_progression.png`

### Chapter 7: Evaluation and Comparative Analysis ✓
- Comparative evaluation, current cross-model evidence, full cVAE test results, model trade-offs
- **Status:** Draft complete → `chapter_drafts/ch7_evaluation.md`
- **Code modules:** `scripts/evaluate_models.py`, `scripts/plot_results.py`, `src/evaluate/metrics.py`, `src/evaluate/comparisons.py`, `src/evaluate/ablations.py`
- **Key results:** cVAE full-test joint RMSE=0.02012; current subset cross-model report still favors cVAE over deterministic baselines and diffusion; final uniform full-test ranking still pending
- **Figures:** `results/plots/accuracy_latency_tradeoff.png`, `results/plots/metric_grid.png`, `results/plots/generative_oracle_gap.png`, `results/plots/per_shape_joint_rmse_heatmap.png`, `results/plots/cvae_full_4500_evaluation_dashboard.png`

### Chapter 8: Conclusion and Future Work ✓
- Thesis summary, contributions, current conclusions, limitations, next steps
- **Status:** Draft complete → `chapter_drafts/ch8_conclusion.md`
- **Code modules:** Synthesizes evidence from Phases 1–7
- **Key results:** cVAE is currently the strongest learned model in the repository; final all-model ranking still requires uniform full-test evaluation

### Chapter 9: Edge Deployment and IoT Integration
- ONNX export, quantization, latency/throughput benchmarking, sensor interface, real-time loop
- **Status:** Scaffold and tests implemented; deployment artifacts pending
- **Code modules:** `src/edge_deployment/`, `scripts/export_model_to_onnx.py`, `scripts/benchmark_edge_inference.py`, `configs/edge_deployment.yaml`
- **Key results:** CPU-first deployment utilities and tests exist, but `results/checkpoints/onnx/`, `results/metrics/edge_deployment/`, and `figures/edge_deployment/` are still empty
- **Figures:** Pending Phase 8 execution

---

## Presentation
- **Status:** Updated draft available → `paper_drafts/presentation_slides.md`
- Diffusion and preliminary evaluation sections are now drafted; final slides still need the full 4,500-sample evaluation and edge-deployment results.
