# Thesis Presentation Slides — Content Outline

**Title:** Generative Models for Trajectory Generation in a 3-Link Planar Robotic Manipulator

**Student:** [Name]  
**Advisor:** [Advisor Name]  
**Date:** [Date]  
**University:** [University Name]

---

## Section 1: Introduction (Slides 1–6)

### Slide 1 — Title
- Project title
- Student name, advisor, date, university logo

### Slide 2 — Motivation
- Trajectory generation is fundamental to robotic manipulation
- Classical approaches: analytical IK, optimization-based planners
- **Problem:** many valid joint trajectories can correspond to one task-space path
- Generative models are attractive because they can represent this one-to-many mapping

### Slide 3 — Problem Statement
- **Input:** task-space trajectory (sequence of end-effector poses)
- **Output:** joint-space trajectory (sequence of joint angles)
- **Challenge:** deterministic models return one solution; generative models can sample multiple valid solutions
- *Figure:* `figures/simulation/ik_branches.png`

### Slide 4 — Research Questions
1. Can generative models (cVAE, diffusion) produce diverse and valid manipulator trajectories?
2. How do they compare against supervised baselines (CNN, CNN+LSTM, CNN+GRU)?
3. What is the trade-off between trajectory quality, diversity, and inference speed?

### Slide 5 — Contributions
- Complete simulation and dataset pipeline for a 3-link planar manipulator
- Deterministic and generative model comparison in one reproducible codebase
- Holdout-shape generalization setup with hexagon reserved for test time
- Evaluation pipeline covering accuracy, smoothness, energy, diversity, and latency

### Slide 6 — Outline
- Robot Model & Data → Baselines → cVAE → Diffusion → Evaluation → Conclusion

---

## Section 2: Background (Slides 7–11)

### Slide 7 — 3-Link Planar Manipulator
- *Figure:* `figures/simulation/arm_diagram.png`
- Link lengths: $l_1 = 1.0$, $l_2 = 1.0$, $l_3 = 0.5$
- Joint limits: $q_i \in [-\pi, \pi]$
- End-effector pose: $(x, y, \theta)$

### Slide 8 — Trajectory Generation Problem
- Input: $\mathbf{W} \in \mathbb{R}^{100 \times 3}$ waypoints
- Output: $\mathbf{Q} \in \mathbb{R}^{100 \times 3}$ joint angles
- 6 shape types: circle, square, pentagon, hexagon, line, random smooth
- *Figure:* `figures/simulation/shape_trajectories_grid.png`

### Slide 9 — Supervised Baselines Overview
- CNN: convolutional feature extraction + regression head
- CNN+LSTM: CNN encoder + recurrent decoder
- CNN+GRU: CNN encoder + GRU decoder
- All are deterministic one-shot predictors

### Slide 10 — Generative Models Overview
- **cVAE:** sample latent code $z$ and decode conditioned on waypoints
- **Diffusion:** iteratively denoise Gaussian noise into a valid trajectory
- Both are conditioned on the same task-space input

### Slide 11 — Related Work
- Classical planners and analytical IK
- Learning-based IK and trajectory regression
- Generative robotics models: VAEs, diffusion, policy generation
- Gap: limited controlled comparison for multi-solution planar trajectory generation

---

## Section 3: Robot Model & Data (Slides 12–15)

### Slide 12 — Kinematics
- Forward kinematics:
  - $x = l_1\cos(q_1) + l_2\cos(q_1+q_2) + l_3\cos(q_1+q_2+q_3)$
  - $y = l_1\sin(q_1) + l_2\sin(q_1+q_2) + l_3\sin(q_1+q_2+q_3)$
- Analytical IK with elbow-up / elbow-down branches
- *Figure:* `figures/simulation/workspace.png`

### Slide 13 — Trajectory Shapes
- 6 shape types with randomized parameters
- 100 waypoints per sample
- Closest-solution IK used to maintain trajectory continuity
- *Figure:* `figures/simulation/shape_trajectories_grid.png`

### Slide 14 — Dataset
- **20,000 training** samples
- **2,500 validation** samples
- **4,500 test** samples total
- Hexagon contributes **2,000 test-only** samples for holdout generalization
- Z-score normalization from training-set statistics
- *Figure:* `figures/simulation/dataset_distribution.png`

### Slide 15 — Sample Trajectories
- Show task-space and joint-space examples
- Useful examples: circle, line, random smooth
- *Figures:* `figures/simulation/circle_trajectory.png`, `figures/simulation/circle_joints.png`

---

## Section 4: Baseline Methods (Slides 16–19)

### Slide 16 — Analytical IK Reference
- Closed-form solution used for data generation
- Not a learned model
- Serves as an oracle-style reference for feasibility and exact reconstruction

### Slide 17 — CNN Baseline Architecture
- Conv1d(3→32) → Conv1d(32→64) → Conv1d(64→128)
- Global pooling → MLP → reshape to $(100, 3)$
- Trained with endpoint-weighted MSE + smoothness regularization

### Slide 18 — CNN+LSTM / CNN+GRU
- CNN condition encoder
- Recurrent temporal decoder
- Designed to better model sequence structure than pure feedforward regression

### Slide 19 — Baseline Results

| Model | Best Val Loss | Best-Epoch Index | Logged Epochs |
|-------|--------------|------------------|---------------|
| CNN | 0.0196 | 131 | 161 |
| CNN+LSTM | **0.0054** | 53 | 83 |
| CNN+GRU | 0.0061 | 26 | 56 |

- Temporal modeling gives a clear 3–4× gain over the plain CNN
- *Figures:* `figures/baselines/training_curves.png`, `figures/baselines/comparison_bar.png`

---

## Section 5: Conditional VAE (Slides 20–25)

### Slide 20 — Why a cVAE?
- The mapping from waypoints to joint trajectories is one-to-many
- cVAE adds a latent variable $z$ to represent multiple valid solutions
- At inference: sample $z \sim \mathcal{N}(0, I)$ and decode conditioned on waypoints

### Slide 21 — Conditioning Strategy
- 1D-CNN condition encoder maps waypoints to a 64-D embedding
- Encoder sees joint trajectory + condition during training
- Decoder receives $[z; c]$ and generates the full joint sequence

### Slide 22 — Architecture
- Condition encoder: compact 1D-CNN
- Encoder: 2-layer BiGRU
- Decoder: 2-layer GRU
- Total parameters: **637,350**

### Slide 23 — Training Objective
- ELBO = reconstruction + KL + energy + smoothness penalties
- Cyclical KL annealing with warmup helps prevent posterior collapse
- Adam, cosine LR, gradient clipping

### Slide 24 — cVAE Results
- Best val reconstruction loss: **0.001590**
- Best epoch: **226**
- Logged epochs: **286**
- KL remains non-zero: latent space is used
- *Figures:* `figures/vae/training_curves.png`, `figures/vae/reconstruction_samples.png`, `figures/vae/latent_tsne.png`

### Slide 25 — cVAE vs Baselines

| Model | Type | Val Loss |
|-------|------|----------|
| CNN | Deterministic | 0.0196 |
| CNN+LSTM | Deterministic | 0.0054 |
| CNN+GRU | Deterministic | 0.0061 |
| **cVAE** | **Generative** | **0.0016** |

- cVAE improves reconstruction quality while also enabling multi-sample generation

---

## Section 6: Diffusion Model (Slides 26–32)

### Slide 26 — Diffusion Idea
- Start from Gaussian noise and iteratively denoise
- Learn $\epsilon_\theta(x_t, t, c)$ to predict noise at each step
- Conditioning variable is the waypoint sequence

### Slide 27 — Conditioning & Guidance
- Condition encoder: 1D-CNN on waypoints
- Classifier-free guidance used during inference
- Guidance scale in current config: **2.0**

### Slide 28 — Architecture
- 1D U-Net denoiser with channels `[64, 128, 256]`
- Sinusoidal timestep embedding + MLP
- Skip connections between down/up blocks
- Total parameters: **1,425,606**

### Slide 29 — Training Setup
- Cosine beta schedule
- 1000 diffusion steps
- AdamW, lr = 2e-4, batch size = 64
- EMA decay = 0.9999

### Slide 30 — Sampling
- DDPM implemented for full reverse diffusion
- DDIM implemented for faster inference
- Current evaluation uses **DDIM with 75 steps**

### Slide 31 — Diffusion Training Results
- Best validation loss: **0.007965**
- Best epoch: **414**
- Logged epochs: **494**
- *Figures:* `figures/diffusion/training_curves.png`, `figures/diffusion/generation_samples.png`

### Slide 32 — Denoising Visualization
- Show progressive refinement from noise to trajectory
- This is helpful both for explanation and debugging
- *Figure:* `figures/diffusion/denoising_progression.png`

---

## Section 7: Evaluation (Slides 33–38)

### Slide 33 — Evaluation Protocol
- Same test-set conditions for all models
- For generative models: generate **K = 10** samples per condition
- Report oracle metrics, mean metrics, diversity, and latency

### Slide 34 — Current Evaluation Status
- Evaluation code is implemented and tested
- Latest completed run is a **6-sample stratified subset**
- One sample per shape, including hexagon holdout
- Full 4,500-sample generative evaluation is still pending

### Slide 35 — Preliminary Main Comparison

| Model | Joint RMSE | Inference (ms) | Diversity |
|-------|------------|----------------|-----------|
| cnn | 0.0556 | 1.22 | N/A |
| cnn_lstm | 0.0314 | 2.55 | N/A |
| cnn_gru | 0.0384 | 27.05 | N/A |
| cvae | **0.0291** | 13.77 | **0.3211** |
| diffusion_ddim | 0.0997 | 785.66 | 0.1639 |

- Current subset evidence favors **cVAE** among learned models

### Slide 36 — Qualitative / Per-Shape Views
- Per-shape RMSE tables are generated
- Overlay and heatmap figures are available
- *Figures:* `figures/comparison/trajectory_overlay.png`, `results/plots/per_shape_joint_rmse_heatmap.png`

### Slide 37 — Trade-Off Figures
- Accuracy-latency trade-off
- Generative oracle-gap comparison
- Metric grid summary
- *Figures:* `results/plots/accuracy_latency_tradeoff.png`, `results/plots/generative_oracle_gap.png`, `results/plots/metric_grid.png`

### Slide 38 — Interpretation and Caveat
- The evaluation pipeline is no longer missing
- The key remaining gap is **execution scale**, not missing methodology
- Final thesis claims must wait for the full-test generative run

---

## Section 8: Conclusion (Slides 39–42)

### Slide 39 — Summary
- Complete code pipeline exists from simulation through training and evaluation
- cVAE is currently the strongest learned model in both validation loss and the completed subset evaluation
- Diffusion is implemented and trained, but currently incurs a large inference-time cost

### Slide 40 — Contributions
- Reproducible codebase for trajectory-generation research
- Controlled deterministic vs generative comparison
- Holdout-shape generalization setup
- Preliminary multi-metric evaluation framework

### Slide 41 — Limitations & Next Steps
- Full 4,500-sample generative evaluation still pending
- Trajectory-analysis artifact generation still pending
- Edge deployment benchmarking still pending
- Future: higher-DOF robots, stronger diffusion acceleration, hardware validation

### Slide 42 — Thank You / Q&A

---

## Key Figures Summary

| Slide | Figure | Source File |
|-------|--------|------------|
| 3 | IK branches | `figures/simulation/ik_branches.png` |
| 7 | Arm diagram | `figures/simulation/arm_diagram.png` |
| 8, 13 | Shape grid | `figures/simulation/shape_trajectories_grid.png` |
| 12 | Workspace | `figures/simulation/workspace.png` |
| 14 | Dataset distribution | `figures/simulation/dataset_distribution.png` |
| 19 | Baseline curves | `figures/baselines/training_curves.png` |
| 19 | Baseline comparison | `figures/baselines/comparison_bar.png` |
| 24 | cVAE curves / recon | `figures/vae/training_curves.png`, `figures/vae/reconstruction_samples.png` |
| 24 | cVAE latent plot | `figures/vae/latent_tsne.png` |
| 31 | Diffusion curves / samples | `figures/diffusion/training_curves.png`, `figures/diffusion/generation_samples.png` |
| 32 | Denoising progression | `figures/diffusion/denoising_progression.png` |
| 36 | Trajectory overlay | `figures/comparison/trajectory_overlay.png` |
| 37 | Trade-off plots | `results/plots/accuracy_latency_tradeoff.png`, `results/plots/generative_oracle_gap.png`, `results/plots/metric_grid.png` |
