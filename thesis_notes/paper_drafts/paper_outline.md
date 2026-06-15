# Paper Outline

## Working Title

**Generative Trajectory Modeling for a 3-Link Planar Manipulator: Comparing Deterministic Baselines, Conditional VAEs, and Diffusion Models**

---

## Current Paper Status

The repository already supports a solid methods-and-results paper draft for:

- simulation and dataset generation
- deterministic baselines
- cVAE modeling
- diffusion modeling
- preliminary comparative evaluation

The final paper should wait for the **full 4,500-sample generative evaluation** before locking the main result claims.

---

## Abstract Sketch

We study trajectory generation for a 3-link planar robotic manipulator as a conditional sequence modeling problem from task-space waypoint trajectories to joint-space motion. The task exhibits a one-to-many structure because multiple valid joint trajectories can realize the same end-effector path. To address this, we build a reproducible research pipeline containing analytical inverse kinematics, deterministic neural baselines (CNN, CNN+LSTM, CNN+GRU), a conditional variational autoencoder, and a conditional diffusion model. The codebase includes data generation, training, evaluation, and figure-generation utilities, together with subset/generalization support through a held-out hexagon shape split. Current results show that the cVAE substantially improves validation reconstruction loss over deterministic baselines while enabling diverse trajectory sampling. A trained diffusion model and evaluation pipeline are also available, although the final full-test generative comparison remains pending.

---

## Proposed Paper Sections

### 1. Introduction
- Motivation for learning-based trajectory generation
- One-to-many inverse-kinematics ambiguity
- Why generative models matter for robotics trajectories

### 2. Problem Formulation
- 3-link planar manipulator
- Input/output definition
- Conditional trajectory generation setting

### 3. Dataset and Simulation Pipeline
- Analytical IK data generation
- Six trajectory-shape families
- Train/val/test split with hexagon holdout

### 4. Baseline and Generative Models
- CNN
- CNN+LSTM
- CNN+GRU
- cVAE
- Conditional diffusion

### 5. Training and Evaluation Protocol
- Loss functions
- Normalization and training setup
- Core metrics
- Best-of-K / diversity protocol for generative models

### 6. Results
- Baseline comparison
- cVAE gains over deterministic models
- Preliminary diffusion behavior
- Generalization comments

### 7. Discussion
- Accuracy-diversity-latency trade-offs
- Where cVAE currently leads
- Why diffusion remains interesting despite current speed cost

### 8. Conclusion
- Summary
- Limitations
- Future work: full analysis, edge deployment, higher-DOF extension

---

## Section Mapping to Thesis Chapters

| Paper Section | Thesis Source |
|---------------|---------------|
| Introduction | Chapters 1–2 |
| Problem + simulation | Chapter 3 |
| Baselines | Chapter 4 |
| cVAE | Chapter 5 |
| Diffusion | Chapter 6 |
| Evaluation | Chapters 7–8 |
| Conclusion | Chapter 10 |

---

## Recommended Figures

Keep the paper to **6–8 figures maximum**. The strongest currently available candidates are:

1. `figures/simulation/arm_diagram.png`
2. `figures/simulation/shape_trajectories_grid.png`
3. `figures/baselines/comparison_bar.png`
4. `figures/vae/reconstruction_samples.png`
5. `figures/vae/latent_tsne.png`
6. `figures/diffusion/denoising_progression.png`
7. `figures/comparison/main_comparison.png`
8. `results/plots/accuracy_latency_tradeoff.png`

---

## Recommended Tables

Limit to **3–4 tables**:

1. **Dataset summary**
   - shape counts
   - split sizes
   - holdout protocol

2. **Model summary**
   - model family
   - deterministic/generative
   - parameter count
   - inference mode

3. **Main comparison table**
   - RMSE
   - end-effector error
   - smoothness
   - latency
   - diversity

4. **Generalization / per-shape table**
   - especially hexagon holdout behavior

---

## Contribution Bullets

- A reproducible end-to-end pipeline for conditional trajectory generation on a 3-link planar manipulator
- A direct comparison between deterministic regressors and generative trajectory models
- A cVAE implementation that improves validation reconstruction quality while enabling multi-sample generation
- A conditional diffusion implementation with DDPM/DDIM sampling and classifier-free guidance
- An evaluation framework that supports accuracy, smoothness, energy, diversity, and latency analysis

---

## What Can Be Written Now vs. Later

### Ready now

- Methods sections for simulation, baselines, cVAE, and diffusion
- Dataset description
- Baseline results
- cVAE training/results section
- Preliminary evaluation narrative

### Should wait for final evidence

- Final main claim about the best overall model
- Final generative-model comparison table
- Strong claims about diffusion quality vs. cVAE
- Final discussion of deployment readiness
