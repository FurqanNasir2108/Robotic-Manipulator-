# Figure Caption Bank

## Purpose

This note provides ready-to-edit caption drafts for the strongest figure assets already present in the repository. The goal is to speed up final manuscript assembly once the remaining experiments are complete.

---

## Core Captions

### `figures/simulation/arm_diagram.png`

**Draft caption:** Geometry of the 3-link planar revolute manipulator used throughout the study. The system operates in the $xy$ plane with link lengths $l_1=1.0$, $l_2=1.0$, and $l_3=0.5$, and serves as the controlled testbed for comparing deterministic and generative trajectory models.

### `figures/simulation/shape_trajectories_grid.png`

**Draft caption:** Representative task-space trajectory families used to generate the dataset, including analytic geometric shapes and random smooth curves. These trajectory families define the conditional inputs for joint-space trajectory prediction.

### `figures/simulation/ik_branches.png`

**Draft caption:** Illustration of inverse-kinematics branch ambiguity in the planar manipulator. Multiple valid joint configurations can realize the same task-space pose sequence, motivating the use of conditional generative models rather than purely deterministic regressors.

### `figures/baselines/training_curves.png`

**Draft caption:** Training and validation loss curves for the deterministic baselines. Recurrent decoders (CNN+LSTM and CNN+GRU) converge to substantially lower validation loss than the pure CNN regressor, demonstrating the importance of temporal modeling for trajectory prediction.

### `figures/vae/reconstruction_samples.png`

**Draft caption:** Example cVAE reconstructions for task-space-conditioned joint trajectories. The figure illustrates that the cVAE can recover smooth joint-space sequences while preserving the structure of the conditioning trajectory.

### `figures/vae/latent_tsne.png`

**Draft caption:** Two-dimensional visualization of the learned cVAE latent representation. The latent structure provides qualitative evidence that the model organizes multiple trajectory realizations in a meaningful conditional embedding space.

### `figures/diffusion/denoising_progression.png`

**Draft caption:** Progressive denoising behavior of the conditional diffusion model. Starting from noise, the model iteratively refines the trajectory sample toward a structured joint-space sequence conditioned on the input waypoints.

### `results/plots/accuracy_latency_tradeoff.png`

**Draft caption:** Accuracy-latency trade-off across analytical, deterministic, and generative methods using the currently completed evaluation artifacts. The figure summarizes the practical balance between predictive quality and inference cost.

---

## Conditional Captions for Final Results

Use these only after the full 4,500-sample benchmark is complete.

### `results/plots/per_shape_joint_rmse_heatmap.png`

**Draft caption:** Per-shape joint RMSE comparison across evaluated methods on the final test benchmark. The heatmap highlights where model behavior is consistent across shape families and where generalization remains challenging.

### `results/plots/generative_oracle_gap.png`

**Draft caption:** Gap between oracle and mean-sample performance for the generative models. A larger gap indicates that the model can produce strong samples but does not do so consistently across all draws.

### `results/plots/metric_grid.png`

**Draft caption:** Multi-metric comparison across evaluated methods, summarizing reconstruction, task-space tracking, smoothness, energy proxy, diversity, and latency in a single view.

---

## Bottom Line

These captions are intentionally conservative and evidence-aligned. When the final benchmark artifacts arrive, they should be updated only where the new results materially strengthen the claims.
