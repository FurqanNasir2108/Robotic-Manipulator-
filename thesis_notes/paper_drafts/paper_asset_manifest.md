# Paper Asset Manifest

## Purpose

This file maps the existing repository assets to the sections of the research paper they can support directly. It is intended to make final manuscript assembly faster once the remaining experiments are complete.

---

## Core Figures Already Available

| Section | Asset | Why It Matters |
|---------|-------|----------------|
| Introduction / System setup | `figures/simulation/arm_diagram.png` | Clean visual introduction to the manipulator itself |
| Introduction / Dataset | `figures/simulation/shape_trajectories_grid.png` | Concise view of the six trajectory families |
| Motivation | `figures/simulation/ik_branches.png` | Direct evidence for one-to-many inverse kinematics |
| Dataset description | `figures/simulation/dataset_distribution.png` | Useful if the venue benefits from an explicit dataset distribution figure |
| Baseline results | `figures/baselines/training_curves.png` | Shows convergence behavior and recurrent-model advantage |
| Baseline comparison | `figures/baselines/comparison_bar.png` | Compact deterministic result summary |
| cVAE results | `figures/vae/training_curves.png` | Supports the cVAE training narrative |
| cVAE qualitative analysis | `figures/vae/reconstruction_samples.png` | Shows reconstruction quality visually |
| cVAE latent analysis | `figures/vae/latent_tsne.png` | Supports discussion of latent structure |
| Diffusion results | `figures/diffusion/training_curves.png` | Supports stable-training claim |
| Diffusion qualitative analysis | `figures/diffusion/generation_samples.png` | Shows sampled trajectories from the diffusion model |
| Diffusion explanation | `figures/diffusion/denoising_progression.png` | Useful for explaining iterative denoising in a paper-friendly way |
| Comparative evaluation | `results/plots/metric_grid.png` | Candidate multi-metric summary figure |
| Comparative evaluation | `results/plots/accuracy_latency_tradeoff.png` | Best current trade-off figure |
| Comparative evaluation | `results/plots/generative_oracle_gap.png` | Useful if the paper emphasizes oracle-vs-mean generative behavior |
| Comparative evaluation | `results/plots/per_shape_joint_rmse_heatmap.png` | Candidate per-shape generalization figure |

---

## Core Tables Already Supported

| Table | Evidence Source |
|-------|-----------------|
| Dataset split table | current dataset counts in repository notes and manuscript |
| Model summary table | model code plus verified parameter counts |
| Baseline validation table | baseline history JSONs |
| Preliminary comparison table | `results/metrics/summary.md` |

---

## Best Compact Figure Set for a Typical Conference Paper

If the venue only allows a small number of figures, the strongest current shortlist is:

1. `figures/simulation/arm_diagram.png`
2. `figures/simulation/shape_trajectories_grid.png`
3. `figures/baselines/training_curves.png`
4. `figures/vae/reconstruction_samples.png`
5. `figures/diffusion/denoising_progression.png`
6. `results/plots/accuracy_latency_tradeoff.png`

If one more figure is allowed, add:

7. `results/plots/per_shape_joint_rmse_heatmap.png`

---

## Assets Still Missing for a Fully Finalized Paper

The manuscript can become stronger once these artifact classes exist:

1. final 4,500-sample comparison plots for the generative models
2. trajectory-analysis figures from `figures/trajectory_analysis/`
3. edge-deployment benchmark plots from `figures/edge_deployment/`
4. any venue-specific polished composite figures built from the current raw assets

---

## Bottom Line

The repository already contains enough figures and reports to assemble a convincing methods-and-preliminary-results paper. The remaining missing visuals are tied mostly to the still-pending full evaluation and deployment phases, not to absent manuscript structure.
