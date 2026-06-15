# Chapter 6: Diffusion-Based Trajectory Generation

## 6.1 Introduction

This chapter presents the second generative approach used in the project: a conditional diffusion model for joint-space trajectory generation. Unlike the cVAE in Chapter 5, which maps a sampled latent code to a trajectory in a single forward pass, the diffusion model starts from noise and iteratively denoises it into a valid trajectory conditioned on the task-space waypoint sequence.

The motivation for adding diffusion is twofold:

1. Diffusion models are often better at representing multi-modal output distributions than direct regressors.
2. The iterative denoising process provides a controllable quality-speed trade-off through different sampling strategies such as DDPM and DDIM.

---

## 6.2 Conditional Diffusion Formulation

### 6.2.1 Forward Noising Process

Let $\mathbf{Q}_0 \in \mathbb{R}^{100 \times 3}$ denote the clean joint-space trajectory. The diffusion forward process gradually corrupts it with Gaussian noise:

$$
q(\mathbf{Q}_t | \mathbf{Q}_{t-1}) = \mathcal{N}\left(\sqrt{1-\beta_t}\mathbf{Q}_{t-1}, \beta_t \mathbf{I}\right)
$$

with a cosine noise schedule over $T = 1000$ steps. As usual, the closed-form marginal is

$$
q(\mathbf{Q}_t | \mathbf{Q}_0) =
\mathcal{N}\left(\sqrt{\bar{\alpha}_t}\mathbf{Q}_0, (1-\bar{\alpha}_t)\mathbf{I}\right)
$$

where $\bar{\alpha}_t = \prod_{s=1}^{t} \alpha_s$ and $\alpha_s = 1 - \beta_s$.

### 6.2.2 Reverse Denoising Objective

The model learns a denoiser $\epsilon_\theta(\mathbf{Q}_t, t, c)$ that predicts the injected noise at timestep $t$, conditioned on the task-space trajectory embedding $c$:

$$
\mathcal{L}_{\text{diff}} = \mathbb{E}_{\mathbf{Q}_0, \epsilon, t}
\left[\left\|\epsilon - \epsilon_\theta(\mathbf{Q}_t, t, c)\right\|_2^2\right]
$$

This follows the standard $\epsilon$-prediction DDPM objective.

### 6.2.3 Conditioning and Guidance

Conditioning is based on the full waypoint sequence $\mathbf{W} \in \mathbb{R}^{100 \times 3}$. The sequence is encoded into a fixed 64-dimensional embedding using a 1D-CNN condition encoder. During training, classifier-free guidance is implemented by randomly dropping the condition with probability 0.1. At inference time, the guided prediction is

$$
\epsilon_{\text{guided}} = \epsilon_{\text{uncond}} + w(\epsilon_{\text{cond}} - \epsilon_{\text{uncond}})
$$

with guidance scale $w = 2.0$ in the current configuration.

---

## 6.3 Architecture

### 6.3.1 Condition Encoder

The condition encoder mirrors the design used in the cVAE:

| Layer | Configuration | Output Shape |
|-------|--------------|-------------|
| Conv1d + BN + ReLU + Dropout | 3 → 32, kernel=5 | $(32, 100)$ |
| Conv1d + BN + ReLU + Dropout | 32 → 64, kernel=3 | $(64, 100)$ |
| AdaptiveAvgPool1d | — | $(64, 1)$ |
| Linear | 64 → 64 | $(64,)$ |

This produces the condition embedding $c \in \mathbb{R}^{64}$.

### 6.3.2 1D U-Net Denoiser

The denoiser operates on noisy trajectories represented as sequences of joint angles with channel-first layout. The implemented network is a compact 1D U-Net with residual blocks, skip connections, and sinusoidal timestep embeddings.

| Component | Configuration |
|-----------|--------------|
| Time embedding | Sinusoidal + MLP, dim=128 |
| Input channels | 3 noisy joints + projected condition channels |
| Encoder channels | [64, 128, 256] |
| Bottleneck | Residual block at 256 channels |
| Decoder | Symmetric upsampling with skip connections |
| Output | Predicted noise with shape $(3, 100)$ |

The condition embedding is projected back to the joint dimension and concatenated with the noisy trajectory at every timestep.

### 6.3.3 Model Size

The full conditional diffusion model contains **1,425,606 trainable parameters**. This makes it larger than both the deterministic baselines and the cVAE, which is expected given the U-Net-style denoiser and conditioning pathway.

---

## 6.4 Training Configuration

The model is trained with the following configuration:

| Parameter | Value |
|-----------|-------|
| Diffusion steps | 1000 |
| Beta schedule | Cosine |
| Optimizer | AdamW |
| Learning rate | 0.0002 |
| Weight decay | 0.01 |
| Batch size | 64 |
| Max epochs | 500 |
| Early stopping patience | 80 |
| Gradient clipping | 1.0 |
| EMA decay | 0.9999 |
| Guidance dropout | 0.1 |
| Random seed | 42 |

For accelerated inference, the sampling configuration currently uses **DDIM with 75 denoising steps**. Full DDPM sampling is implemented as well, but DDIM is the practical default because of its lower inference cost.

---

## 6.5 Training Results

### 6.5.1 Convergence

The trained model reached:

| Metric | Value |
|--------|-------|
| Best validation loss | **0.007965** |
| Best epoch | **414** |
| Logged epochs | **494** |

The training curves in `figures/diffusion/training_curves.png` show a clear downward trend with relatively stable late-stage convergence. The presence of a saved EMA checkpoint in `results/checkpoints/diffusion/diffusion_best.pth` indicates that the final sampling path uses smoothed weights rather than raw online weights.

### 6.5.2 Qualitative Outputs

Three thesis-ready figure assets already exist for this chapter:

1. `figures/diffusion/training_curves.png`
2. `figures/diffusion/generation_samples.png`
3. `figures/diffusion/denoising_progression.png`

These cover optimization behavior, sample quality, and the denoising trajectory from noisy initialization toward the final predicted joint-space trajectory.

---

## 6.6 Preliminary Evaluation Status

The diffusion model is implemented and trained, but its final project-wide evaluation is not fully complete yet. Unlike the cVAE, which now has a completed integrated 4,500-sample test evaluation, the diffusion model is still represented mainly by the earlier **6-sample stratified subset** cross-model report. On that preliminary subset, the DDIM sampler achieved:

| Metric | Value |
|--------|-------|
| Oracle joint RMSE | 0.0997 |
| Mean joint RMSE across K samples | 0.2629 |
| Diversity score | 0.1639 |
| Inference latency | 785.66 ms |

These subset-level results show the expected qualitative behavior of a stochastic model, but they are not yet strong enough to support a final thesis claim that diffusion outperforms the cVAE. In fact, on the current available evidence, the cVAE is both more accurate and far faster. The full diffusion evaluation must therefore be completed before drawing final comparative conclusions.

---

## 6.7 Discussion

### 6.7.1 Strengths

1. A complete conditional diffusion pipeline is now in place, including training, EMA, DDPM sampling, DDIM sampling, and classifier-free guidance.
2. The model can generate multiple trajectories per condition without requiring an explicit latent encoder at inference time.
3. The denoising progression can be visualized directly, which is useful for thesis explanation and debugging.

### 6.7.2 Current Limitations

1. Only the 1D U-Net denoiser has been trained; the optional flow-matching alternative is still unimplemented.
2. The current quantitative comparison is still based only on the 6-sample subset report.
3. The diffusion model remains expensive at inference time even with DDIM acceleration.
4. Step-count and guidance-scale ablation studies requested by the instruction plan have not been executed yet.
5. A uniform full-test comparison against the deterministic baselines and cVAE is still missing.

---

## 6.8 Summary

The diffusion phase of the project is substantially complete from an implementation standpoint. A conditional 1D U-Net diffusion model with cosine noise scheduling, classifier-free guidance, EMA, and DDPM/DDIM sampling has been built, trained, checkpointed, and visualized.

The main quantitative training result is a **best validation loss of 0.007965 at epoch 414**, with **1.43M parameters** and **494 logged epochs**. However, the final thesis narrative for this chapter must still be careful: training is complete, but the model's final comparative standing remains provisional until the full diffusion evaluation and ablation phases are run. At the current project stage, the diffusion model should be presented as a successfully implemented and scientifically relevant generative alternative, not yet as the leading practical method.

**Key figures for this chapter:**

| Figure | File |
|--------|------|
| 6.1 Diffusion architecture | TBD (manual diagram) |
| 6.2 Training curves | `figures/diffusion/training_curves.png` |
| 6.3 Generated samples | `figures/diffusion/generation_samples.png` |
| 6.4 Denoising progression | `figures/diffusion/denoising_progression.png` |
