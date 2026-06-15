# Phase 4: Diffusion Model

> **Prerequisite:** Phase 3 complete. Read `00_main_orchestrator.instructions.md` first.

---

## Goal

Implement a conditional diffusion model for high-quality trajectory generation.

---

## Tasks

### 4.1 Forward Process
- Standard DDPM Gaussian noise schedule:
  - Linear or cosine beta schedule
  - T = 1000 steps (configurable)
  - `q(x_t | x_{t-1}) = N(sqrt(1-β_t) * x_{t-1}, β_t * I)`
- Pre-compute `alpha_bar`, `sqrt_alpha_bar`, `sqrt_one_minus_alpha_bar` for efficiency

### 4.2 Denoiser Network
Implement **at least one** of:
- **1D U-Net (primary):**
  - 1D convolutions operating on trajectory length dimension
  - Skip connections between encoder and decoder
  - Time embedding via sinusoidal encoding → MLP → added to features
  - Condition embedding concatenated or cross-attended
  - Configurable: depth, channels, attention layers
- **Transformer denoiser (alternative):**
  - Trajectory as sequence of (q1, q2, q3) tokens
  - Time step embedding added to each token
  - Condition tokens prepended or cross-attended
  - Configurable: layers, heads, hidden dim

### 4.3 Conditioning
Implement conditioning via:
- **Concatenation:** Append condition embedding to noisy trajectory at each diffusion step
- **Cross-attention:** Condition as key/value in attention layers (if using Transformer)
- **Classifier-free guidance (recommended):**
  - Randomly drop condition 10-20% of the time during training
  - At inference: `ε_guided = ε_uncond + w * (ε_cond - ε_uncond)`
  - Guidance scale `w` configurable (start with 1.0–3.0)

### 4.4 Training
- Loss: MSE between predicted noise and actual noise
  ```python
  loss = MSE(ε_θ(x_t, t, c), ε)
  ```
- Uniform timestep sampling during training
- Optional: v-prediction instead of ε-prediction for better sample quality
- Config in `configs/diffusion.yaml`:
  ```yaml
  model:
    denoiser_type: unet_1d    # unet_1d | transformer
    num_timesteps: 1000
    beta_schedule: cosine      # linear | cosine
    channels: [64, 128, 256]
    attention_layers: [2]      # which resolution levels use attention
    time_embed_dim: 128
    condition_type: waypoint
    condition_dim: 64
    classifier_free_guidance: true
    guidance_dropout: 0.1
    guidance_scale: 2.0
  training:
    epochs: 500
    batch_size: 64
    lr: 0.0002
    optimizer: adamw
    weight_decay: 0.01
    ema_decay: 0.9999          # exponential moving average for denoiser weights
    seed: 42
  sampling:
    method: ddpm               # ddpm | ddim | dpm_solver
    ddim_steps: 50
    dpm_solver_steps: 20
  ```

### 4.5 Sampling Strategies (Critical for Inference Speed)

| Method | Steps | Quality | Speed | Implementation Priority |
|--------|-------|---------|-------|------------------------|
| DDPM | 1000 | Highest | Slow | Required |
| DDIM | 50 | High | Fast | Required |
| DPM-Solver | 10–20 | High | Very fast | Recommended |
| Consistency Distillation | 1–4 | Good | Fastest | Optional fallback |

- Implement **DDPM** and **DDIM** at minimum
- DPM-Solver is strongly recommended for thesis demo speeds
- Use EMA weights for sampling

### 4.6 Inference
- Generate K=10 trajectories per condition
- Measure inference time for each sampling method
- Compare sample quality across methods

### 4.7 Evaluation
- Compare against baselines AND cVAE on all 9 metrics
- Diffusion-specific analysis:
  - Quality vs. number of sampling steps curve
  - Guidance scale ablation
  - Denoising trajectory visualization (show x_T → x_0 progression)
- Save denoising step visualizations to `figures/diffusion/`

---

## Deliverables

- [ ] Diffusion model with at least DDPM + DDIM sampling
- [ ] Classifier-free guidance implemented
- [ ] Training script (`scripts/train_diffusion.py`)
- [ ] Trained model checkpoint with EMA weights
- [ ] Sampling speed vs quality trade-off analysis
- [ ] Denoising step visualizations
- [ ] Comparison table: diffusion vs cVAE vs baselines
- [ ] Sample trajectory plots in `figures/diffusion/`

---

## Go/No-Go Checks

| Check | Action if Failed |
|-------|-----------------|
| Diffusion loss doesn't decrease | Check noise schedule; verify timestep sampling; ensure data normalization |
| Generated trajectories are noisy | Increase sampling steps; check EMA decay; verify denoiser capacity |
| DDIM produces artifacts | Try cosine schedule instead of linear; tune eta parameter |
| Inference too slow for demo | Switch to DPM-Solver (10–20 steps) or distill to consistency model |
| Guidance degrades quality | Reduce guidance scale; verify unconditional training samples are correct |

---

## Fallbacks

| Risk | Alternative |
|------|-------------|
| Diffusion too slow even with DDIM | Implement **Flow Matching** (`src/models/flow_matching.py`) — single-pass generation with ODE solver |
| U-Net too heavy | Use lightweight **ResNet-based denoiser** with fewer parameters |
| Training requires too many epochs | Use **pre-training on all shapes** then **fine-tuning per shape** |
| High compute / no GPU | Train with smaller T (100–200), fewer channels, and AMP; use DDIM with 10 steps at inference |

---

## Optional: Flow Matching Alternative

If diffusion proves too expensive, implement Flow Matching as a drop-in replacement:
- Learns a velocity field `v_θ(x_t, t, c)` mapping noise to data
- Training loss: `MSE(v_θ(x_t, t, c), x_1 - x_0)`
- Inference: ODE solve from noise to data (adaptive step solver)
- Typically faster than diffusion with comparable quality
- Reference: Lipman et al., "Flow Matching for Generative Modeling" (2023)
