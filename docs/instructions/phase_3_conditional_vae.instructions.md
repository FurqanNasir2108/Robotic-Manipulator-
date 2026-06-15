# Phase 3: Conditional VAE

> **Prerequisite:** Phase 2 complete. Read `00_main_orchestrator.instructions.md` first.

---

## Goal

Implement a Conditional Variational Autoencoder for stochastic trajectory generation.

---

## Tasks

### 3.1 Conditioning Scheme Design
Define how task-level conditions are encoded:
- **Option A (waypoint sequence):** Encode via 1D-CNN or small Transformer
- **Option B (goal pose):** Encode `(x, y, theta)` via MLP
- **Option C (shape descriptor):** One-hot shape + continuous parameters via MLP
- Make conditioning modular — the encoder/decoder accept a generic condition embedding

### 3.2 Encoder
- Input: condition embedding + ground-truth joint trajectory
- Architecture options:
  - Bidirectional GRU/LSTM → latent parameters `(mu, log_var)`
  - 1D-CNN → pooling → latent parameters
  - Transformer encoder → CLS token → latent parameters
- Output: `mu` ∈ R^d, `log_var` ∈ R^d
- Latent dimension `d` configurable (start with 16, sweep [8, 16, 32, 64])

### 3.3 Decoder
- Input: condition embedding + sampled `z ~ N(mu, sigma)`
- Architecture options (implement at least two):
  - **GRU decoder** (primary): autoregressive, outputs one timestep at a time
  - **1D-CNN decoder** (fallback): parallel, outputs full trajectory at once
  - **Transformer decoder** (optional): attention-based, good for long sequences
- Output: predicted joint trajectory `(N, 3)`

### 3.4 Loss Function
```python
loss = recon_weight * reconstruction_loss + kl_weight * kl_divergence
       + energy_weight * energy_penalty
       + smoothness_weight * smoothness_penalty
```
- **Reconstruction:** MSE on joint trajectory
- **KL divergence:** Standard Gaussian prior
- **Energy penalty:** Sum of squared joint velocities (from predicted trajectory)
- **Smoothness penalty:** Sum of squared jerk (third derivative)

### 3.5 KL Annealing (Critical — Prevents Posterior Collapse)
Implement **at least one** annealing strategy:
- **Monotonic warmup:** KL weight linearly increases from 0 to target over K epochs
- **Cyclical annealing:** KL weight oscillates between 0 and target every M epochs
- **Free bits:** Ensure KL per dimension ≥ λ before penalizing (λ ≈ 0.1–0.5)

Config in `configs/vae.yaml`:
```yaml
model:
  latent_dim: 16
  encoder_type: bigru       # bigru | cnn | transformer
  decoder_type: gru         # gru | cnn | transformer
  condition_type: waypoint  # waypoint | goal_pose | shape
  condition_dim: 64
losses:
  recon_weight: 1.0
  kl_weight: 0.01
  energy_weight: 0.001
  smoothness_weight: 0.001
  kl_annealing: cyclical    # monotonic | cyclical | free_bits
  kl_warmup_epochs: 50
  kl_cycle_epochs: 20
training:
  epochs: 500
  batch_size: 64
  lr: 0.0003
  optimizer: adam
  seed: 42
```

### 3.6 Training
- Train with KL annealing
- Log: total loss, recon loss, KL loss, energy loss, smoothness loss
- Save best model (lowest validation recon loss) and periodic checkpoints
- Save latent space samples for visualization

### 3.7 Inference & Sampling
- **Reconstruction:** Encode test sample → decode with sampled z
- **Generation:** Sample z ~ N(0, I), combine with condition → decode
- Generate K=10 trajectories per condition to measure diversity
- Measure inference time per trajectory

### 3.8 Evaluation
- Compare against baselines on all 9 metrics
- Additional VAE-specific metrics:
  - Reconstruction quality vs. baselines
  - Diversity: pairwise DTW distance across K samples per condition
  - Latent space structure: t-SNE/UMAP colored by shape type
- Save latent space plots to `figures/vae/`

---

## Deliverables

- [ ] cVAE model with at least 2 decoder variants
- [ ] KL annealing implemented and tested
- [ ] Training script (`scripts/train_vae.py`)
- [ ] Trained model checkpoint
- [ ] Latent space visualizations (t-SNE/UMAP)
- [ ] Diversity measurements
- [ ] Comparison table: cVAE vs baselines
- [ ] Sample trajectory plots in `figures/vae/`

---

## Go/No-Go Checks

| Check | Action if Failed |
|-------|-----------------|
| KL collapses to 0 | Activate free-bits or increase cyclical annealing frequency |
| Reconstruction worse than baselines | Increase latent dim; try CNN decoder for parallel prediction |
| Generated trajectories are jittery | Increase smoothness penalty weight; verify velocity computation |
| Latent space has no structure | Latent dim too large → reduce; or add supervised loss on shape labels |
| Posterior collapse (decoder ignores z) | Use aggressive KL annealing + input dropout on condition |

---

## Fallbacks

| Risk | Alternative |
|------|-------------|
| Standard VAE produces blurry trajectories | Switch to **VQ-VAE** with discrete latent codebook |
| GRU decoder accumulates errors | Use **1D-CNN decoder** for non-autoregressive generation |
| VAE training unstable | Lower learning rate to 1e-4, use gradient clipping, ensure data is properly normalized |
| Diversity too low | Add **conditional dropout** during training; increase latent dim; reduce KL warmup |
