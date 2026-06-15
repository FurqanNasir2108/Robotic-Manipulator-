# Phase 2: Baseline Reproduction

> **Prerequisite:** Phase 1 complete. Read `00_main_orchestrator.instructions.md` first.

---

## Goal

Reproduce the student's earlier supervised trajectory prediction models and establish baseline performance.

---

## Tasks

### 2.1 Analytical IK Baseline
- Use the IK solver from Phase 1 as the ground-truth reference
- Implement all solution branches (elbow-up, elbow-down)
- Select the branch that minimizes energy cost along the trajectory
- Compute metrics: position error, orientation error, energy, smoothness

### 2.2 CNN Baseline
- **Architecture:**
  - Input: task-space condition `(x, y, theta)` or waypoint sequence (flattened or as 1D signal)
  - 1D convolutional layers → FC layers → joint trajectory output `(N, 3)`
  - Configurable: number of layers, kernel sizes, hidden dims
- **Config:** `configs/baseline_cnn.yaml`
  ```yaml
  model:
    type: cnn
    input_dim: depends_on_conditioning
    output_steps: 100
    output_dim: 3
    conv_channels: [32, 64, 128]
    kernel_sizes: [5, 5, 3]
    fc_dims: [256, 128]
    activation: relu
    dropout: 0.1
  training:
    epochs: 200
    batch_size: 64
    lr: 0.001
    optimizer: adam
    scheduler: cosine
    seed: 42
  ```

### 2.3 CNN + LSTM Baseline
- **Architecture:**
  - CNN feature extractor → LSTM sequence decoder
  - LSTM outputs joint angles step-by-step
  - Teacher forcing during training with scheduled sampling
- **Config:** `configs/baseline_cnn_lstm.yaml`
- **Fallback:** If LSTM training is unstable, try gradient clipping (max_norm=1.0) and layer normalization

### 2.4 CNN + GRU Baseline
- **Architecture:**
  - CNN feature extractor → GRU sequence decoder
  - Same structure as LSTM but with GRU cells
  - Typically faster training and similar performance
- **Config:** `configs/baseline_cnn_gru.yaml`
- **Fallback:** If GRU underperforms LSTM significantly, investigate hidden state dimensions and add residual connections

### 2.5 (Optional) Transformer Baseline
- **Architecture:**
  - Condition encoder → Transformer decoder with causal masking
  - Positional encoding for time steps
- **Rationale:** Provides a stronger deterministic reference for generative model comparison
- **Fallback:** Skip if compute budget is tight; the three baselines above are sufficient

### 2.6 Training Infrastructure
- Implement `src/train/trainer.py` with:
  - Generic training loop supporting all model types
  - Checkpoint saving (best + periodic)
  - Early stopping on validation loss
  - TensorBoard logging
  - Mixed-precision training (AMP) support
  - Gradient accumulation support for large effective batch sizes
- Implement `src/train/losses.py`:
  - MSE loss on joint trajectories
  - Optional weighted loss giving more importance to start/end of trajectory
  - Optional energy-aware loss term

### 2.7 Evaluation
- Run all baselines on the same test set
- Compute all 9 metrics from the orchestrator
- Generate comparison table
- Plot sample trajectories: predicted vs ground truth
- Save results to `results/metrics/baselines/`

---

## Deliverables

- [ ] 4 trained baseline models with saved checkpoints
- [ ] Training curves (loss vs epoch) for each baseline
- [ ] Comparison table: all metrics across all baselines
- [ ] Sample trajectory visualizations in `figures/baselines/`
- [ ] Baseline training script (`scripts/train_baselines.py`)
- [ ] All configs in `configs/`

---

## Go/No-Go Checks

| Check | Action if Failed |
|-------|-----------------|
| Baselines worse than random | Verify data loading, normalization, input/output alignment |
| CNN loss doesn't converge | Reduce LR, check data scaling, try BatchNorm |
| LSTM/GRU explodes | Apply gradient clipping, reduce LR, check sequence length |
| Transformer OOM | Reduce attention heads/layers, use smaller batch with gradient accumulation |
| All baselines equally bad | Data problem — go back to Phase 1 and validate dataset quality |

---

## Fallbacks

| Risk | Alternative |
|------|-------------|
| Teacher forcing causes train/test mismatch | Use **scheduled sampling**: gradually reduce teacher forcing ratio during training |
| Models overfit small dataset | Add **data augmentation** (time warping, noise injection) from `src/data/augmentation.py` |
| Training too slow on CPU | Use smaller model configs (`configs/baseline_*_small.yaml`) with fewer layers/smaller hidden dims |
