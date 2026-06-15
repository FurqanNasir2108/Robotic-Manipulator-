# Phase 7: Evaluation and Comparison

> **Prerequisite:** Phases 2–6 complete. Read `00_main_orchestrator.instructions.md` first.

---

## Goal

Produce a thesis-grade comparative evaluation of all methods with ablation studies and generalization tests.

---

## Tasks

### 5.1 Standardized Evaluation Protocol
- All models evaluated on the **same test set** with the **same random seeds**
- For generative models, generate K=10 samples per condition and report:
  - Best sample metrics (oracle selection)
  - Mean sample metrics
  - Diversity metrics
- Implement `scripts/evaluate_models.py` that loads any saved checkpoint and runs full evaluation

### 5.2 Core Metrics Implementation
Implement in `src/evaluate/metrics.py`:

```python
def joint_rmse(pred, gt) -> float
def ee_position_error(pred, gt, fk_fn) -> float
def ee_orientation_error(pred, gt, fk_fn) -> float
def path_tracking_error(pred, gt, method='dtw') -> float  # DTW or Frechet
def energy_proxy(trajectory, method='velocity') -> float    # velocity or torque
def smoothness_jerk(trajectory, dt) -> float
def inference_time(model, condition, n_runs=100) -> float   # median ms
def diversity_score(samples, method='dtw') -> float         # pairwise distance
def constraint_violation_rate(trajectory, joint_limits) -> float
```

### 5.3 Comparison Tables
Generate results tables:

**Table 1: Main Comparison**
| Model | Joint RMSE | EE Pos Error | EE Orient Error | Energy | Smoothness | Inference (ms) | Diversity |
|-------|-----------|-------------|----------------|--------|------------|---------------|-----------|
| Analytical IK | — | — | — | — | — | — | — |
| CNN | — | — | — | — | — | — | N/A |
| CNN+LSTM | — | — | — | — | — | — | N/A |
| CNN+GRU | — | — | — | — | — | — | N/A |
| cVAE | — | — | — | — | — | — | — |
| Diffusion (DDPM) | — | — | — | — | — | — | — |
| Diffusion (DDIM) | — | — | — | — | — | — | — |

**Table 2: Per-Shape Breakdown** (one table per shape type)

**Table 3: Generalization to Unseen Shapes** (shapes held out during training)

### 5.4 Ablation Studies
Implement in `src/evaluate/ablations.py`:

| Ablation | Variable | Values |
|----------|----------|--------|
| VAE latent size | latent_dim | 8, 16, 32, 64 |
| VAE KL annealing | strategy | monotonic, cyclical, free_bits, none |
| VAE decoder type | decoder | gru, cnn, transformer |
| Diffusion steps | T | 100, 500, 1000 |
| DDIM sampling steps | steps | 10, 25, 50, 100 |
| Guidance scale | w | 0.0, 1.0, 2.0, 5.0 |
| Energy penalty weight | weight | 0.0, 0.001, 0.01, 0.1 |
| Smoothness weight | weight | 0.0, 0.001, 0.01, 0.1 |

### 5.5 Generalization Tests
| Test | Description |
|------|-------------|
| Unseen shape | Train without hexagon, test on hexagon |
| Altered parameters | Train on radius R, test on 0.5R and 1.5R |
| Different waypoint density | Train on 100 steps, test on 50 and 200 |
| Longer trajectories | Train on T=100, test on T=200 (if model supports) |

### 5.6 Statistical Rigor
- Report **mean ± standard deviation** across 3 random seeds per model
- Use paired tests (Wilcoxon signed-rank) for model comparisons if needed
- Report 95% confidence intervals where appropriate

### 5.7 Visualization
Generate and save to `figures/comparison/`:
- Trajectory overlay plots (all models on same condition)
- Metric bar charts with error bars
- Radar/spider charts for multi-metric comparison
- Ablation line plots
- Generalization heatmaps

### 5.8 Results Summary
- Write `results/metrics/summary.json` with all numeric results
- Write `results/metrics/summary.md` with formatted tables and key findings
- Generate `results/plots/` with all publication-quality figures

---

## Deliverables

- [ ] Full evaluation script (`scripts/evaluate_models.py`)
- [ ] All 9 metrics implemented and tested
- [ ] Main comparison table
- [ ] Per-shape breakdown tables
- [ ] Ablation study results
- [ ] Generalization test results
- [ ] Publication-quality figures in `figures/comparison/`
- [ ] Summary report in `results/metrics/summary.md`

---

## Go/No-Go Checks

| Check | Action if Failed |
|-------|-----------------|
| Generative models worse than baselines on all metrics | Verify training converged; check conditioning; ensure fair comparison |
| Diversity is zero for generative models | Check latent sampling; verify KL didn't collapse; inspect generated samples |
| Metrics disagree with visual inspection | Double-check metric computation; verify FK function; inspect normalization |
| Ablation results are noisy | Increase number of seeds; increase test set size; use more samples per condition |

---

## Fallbacks

| Risk | Alternative |
|------|-------------|
| Some models failed to train | Report partial results honestly; mark missing entries as "training failed" with explanation |
| Generalization tests all fail | Focus thesis narrative on in-distribution performance; discuss generalization as future work |
| Too many ablations to run | Prioritize latent_dim and sampling_steps ablations; drop lower-priority ones |
