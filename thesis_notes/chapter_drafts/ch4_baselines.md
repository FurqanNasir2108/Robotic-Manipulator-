# Chapter 4: Supervised Baseline Models

## 4.1 Introduction

This chapter presents the supervised baseline models used to establish reference performance for the trajectory generation task. Three neural network architectures are trained to learn a deterministic mapping from task-space waypoints to joint-space trajectories: a pure convolutional network (CNN), a CNN with LSTM decoder, and a CNN with GRU decoder. These baselines represent increasing levels of temporal modeling capability and serve as comparison benchmarks for the generative models introduced in Chapters 5 and 6.

---

## 4.2 Problem Formulation

The supervised baseline task is formulated as a sequence-to-sequence regression:

$$
f_\theta: \mathbf{W} \in \mathbb{R}^{T \times 3} \rightarrow \mathbf{Q} \in \mathbb{R}^{T \times 3}
$$

where $\mathbf{W} = [w_1, \ldots, w_T]$ is the input task-space trajectory (waypoints as $(x, y, \theta)$ at each timestep), $\mathbf{Q} = [q_1, \ldots, q_T]$ is the output joint-space trajectory ($q_1, q_2, q_3$ at each timestep), and $T = 100$ is the number of timesteps.

The model is trained to minimize the error between predicted and ground-truth joint trajectories using a combined loss function (Section 4.6).

---

## 4.3 CNN Baseline

### 4.3.1 Architecture

The CNN baseline treats the trajectory as a 1D signal and applies temporal convolutions to extract features, followed by fully connected layers for regression.

**Encoder (Convolutional Feature Extractor):**

| Layer | Channels | Kernel Size | Output Shape |
|-------|----------|-------------|-------------|
| Conv1d + ReLU + BN | 32 | 5 | $(32, T)$ |
| Conv1d + ReLU + BN | 64 | 5 | $(64, T)$ |
| Conv1d + ReLU + BN | 128 | 3 | $(128, T)$ |
| AdaptiveAvgPool1d | 128 | — | $(128, 1)$ |
| Flatten | — | — | $(128,)$ |

**Decoder (Fully Connected):**

| Layer | Dimensions | Activation |
|-------|-----------|------------|
| Linear + ReLU + Dropout(0.1) | 128 → 256 | ReLU |
| Linear + ReLU + Dropout(0.1) | 256 → 128 | ReLU |
| Linear | 128 → $T \times 3$ | None |
| Reshape | — | $(T, 3)$ |

The convolutional layers use `padding='same'` to preserve temporal resolution. Batch normalization is applied after each convolution for training stability.

### 4.3.2 Design Rationale

The pure CNN architecture serves as a **non-temporal baseline**: after global average pooling, all temporal information is compressed into a fixed-length vector. This tests whether the global shape of the input trajectory provides sufficient information for joint trajectory prediction, without explicit sequential modeling.

---

## 4.4 CNN + LSTM

### 4.4.1 Architecture

The CNN+LSTM model combines convolutional feature extraction with a recurrent decoder to model temporal dependencies in the output joint trajectory.

**Encoder (CNN):**

| Layer | Channels | Kernel Size | Output Shape |
|-------|----------|-------------|-------------|
| Conv1d + ReLU + BN | 32 | 5 | $(32, T)$ |
| Conv1d + ReLU + BN | 64 | 3 | $(64, T)$ |

The encoder output is transposed to $(T, 64)$ — a sequence of 64-dimensional feature vectors — and fed to the LSTM.

**Decoder (LSTM):**

| Component | Configuration |
|-----------|--------------|
| LSTM layers | 2 |
| Hidden size | 128 |
| Dropout | 0.1 (between layers) |
| Input | $(T, 64)$ from CNN encoder |
| Output | $(T, 128)$ hidden states |

**Output Projection:**

| Layer | Dimensions |
|-------|-----------|
| Linear + ReLU | 128 → 64 |
| Linear | 64 → 3 |

The output at each timestep is independently projected to 3 joint angles, producing a $(T, 3)$ trajectory.

### 4.4.2 Design Rationale

The LSTM decoder introduces **temporal memory** through gated recurrence. Unlike the pure CNN, the LSTM can learn dependencies between consecutive joint angles — for example, ensuring smooth transitions and enforcing continuity constraints implicitly through the hidden state.

---

## 4.5 CNN + GRU

### 4.5.1 Architecture

The CNN+GRU model has the same structure as the CNN+LSTM, but replaces the LSTM decoder with a GRU (Gated Recurrent Unit).

| Component | Configuration |
|-----------|--------------|
| Encoder | Same as CNN+LSTM (Conv1d 32→64) |
| GRU layers | 2 |
| Hidden size | 128 |
| Dropout | 0.1 |
| Output projection | 128 → 64 → 3 |

### 4.5.2 Design Rationale

The GRU is a computationally lighter alternative to the LSTM, using two gates (reset and update) instead of three (input, forget, output). This typically results in:
- Fewer parameters
- Faster training
- Comparable performance on many sequence tasks

The CNN+GRU baseline tests whether the full LSTM gating is necessary for this trajectory generation task.

---

## 4.6 Loss Function

All baselines are trained with a combined loss function consisting of three terms:

$$
\mathcal{L} = \lambda_{\text{MSE}} \cdot \mathcal{L}_{\text{MSE}} + \lambda_{\text{smooth}} \cdot \mathcal{L}_{\text{smooth}} + \lambda_{\text{endpoint}} \cdot \text{endpoint weighting}
$$

### 4.6.1 Mean Squared Error (MSE)

$$
\mathcal{L}_{\text{MSE}} = \frac{1}{T \cdot 3} \sum_{t=1}^{T} \sum_{j=1}^{3} (q_{t,j} - \hat{q}_{t,j})^2
$$

### 4.6.2 Smoothness Regularization

$$
\mathcal{L}_{\text{smooth}} = \frac{1}{(T-1) \cdot 3} \sum_{t=2}^{T} \sum_{j=1}^{3} (\hat{q}_{t,j} - \hat{q}_{t-1,j})^2
$$

This penalizes large joint angle changes between consecutive timesteps, encouraging smooth predicted trajectories.

### 4.6.3 Endpoint Weighting

The MSE contribution from the first and last timesteps is scaled by a factor of 2.0. This ensures accurate start and end positions, which are critical for trajectory execution.

### 4.6.4 Loss Weights

| Weight | Value |
|--------|-------|
| $\lambda_{\text{MSE}}$ | 1.0 |
| $\lambda_{\text{smooth}}$ | 0.01 |
| Endpoint weight | 2.0× |

---

## 4.7 Training Details

### 4.7.1 Common Configuration

All three baselines share the following training setup:

| Parameter | Value |
|-----------|-------|
| Optimizer | Adam ($\beta_1 = 0.9$, $\beta_2 = 0.999$) |
| Learning rate | 0.001 |
| Scheduler | Cosine annealing |
| Batch size | 64 |
| Maximum epochs | 200 |
| Early stopping patience | 30 epochs |
| Gradient clipping | max norm = 1.0 |
| Random seed | 42 |
| Data normalization | Z-score |
| Training samples | 20,000 |
| Validation samples | 2,500 |

### 4.7.2 Input/Output Format

- **Input:** Normalized task-space waypoints $\mathbf{W} \in \mathbb{R}^{100 \times 3}$ (transposed to $(3, 100)$ for Conv1d)
- **Output:** Normalized joint angles $\mathbf{Q} \in \mathbb{R}^{100 \times 3}$

---

## 4.8 Results

### 4.8.1 Training Convergence

| Model | Best Val Loss | Best Epoch | Total Epochs | Training Time |
|-------|--------------|------------|-------------|---------------|
| CNN | 0.0196 | 161 | 200 | ~8 min |
| CNN+LSTM | 0.0054 | 83 | 113 | ~12 min |
| CNN+GRU | 0.0061 | 56 | 86 | ~10 min |

The recurrent models (CNN+LSTM and CNN+GRU) significantly outperform the pure CNN baseline, achieving approximately 3–4× lower validation loss. Both recurrent models converge much earlier and trigger early stopping, while the CNN continues training for the full 200 epochs without matching their performance.

> **Figure 4.1:** Training and validation loss curves for all three baselines.
> See `figures/baselines/training_curves.png`

> **Figure 4.2:** Best validation loss comparison (bar chart).
> See `figures/baselines/comparison_bar.png`

### 4.8.2 Analysis

**CNN Baseline.** The pure CNN achieves reasonable performance but is fundamentally limited by the global average pooling operation, which discards temporal structure. The model must reconstruct a 300-dimensional output ($100 \times 3$) from a 128-dimensional bottleneck, making it difficult to capture fine-grained temporal patterns.

**CNN+LSTM.** The LSTM decoder achieves the best performance (val loss = 0.0054), demonstrating that temporal modeling is critical for accurate trajectory prediction. The autoregressive nature of the LSTM allows each predicted joint angle to depend on previously predicted values, naturally enforcing temporal coherence.

**CNN+GRU.** The GRU decoder achieves comparable performance to the LSTM (val loss = 0.0061) with faster convergence (best epoch 56 vs 83). The marginal performance difference suggests that the simpler GRU gating is sufficient for this trajectory length ($T = 100$).

### 4.8.3 Limitations of Supervised Baselines

Despite their reasonable accuracy, all three baselines share a fundamental limitation: they are **deterministic**. For a given task-space trajectory, they produce exactly one joint-space trajectory. However, as discussed in Chapter 3 (Section 3.4.2), the inverse kinematics admits multiple valid solutions at each waypoint. The supervised baselines can only learn the average of these solutions (which may itself be invalid), or collapse to one mode of the solution space.

This limitation motivates the generative approaches presented in Chapters 5 and 6, which can model the full distribution of valid trajectories.

---

## 4.9 Summary

Three supervised baseline models were trained and evaluated for the trajectory generation task:

| Model | Architecture | Val Loss | Key Property |
|-------|-------------|----------|--------------|
| CNN | Conv1d → Pool → FC | 0.0196 | No temporal modeling |
| CNN+LSTM | Conv1d → LSTM → FC | 0.0054 | Best accuracy |
| CNN+GRU | Conv1d → GRU → FC | 0.0061 | Fastest convergence |

The recurrent models significantly outperform the pure CNN, confirming the importance of temporal modeling for trajectory prediction. The CNN+LSTM achieves the best validation loss and serves as the primary supervised baseline for comparison with generative models. However, all supervised baselines are inherently limited to single-mode predictions, motivating the generative approaches in subsequent chapters.

**Key figures for this chapter:**
| Figure | File |
|--------|------|
| 4.1 Training Curves | `figures/baselines/training_curves.png` |
| 4.2 Baseline Comparison | `figures/baselines/comparison_bar.png` |
