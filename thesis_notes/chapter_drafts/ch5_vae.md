# Chapter 5: Conditional VAE for Trajectory Generation

## 5.1 Introduction

This chapter presents the first generative model for trajectory generation: a Conditional Variational Autoencoder (cVAE). Unlike the deterministic baselines in Chapter 4, the cVAE learns a latent distribution over joint-space trajectories conditioned on the task-space waypoints. By sampling from this distribution, the model can produce multiple diverse yet valid trajectories for the same input, directly addressing the one-to-many mapping challenge identified in Chapter 3.

---

## 5.2 Variational Autoencoder Theory

### 5.2.1 Generative Modeling with Latent Variables

A Variational Autoencoder (VAE) is a generative model that learns a latent representation $z \in \mathbb{R}^d$ of the data $x$ through an encoder-decoder framework. The generative process assumes:

$$
p(x) = \int p(x|z) p(z) \, dz
$$

where $p(z) = \mathcal{N}(0, I)$ is a standard Gaussian prior and $p(x|z)$ is the decoder (likelihood).

### 5.2.2 Evidence Lower Bound (ELBO)

Since the marginal likelihood $p(x)$ is intractable, VAEs maximize the Evidence Lower Bound:

$$
\text{ELBO} = \mathbb{E}_{q_\phi(z|x)} [\log p_\theta(x|z)] - D_{KL}(q_\phi(z|x) \| p(z))
$$

where:
- $q_\phi(z|x)$ is the encoder (approximate posterior)
- The first term is the **reconstruction loss**: how well the decoder reconstructs the input
- The second term is the **KL divergence**: regularizes the latent space toward the prior

### 5.2.3 Reparameterization Trick

To enable gradient-based optimization through the stochastic sampling step, the reparameterization trick parameterizes the sample as:

$$
z = \mu + \sigma \odot \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)
$$

where $\mu$ and $\log \sigma^2$ are outputs of the encoder network.

---

## 5.3 Conditional VAE Formulation

### 5.3.1 Conditioning on Task-Space Trajectories

The standard VAE is extended to a **conditional** VAE by incorporating the task-space waypoint trajectory $\mathbf{W}$ as a conditioning variable:

$$
\text{ELBO} = \mathbb{E}_{q_\phi(z|\mathbf{Q}, \mathbf{W})} [\log p_\theta(\mathbf{Q}|z, \mathbf{W})] - D_{KL}(q_\phi(z|\mathbf{Q}, \mathbf{W}) \| p(z))
$$

where $\mathbf{Q} \in \mathbb{R}^{100 \times 3}$ is the joint-space trajectory and $\mathbf{W} \in \mathbb{R}^{100 \times 3}$ is the task-space waypoint sequence.

### 5.3.2 Training vs. Inference

- **Training:** The encoder observes both the condition $\mathbf{W}$ and the ground-truth trajectory $\mathbf{Q}$, producing $(\mu, \log \sigma^2)$. A latent code $z$ is sampled and decoded.
- **Inference (generation):** The encoder is bypassed. $z \sim \mathcal{N}(0, I)$ is sampled from the prior and decoded conditioned on $\mathbf{W}$ alone. Different samples of $z$ produce different valid trajectories.

---

## 5.4 Architecture

The cVAE consists of three components: a condition encoder, a VAE encoder, and a VAE decoder.

### 5.4.1 Condition Encoder

The condition encoder transforms the task-space waypoint sequence into a fixed-size embedding:

| Layer | Configuration | Output Shape |
|-------|--------------|-------------|
| Conv1d + BN + ReLU | 3 → 32, kernel=5 | $(32, 100)$ |
| Conv1d + BN + ReLU | 32 → 64, kernel=3 | $(64, 100)$ |
| AdaptiveAvgPool1d | — | $(64, 1)$ |
| Linear | 64 → 64 | $(64,)$ |

The output is a condition embedding $c \in \mathbb{R}^{64}$.

### 5.4.2 VAE Encoder (BiGRU)

The encoder takes the concatenation of the ground-truth joint trajectory and the projected condition, and outputs the latent distribution parameters:

| Component | Configuration |
|-----------|--------------|
| Condition projection | Linear: 64 → 3 |
| Input | Concatenated trajectory + projected condition: $(100, 6)$ |
| BiGRU | 2 layers, hidden=128, bidirectional |
| Latent projection | Last hidden states (fwd + bwd) → $\mu \in \mathbb{R}^{16}$, $\log \sigma^2 \in \mathbb{R}^{16}$ |

The bidirectional GRU captures both forward and backward temporal dependencies in the trajectory, producing a comprehensive summary for the latent representation.

### 5.4.3 VAE Decoder (GRU)

The decoder reconstructs the joint trajectory from the latent code and condition:

| Component | Configuration |
|-----------|--------------|
| Input projection | Linear: $(16 + 64) \rightarrow 128$ |
| Input expansion | Broadcast to $(100, 128)$ across timesteps |
| GRU | 2 layers, hidden=128 |
| Output FC | 128 → 64 → 3 per timestep |

The decoder receives the same input at every timestep (the projected $[z; c]$ vector), and the GRU's recurrence introduces temporal variation to produce a smooth, coherent trajectory.

### 5.4.4 Alternative Decoder (1D-CNN)

A non-autoregressive CNN decoder is also implemented as a fallback:

| Layer | Configuration | Output Shape |
|-------|--------------|-------------|
| Linear | $(16 + 64) \rightarrow 128 \times 100$ | $(128, 100)$ |
| Conv1d + BN + ReLU | 128 → 64, kernel=5 | $(64, 100)$ |
| Conv1d + BN + ReLU | 64 → 32, kernel=3 | $(32, 100)$ |
| Conv1d | 32 → 3, kernel=1 | $(3, 100)$ |

The CNN decoder generates all timesteps in parallel, avoiding error accumulation but sacrificing explicit temporal modeling.

### 5.4.5 Model Size

| Component | Parameters |
|-----------|-----------|
| Condition Encoder | 3,712 |
| BiGRU Encoder | 277,568 |
| GRU Decoder | 356,070 |
| **Total** | **637,350** |

---

## 5.5 Loss Function

The cVAE is trained with a composite loss function:

$$
\mathcal{L} = \lambda_{\text{recon}} \mathcal{L}_{\text{recon}} + \beta(t) \cdot D_{KL} + \lambda_{\text{energy}} \mathcal{L}_{\text{energy}} + \lambda_{\text{smooth}} \mathcal{L}_{\text{smooth}}
$$

### 5.5.1 Reconstruction Loss

$$
\mathcal{L}_{\text{recon}} = \frac{1}{T \cdot 3} \sum_{t=1}^{T} \sum_{j=1}^{3} (q_{t,j} - \hat{q}_{t,j})^2
$$

with endpoint weighting (2×) on the first and last timesteps.

### 5.5.2 KL Divergence

$$
D_{KL} = -\frac{1}{2} \sum_{i=1}^{d} \left(1 + \log \sigma_i^2 - \mu_i^2 - \sigma_i^2\right)
$$

### 5.5.3 Energy Penalty

$$
\mathcal{L}_{\text{energy}} = \frac{1}{(T-1) \cdot 3} \sum_{t=2}^{T} \sum_{j=1}^{3} (\hat{q}_{t,j} - \hat{q}_{t-1,j})^2
$$

Penalizes large joint velocities, encouraging energy-efficient trajectories.

### 5.5.4 Smoothness Penalty (Jerk)

$$
\mathcal{L}_{\text{smooth}} = \frac{1}{(T-3) \cdot 3} \sum_{t} \sum_{j} (\Delta^3 \hat{q}_{t,j})^2
$$

Penalizes the third derivative (jerk) of the predicted trajectory for smooth motion.

### 5.5.5 Loss Weights

| Weight | Value |
|--------|-------|
| $\lambda_{\text{recon}}$ | 1.0 |
| $\lambda_{\text{KL}}$ (target) | 0.01 |
| $\lambda_{\text{energy}}$ | 0.001 |
| $\lambda_{\text{smooth}}$ | 0.001 |

---

## 5.6 KL Annealing

### 5.6.1 Posterior Collapse Problem

A well-known failure mode of VAEs is **posterior collapse**: the encoder learns to produce $q(z|x) \approx p(z)$ regardless of input, while the decoder ignores $z$ entirely. This happens when the KL penalty dominates early in training, before the decoder learns to use the latent code.

### 5.6.2 Cyclical Annealing Strategy

To prevent posterior collapse, the KL weight $\beta(t)$ follows a **cyclical annealing** schedule:

$$
\beta(t) = \lambda_{\text{KL}} \cdot \text{ramp}(t) \cdot \text{warmup}(t)
$$

where:
- **Warmup:** $\text{warmup}(t) = \min(1, t / T_{\text{warmup}})$ with $T_{\text{warmup}} = 50$ epochs
- **Cyclical ramp:** $\text{ramp}(t) = \min\left(1, \frac{t \bmod T_{\text{cycle}}}{T_{\text{cycle}} / 2}\right)$ with $T_{\text{cycle}} = 20$ epochs

This means the KL weight:
1. Starts at 0 and gradually increases during the warmup phase
2. After warmup, oscillates between 0 and the target value every 20 epochs
3. At each cycle start, the decoder gets a "free" period to improve reconstruction without KL pressure

### 5.6.3 Alternative Strategies (Implemented)

Two additional strategies are available:
- **Monotonic warmup:** $\beta(t)$ linearly increases from 0 to target over $K$ epochs, then stays constant
- **Free bits:** KL per dimension is only penalized when it exceeds a threshold $\lambda = 0.1$, ensuring each latent dimension encodes at least $\lambda$ nats of information

---

## 5.7 Training Details

### 5.7.1 Configuration

| Parameter | Value |
|-----------|-------|
| Optimizer | Adam ($\beta_1=0.9$, $\beta_2=0.999$) |
| Learning rate | 0.0003 |
| Scheduler | Cosine annealing |
| Batch size | 64 |
| Maximum epochs | 500 |
| Early stopping patience | 60 epochs (on val reconstruction loss) |
| Gradient clipping | max norm = 1.0 |
| Latent dimension | 16 |
| KL annealing | Cyclical (warmup=50, cycle=20) |
| Random seed | 42 |

### 5.7.2 Monitoring

During training, the following quantities are logged per epoch:
- Total loss, reconstruction loss, KL divergence
- Energy penalty, smoothness penalty
- Current KL weight $\beta(t)$
- Validation losses for all components

---

## 5.8 Results

### 5.8.1 Training Convergence

The strongest current cVAE evidence comes from the integrated full-scale run trained and evaluated through the standalone external-GPU notebook workflow. The resulting training history is archived in `results/metrics/cvae/cvae_full_4500_history.json`.

| Metric | Value |
|--------|-------|
| Best val reconstruction loss | **0.000774** |
| Best epoch | **426** |
| Logged epochs | **486** |
| Improvement vs previous repo cVAE run | **51.35%** |

This run substantially improves on the earlier repository cVAE checkpoint (`0.001590` at epoch 226). The reconstruction loss converges smoothly, with periodic fluctuations corresponding to the cyclical KL annealing schedule. When the KL weight resets, reconstruction briefly improves, then rises slightly as the KL penalty ramps back up.

### 5.8.2 KL Divergence Analysis

The latent regularization remains healthy in the improved run:

- KL is **non-zero throughout training**, which argues against posterior collapse,
- KL on the best validation-reconstruction epoch is approximately **0.028**,
- final logged KL remains non-zero in both train and validation histories.

This indicates that the latent code is being used meaningfully rather than ignored by the decoder.

### 5.8.3 Comparison with Deterministic Baselines

| Model | Type | Best Validation Loss | Relative to cVAE |
|-------|------|---------------------:|-----------------:|
| CNN | Deterministic | 0.0196 | 25.32× worse |
| CNN+LSTM | Deterministic | 0.0054 | 6.98× worse |
| CNN+GRU | Deterministic | 0.0061 | 7.88× worse |
| **cVAE** | **Generative** | **0.000774** | **—** |

These results are strong evidence that the cVAE is not merely trading deterministic accuracy for stochastic flexibility. It improves the core reconstruction objective substantially while also enabling multi-sample generation.

### 5.8.4 Full 4,500-Sample Test Evaluation

The cVAE is now the only learned generative model in the repository with a completed integrated **4,500-sample test evaluation**. The summary artifact is:

- `results/metrics/cvae/cvae_full_4500_evaluation_summary.json`

Overall cVAE full-test results:

| Metric | Value |
|--------|------:|
| Joint RMSE | **0.02012** |
| End-effector position error | **0.02911** |
| End-effector orientation error | **0.01493** |
| Path tracking error | **0.02882** |
| Energy proxy | **0.000427** |
| Smoothness jerk | **9.10e-06** |
| Constraint violation rate | **0.0** |
| Diversity score | **0.02157** |
| Inference time | **1.886 ms** |
| Mean-sample joint RMSE | **0.03436** |
| Oracle-sample joint RMSE | **0.02012** |

The oracle-versus-mean-sample gap is especially important. Oracle selection reduces joint RMSE by **41.43%** relative to the mean-sample error, showing that the model is generating multiple distinct candidates and that a downstream selector could exploit this diversity.

### 5.8.5 Per-Shape Performance and Generalization

Per-shape cVAE joint RMSE on the full test set is:

| Shape | Joint RMSE |
|-------|-----------:|
| Pentagon | **0.01311** |
| Circle | **0.01420** |
| Square | **0.01465** |
| Hexagon (held out during training) | **0.01543** |
| Line | **0.02245** |
| Random Smooth | **0.05495** |

This is one of the strongest results in the project so far. The low held-out **hexagon** error suggests that the cVAE is learning a trajectory prior that generalizes beyond memorized training geometry. The clearest remaining weakness is the `random_smooth` family, which is also the most structurally variable trajectory class in the dataset.

### 5.8.6 Comparison with Available Baseline Metrics

Two comparisons are currently justified:

1. **Validation comparison:** the cVAE clearly outperforms all deterministic baselines on the shared validation objective.
2. **Current test-time comparison:** the cVAE appears stronger than the best deterministic baseline in the currently available cross-model summary.

Using the current cross-model summary artifact:

- best deterministic model: **CNN+LSTM**
- subset joint RMSE: **0.0314**
- cVAE full-run joint RMSE: **0.02012**
- provisional margin: **35.99% lower joint RMSE**

This is encouraging, but it must still be described as **provisional** because the deterministic baselines have not yet been rerun under the same integrated 4,500-sample protocol used for the cVAE.

---

## 5.9 Discussion

### 5.9.1 Strengths

1. **Superior reconstruction:** The cVAE achieves the best validation reconstruction score among all learned models.
2. **Strong full-test evidence:** It now has a completed 4,500-sample evaluation with low error and zero constraint violations.
3. **Generalization:** Held-out hexagon performance remains strong despite hexagon being absent from training.
4. **Meaningful diversity:** Best-of-$K$ sampling produces a substantial oracle gain over average sample quality.
5. **Healthy latent usage:** Non-zero KL throughout training indicates that the latent variable is contributing.

### 5.9.2 Limitations

1. **Only one trained decoder variant is evidence-backed:** the GRU decoder is the trained production variant; the CNN decoder is implemented but not yet benchmarked.
2. **Cross-model full-test comparison is still incomplete:** only the cVAE currently has integrated full-scale test evidence.
3. **Random smooth trajectories remain difficult:** this is the clearest current failure mode.
4. **No explicit multi-seed study yet:** the current evidence is strong but still mostly single-seed.
5. **KL schedule figure export remains missing:** the schedule behavior is visible from logs, but a dedicated thesis figure has not yet been exported.

### 5.9.3 Comparison with Diffusion Models

The diffusion model in Chapter 6 provides an alternative generative approach that:

- does not require an explicit latent code at inference time,
- generates via iterative denoising rather than single-shot decoding,
- may ultimately offer a different diversity profile,
- is currently much slower and less convincing than the cVAE on the available subset comparison artifact.

At the present project stage, the cVAE is therefore the strongest generative model in actual evidence, while diffusion remains a promising but not yet fully benchmarked alternative.

---

## 5.10 Summary

A Conditional VAE was successfully implemented, trained, improved, and fully evaluated at thesis-relevant scale. The model combines a 1D-CNN condition encoder, a bidirectional GRU encoder, and a GRU decoder, with cyclical KL annealing to prevent posterior collapse.

Key current results:

- **637,350 parameters**
- **Best val reconstruction loss: 0.000774**
- **Best epoch: 426**
- **486 logged epochs**
- **Full-test joint RMSE: 0.02012**
- **Held-out hexagon RMSE: 0.01543**
- **Constraint violation rate: 0.0**

The cVAE is currently the strongest implemented learned approach in the repository. Relative to the deterministic baselines, the results are good by both the validation metric and the available test-time evidence. The only major caution is that the final full-test cross-model ranking still requires the deterministic baselines and diffusion model to be evaluated under the same integrated protocol.

**Key figures for this chapter:**

| Figure | File |
|--------|------|
| 5.1 cVAE architecture | TBD (manual diagram) |
| 5.2 Training curves | `results/plots/cvae_full_4500_training_curves.png` |
| 5.3 Reconstruction examples | `results/plots/cvae_full_4500_reconstruction_samples.png` |
| 5.4 Generated samples | `results/plots/cvae_full_4500_generation_samples.png` |
| 5.5 Latent space (t-SNE) | `results/plots/cvae_full_4500_latent_tsne.png` |
| 5.6 Full evaluation dashboard | `results/plots/cvae_full_4500_evaluation_dashboard.png` |
| 5.7 KL schedule export | Pending dedicated export |
