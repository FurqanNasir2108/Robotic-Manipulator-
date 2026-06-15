# Generative Trajectory Modeling for a 3-Link Planar Robotic Manipulator: Comparing Deterministic Baselines, Conditional VAEs, and Diffusion Models

**Author:** [Student Name]  
**Affiliation:** [Department / University]  
**Advisor:** [Advisor Name]  
**Status:** Draft manuscript based on the current repository state and currently available results artifacts

---

## Abstract

Trajectory generation for robotic manipulators is commonly formulated as a mapping from task-space motion requirements to joint-space trajectories. Even for low-degree-of-freedom systems, this mapping can be intrinsically one-to-many because multiple joint configurations can realize the same end-effector pose sequence. Deterministic regression models therefore face a structural limitation: they can recover a single representative trajectory, but they cannot naturally model the diversity of valid solutions. This work studies trajectory generation for a 3-link planar revolute manipulator as a conditional sequence generation problem. We build a reproducible research pipeline that includes analytical inverse kinematics, deterministic neural baselines (CNN, CNN+LSTM, CNN+GRU), a conditional variational autoencoder (cVAE), and a conditional diffusion model with DDPM/DDIM-style sampling. The dataset contains 27,000 trajectories across six geometric shape families, with a held-out hexagon split used for generalization analysis. The deterministic baselines achieve best validation losses of 0.0196 (CNN), 0.0054 (CNN+LSTM), and 0.0061 (CNN+GRU), while the cVAE attains a best validation reconstruction loss of 0.000774. On a completed 4,500-sample cVAE test run, the model achieves 0.0201 joint RMSE, 0.0291 end-effector position error, zero constraint violations, and 0.0154 joint RMSE on the held-out hexagon class. A conditional diffusion model is also implemented and trained to a best validation loss of 0.007965. Cross-model test-time comparison remains preliminary because the deterministic baselines and diffusion model are still summarized mainly through the earlier 6-sample stratified comparison artifact. These findings support the use of latent-variable generative models for manipulator trajectory generation, while also highlighting the importance of completing a uniform full-benchmark comparison across all model families.

**Keywords:** robotic manipulators, trajectory generation, inverse kinematics, conditional variational autoencoders, diffusion models, sequence modeling

---

## 1. Introduction

Trajectory generation is a core capability in robotic manipulation, motion planning, and autonomous control. For a manipulator to execute a desired end-effector path, the system must produce a smooth, feasible, and accurate joint-space trajectory that respects the robot kinematics and actuator limitations. Classical approaches address this problem using analytical inverse kinematics, optimization-based planners, or iterative numerical solvers. These methods are often highly accurate when the manipulator model is known, but they can become brittle under changing task requirements, repeated re-planning, or demands for real-time deployment.

Recent learning-based methods have shown that neural sequence models can learn inverse mappings directly from data, potentially amortizing the cost of trajectory inference. However, an important challenge remains under-emphasized in simple regression formulations: the mapping from task-space trajectories to joint-space trajectories is frequently **one-to-many**. Even in a planar manipulator, multiple inverse-kinematics branches can generate distinct but valid joint trajectories for the same end-effector path. A single deterministic regressor cannot explicitly represent this ambiguity and may instead collapse to one mode or average across modes.

This observation motivates the use of **generative models** for manipulator trajectory prediction. Rather than predicting one fixed trajectory, a generative model can represent a distribution over valid joint-space realizations conditioned on a task-space input. Such a formulation is attractive not only for accuracy, but also for downstream applications that benefit from diversity, such as energy-aware control, trajectory selection under secondary objectives, or deployment-time choice among feasible alternatives.

In this work, we study generative trajectory modeling for a 3-link planar robotic manipulator. The robot is intentionally simple enough to support analytical inverse kinematics and rigorous dataset generation, while still exhibiting the multi-solution structure needed to motivate generative learning. We compare deterministic baselines against two conditional generative model families:

1. a **conditional variational autoencoder (cVAE)** with a latent trajectory representation, and
2. a **conditional diffusion model** with iterative denoising and DDIM-based accelerated sampling.

The project is organized as a reproducible research codebase spanning simulation, dataset generation, training, evaluation, and figure production. At the current stage, the simulation and training pipeline is complete, and the evaluation framework is implemented. The strongest fully supported quantitative claims now come from validation-set performance together with a completed 4,500-sample cVAE evaluation and an earlier 6-sample cross-model comparison artifact. Full apples-to-apples test-time comparison across deterministic, cVAE, and diffusion models is still incomplete and should be treated as future strengthening evidence rather than established final ranking.

### 1.1 Contributions

The main contributions of the current work are:

1. A reproducible end-to-end trajectory-generation pipeline for a 3-link planar manipulator, including analytical inverse kinematics, normalized dataset construction, and shape-based generalization splits.
2. A controlled comparison between deterministic sequence regressors and generative trajectory models on the same manipulator task.
3. A conditional VAE that substantially improves validation reconstruction error over deterministic baselines while enabling multi-sample joint-trajectory generation.
4. A conditional diffusion implementation with 1D U-Net denoising, cosine noise scheduling, classifier-free guidance, and DDPM/DDIM sampling.
5. A multi-metric evaluation framework covering reconstruction quality, task-space error, smoothness, energy proxy, diversity, and inference latency.

### 1.2 Research Questions

The current study is guided by three research questions that can already be grounded in the repository artifacts:

1. **RQ1:** How far can deterministic neural baselines reduce trajectory-prediction error for a 3-link planar manipulator when conditioned on task-space waypoint sequences?
2. **RQ2:** Does a conditional generative model, especially a cVAE, improve upon deterministic baselines while also supporting multi-sample trajectory generation for one-to-many inverse-kinematics mappings?
3. **RQ3:** What do the currently completed results suggest about the trade-offs among accuracy, diversity, and inference cost when comparing deterministic, latent-variable, and diffusion-based approaches?

### 1.3 Novelty and Scope

The novelty of the current work lies less in proposing a fundamentally new generative architecture and more in building a controlled, reproducible comparison pipeline for one-to-many manipulator trajectory generation. In particular, the project combines analytical inverse kinematics, deterministic sequence baselines, a conditional VAE, a conditional diffusion model, shared normalization and evaluation infrastructure, and explicit generalization support through a held-out shape family. This makes the contribution valuable as an experimentally grounded benchmark-style study at the intersection of inverse kinematics and modern generative modeling, while keeping the paper's claims appropriately scoped to the completed evidence.

### 1.4 Paper Organization

The remainder of the paper is organized as follows. Section 2 reviews relevant literature. Section 3 presents the robot model, kinematics, and dataset generation pipeline. Section 4 describes the deterministic and generative models. Section 5 details the training and evaluation protocol. Section 6 reports the currently available results. Section 7 discusses implications and the current status of the evaluation evidence. Section 8 summarizes the principal limitations. Section 9 outlines the most important future-work directions. Section 10 provides a reproducibility statement. Section 11 states data and code availability. Section 12 concludes the paper.

---

## 2. Literature Survey and Related Work

Trajectory generation for manipulators has traditionally been approached through a combination of analytical inverse kinematics, sampling-based motion planning, and trajectory optimization. Optimization-based methods such as CHOMP, STOMP, and TrajOpt are especially influential because they explicitly trade off smoothness, collision avoidance, and dynamic feasibility. Their principal strength lies in geometric and physical interpretability; their main drawback is the computational burden associated with repeated optimization, especially when many similar queries must be answered online.

Learning-based inverse kinematics and motion generation methods attempt to amortize this computation by replacing iterative solving with direct prediction. Deterministic regressors can be highly efficient at inference time, but they remain limited when the output space is multi-modal. In inverse kinematics, this issue is fundamental: the same task-space target may admit multiple valid joint configurations, and an entire trajectory can often be realized by more than one branch-consistent joint sequence.

Latent-variable generative models provide one mechanism for addressing this ambiguity. Variational autoencoders, introduced in the deep latent-variable setting by Kingma and Welling [1], learn a structured latent space that supports sampling, interpolation, and conditional generation. In robotics, VAEs have been used for motion representation, policy learning, and inverse problem formulations where multiple plausible outputs exist.

Diffusion models provide a second route to generative modeling. Denoising Diffusion Probabilistic Models (DDPMs) [2] learn to reverse a stochastic noising process, while Denoising Diffusion Implicit Models (DDIMs) [3] provide a practical acceleration strategy at inference time. Classifier-free guidance [4] allows conditional generation without an auxiliary classifier and has become a standard conditioning technique. More recently, diffusion-based policies have demonstrated strong performance on complex robotics control tasks, suggesting that diffusion-style generative modeling is relevant beyond image synthesis [5].

Within robotics, these trends point to an important gap: while generative models are increasingly popular for action and policy generation, there is comparatively less focused study on **trajectory diversity for low-dimensional manipulator kinematics under controlled, analytically verifiable conditions**. This work targets that niche directly. By using a planar manipulator with closed-form inverse kinematics, we obtain a setting where the ambiguity of the trajectory-generation problem is real, measurable, and experimentally tractable.

---

## 3. Robot Model and Dataset

### 3.1 Manipulator Definition

The robot considered in this study is a 3-link planar revolute manipulator operating in the $xy$ plane. The link lengths are:

$$
l_1 = 1.0,\quad l_2 = 1.0,\quad l_3 = 0.5
$$

with joint limits

$$
q_i \in [-\pi, \pi], \quad i \in \{1,2,3\}.
$$

The manipulator base is fixed at the origin. The end-effector pose is represented by $(x, y, \theta)$, where $\theta$ is the planar orientation.

Suggested final figure: `figures/simulation/arm_diagram.png`
Use this figure in the final manuscript to introduce the geometric testbed before the kinematic equations.

### 3.2 Forward and Inverse Kinematics

For joint configuration $\mathbf{q} = [q_1, q_2, q_3]^T$, the forward kinematics are

$$
x = l_1 \cos(q_1) + l_2 \cos(q_1 + q_2) + l_3 \cos(q_1 + q_2 + q_3),
$$

$$
y = l_1 \sin(q_1) + l_2 \sin(q_1 + q_2) + l_3 \sin(q_1 + q_2 + q_3),
$$

$$
\theta = q_1 + q_2 + q_3.
$$

Inverse kinematics is solved analytically by first computing the wrist position and then solving the induced 2-link problem for the first two joints. In most reachable cases, this yields two valid branches: elbow-up and elbow-down. This branch multiplicity is a central reason the task is well suited to generative modeling.

Suggested final figure: `figures/simulation/ik_branches.png`
Use this figure where the paper motivates one-to-many mapping behavior.

### 3.3 Trajectory Families

Six shape families are used to populate the dataset:

1. circle
2. square
3. pentagon
4. hexagon
5. line
6. random smooth spline-like trajectories

Each sample contains $T=100$ task-space waypoints and a corresponding joint-space trajectory. When a waypoint sequence admits multiple IK solutions, the dataset generation pipeline uses **closest-solution tracking** to maintain branch consistency and avoid discontinuous jumps between successive joint configurations.

### 3.4 Dataset Split

The full dataset contains **27,000 trajectories**:

- 25,000 samples from five primary training shapes: circle, square, pentagon, line, random smooth
- 2,000 samples from hexagon reserved for test-time generalization

The resulting split is:

| Split | Samples |
|-------|---------|
| Train | 20,000 |
| Validation | 2,500 |
| Test | 4,500 |

The test set contains both in-distribution samples and the held-out hexagon class, allowing a first view of generalization to unseen geometry.

Suggested final figure: `figures/simulation/shape_trajectories_grid.png`
Use this figure to show the shape families that define the conditional input distribution.

### 3.5 Normalization

All task-space and joint-space trajectories are normalized using z-score statistics computed from the training set only:

$$
\hat{x} = \frac{x - \mu}{\sigma}.
$$

This normalization is applied consistently during training, evaluation, and inverse transformation for metric reporting.

---

## 4. Methodology

### 4.1 Problem Formulation

The learning problem is formulated as conditional sequence generation:

$$
f_\theta: \mathbf{W} \in \mathbb{R}^{100 \times 3} \rightarrow \mathbf{Q} \in \mathbb{R}^{100 \times 3},
$$

where $\mathbf{W}$ is the task-space waypoint sequence and $\mathbf{Q}$ is the joint-space trajectory. Deterministic models learn a direct mapping from $\mathbf{W}$ to one predicted $\mathbf{Q}$. Generative models instead learn a conditional distribution $p(\mathbf{Q}\mid\mathbf{W})$.

#### 4.1.1 Notation

Throughout the paper, $\mathbf{W}$ denotes a task-space waypoint sequence of length $T=100$, $\mathbf{Q}$ denotes the corresponding joint-space trajectory, and $z$ denotes a latent variable used by the cVAE. For stochastic models, $K$ denotes the number of generated samples used during evaluation when computing oracle or mean-sample metrics.

#### 4.1.2 Reference Analytical IK Baseline

In addition to learned models, the project includes an analytical inverse-kinematics reference solver. This method is not treated as a learned competitor; instead, it serves as a kinematics-grounded reference point for evaluating how closely the learned models approach branch-consistent trajectory reconstruction on the synthetic dataset. In the current subset comparison artifact, the analytical IK method produces near-zero geometric errors, which is expected because the dataset itself is generated from the same kinematic structure.

### 4.2 Deterministic Baselines

Three deterministic baselines are implemented.

### 4.2.1 CNN Regressor

The CNN baseline treats the trajectory as a 1D signal and performs temporal convolution followed by global pooling and fully connected regression. It provides a non-recurrent reference point with modest inference cost.

### 4.2.2 CNN+LSTM

The CNN+LSTM baseline first encodes the waypoint sequence using temporal convolutions and then predicts joint trajectories through a two-layer LSTM decoder. This model explicitly incorporates temporal recurrence and provides the strongest deterministic validation performance in the current codebase.

### 4.2.3 CNN+GRU

The CNN+GRU baseline mirrors the CNN+LSTM structure but replaces the LSTM decoder with a GRU. This tests whether a lighter recurrent cell can recover similar temporal structure with faster convergence or lower computational cost.

### 4.3 Conditional Variational Autoencoder

The cVAE is the primary generative baseline in the current study.

### 4.3.1 Architecture

The cVAE consists of:

1. a 1D-CNN condition encoder that maps task-space waypoints to a 64-dimensional embedding,
2. a bidirectional GRU encoder that predicts the latent mean and log-variance,
3. a GRU decoder that reconstructs the full joint-space trajectory from the latent code and condition embedding.

The trained configuration uses a latent dimension of 16 and contains **637,350 trainable parameters**.

### 4.3.2 Objective

Training minimizes a weighted combination of reconstruction error, KL divergence, an energy proxy, and a smoothness penalty:

$$
\mathcal{L} =
\lambda_{\text{recon}}\mathcal{L}_{\text{recon}}
 + \beta(t) D_{KL}
 + \lambda_{\text{energy}}\mathcal{L}_{\text{energy}}
 + \lambda_{\text{smooth}}\mathcal{L}_{\text{smooth}}.
$$

KL annealing is implemented using a cyclical schedule with warmup, which is important for avoiding posterior collapse while retaining a meaningful latent representation.

### 4.4 Conditional Diffusion Model

The second generative model is a conditional diffusion model.

### 4.4.1 Architecture

The diffusion system uses:

1. a 1D-CNN condition encoder,
2. a 1D U-Net denoiser,
3. sinusoidal time embeddings,
4. classifier-free guidance during inference.

The current trained model has **1,425,606 trainable parameters**.

### 4.4.2 Noise Model and Sampling

The forward process uses a cosine noise schedule with 1,000 diffusion steps. During inference, both DDPM and DDIM sampling are implemented, though DDIM is the practical default because of its lower runtime cost. The current configuration uses **75 DDIM steps** with guidance scale **2.0**.

### 4.5 Model Summary

The principal learned model families currently implemented in the project are summarized below.

| Model | Type | Parameters | Inference Style |
|-------|------|-----------:|-----------------|
| CNN | Deterministic | 140,588 | one-shot regression |
| CNN+LSTM | Deterministic | 246,787 | one-shot recurrent decoding |
| CNN+GRU | Deterministic | 188,931 | one-shot recurrent decoding |
| cVAE | Generative | 637,350 | latent sampling + decode |
| Diffusion | Generative | 1,425,606 | iterative denoising |

---

## 5. Training and Evaluation Protocol

### 5.1 Training Setup

All models are trained on normalized trajectories with fixed random seeds and gradient-based optimization.

### Deterministic models

- optimizer: Adam
- learning rate: 0.001
- scheduler: cosine annealing
- batch size: 64
- early stopping enabled

### cVAE

- optimizer: Adam
- learning rate: 0.0003
- batch size: 64
- maximum epochs: 500
- early stopping patience: 60
- cyclical KL annealing with warmup

### Diffusion

- optimizer: AdamW
- learning rate: 0.0002
- batch size: 64
- maximum epochs: 500
- early stopping patience: 80
- cosine beta schedule
- EMA tracking

### 5.2 Metrics

The evaluation framework computes:

1. joint-space RMSE
2. end-effector position error
3. end-effector orientation error
4. path tracking error
5. energy proxy
6. smoothness (jerk-based)
7. inference latency
8. diversity score for stochastic models
9. constraint violation rate

For generative models, the framework reports both:

- **oracle metrics:** best sample among $K$ generated trajectories for each condition
- **mean sample metrics:** average quality across the $K$ generated samples

The principal metrics used in the current comparison pipeline are defined as follows. Joint-space accuracy is measured with root-mean-squared error:

$$
\mathrm{RMSE}(\hat{\mathbf{Q}}, \mathbf{Q})
=
\sqrt{\frac{1}{3T}\sum_{t=1}^{T}\|\hat{\mathbf{q}}_t - \mathbf{q}_t\|_2^2 }.
$$

End-effector position error is computed as the mean Euclidean distance between predicted and reference task-space positions after forward kinematics, while orientation error is the mean absolute wrapped angular difference over the full trajectory. Path-tracking error is computed with dynamic time warping in the current evaluation code, normalized by trajectory length.

The energy proxy reported in the present comparison uses the mean squared joint-velocity magnitude:

$$
\mathcal{E}_{\mathrm{vel}}
=
\frac{1}{T-1}\sum_{t=1}^{T-1}\|\mathbf{q}_{t+1} - \mathbf{q}_t\|_2^2,
$$

while smoothness is quantified through the mean squared jerk magnitude obtained by finite differences. Inference latency is reported as the median runtime over repeated forward passes. For stochastic models, diversity is measured as the mean pairwise path-tracking distance between generated samples, and joint-limit feasibility is summarized through the fraction of joint values that violate the allowed range.

### 5.3 Current Evaluation Status

The evaluation pipeline is implemented and now supports two completed evidence layers:

1. a **full 4,500-sample cVAE evaluation** executed through the standalone external-GPU notebook workflow,
2. an earlier **6-sample stratified cross-model comparison** containing one example from each test-set shape class.

This distinction matters for the scientific interpretation of the results. The cVAE now has full-test evidence, but model-to-model ranking across all learned methods should still be treated as provisional until the deterministic baselines and diffusion model are evaluated under the same full protocol.

### 5.4 Generalization Protocol

Generalization is probed through the dataset split itself: the test set contains both in-distribution samples and trajectories from a held-out hexagon shape family that is not used during training. This gives the project two complementary evaluation views, namely interpolation across seen shape families and limited out-of-distribution assessment on unseen geometry. The completed full cVAE artifact retains this split structure and reports strong held-out hexagon behavior, while the broader cross-model comparison still relies on the earlier subset artifact.

### 5.5 Generative Evaluation Protocol

For generative models, the current evaluation pipeline draws $K=10$ samples per conditioning input and reports both oracle and mean-sample behavior. In the completed implementation, oracle selection is based on the lowest joint-space RMSE among the sampled trajectories, while diversity is computed as the mean pairwise dynamic-time-warping distance between generated samples. To support downstream analysis, the pipeline can cache the full set of generated samples for each evaluated condition rather than only the oracle trajectory.

Inference latency is benchmarked separately from per-sample metric accumulation. The current implementation measures latency on a small stratified subset of conditions, uses repeated forward passes per condition, reports the median latency for each condition, and then averages those per-condition medians into the model-level latency number shown in the summary tables. This avoids conflating raw model runtime with dataset traversal overhead.

The principal evaluation settings reflected in the current code and completed artifacts are summarized below:

| Setting | Current Value / Behavior |
|---------|--------------------------|
| Completed cVAE artifact size | 4,500 test samples |
| Completed cross-model artifact size | 6 stratified test samples |
| Completed shape coverage in cross-model artifact | 1 sample each from circle, square, pentagon, hexagon, line, and random_smooth |
| Generative samples per condition | $K=10$ |
| Oracle selection rule | Lowest joint-space RMSE among generated samples |
| Path-tracking metric | DTW-based distance |
| Diversity metric | Mean pairwise DTW distance across generated samples |
| Latency benchmarking | 10 repeated runs on 3 stratified conditions, aggregated as mean of per-condition medians |
| Generative cache behavior | Full generated sample set stored for each evaluated condition |
| Included models in current configuration | analytical_ik, CNN, CNN+LSTM, CNN+GRU, cVAE, diffusion_ddim |

### 5.6 Statistical Reporting Conventions

The current manuscript reports validation losses from the best logged checkpoints of the completed training runs, full-test cVAE metrics from the integrated 4,500-sample artifact, and cross-model evaluation metrics from the earlier stratified subset artifact. These numbers are therefore best interpreted as reproducible descriptive results from the present repository state, not as final multi-seed inferential estimates. Formal statistical comparisons, confidence intervals, and multi-seed significance analysis are intentionally deferred until the uniform full benchmark and ablation phases are completed.

### 5.7 Reproducibility and Artifact Availability

The present study is built around a reproducible codebase with modular simulation, training, and evaluation components. Current repository artifacts include:

1. dataset generation and normalization code,
2. trained checkpoints for deterministic, cVAE, and diffusion models,
3. evaluation scripts and summary reports,
4. paper and thesis drafting material.

In addition, a standalone self-contained notebook has been prepared for portable cVAE training and evaluation on an external GPU environment. This notebook has now been used to complete the full-scale 4,500-sample cVAE evaluation without requiring direct dependency on the local repository structure, and it remains the recommended path for larger external-GPU reruns.

---

## 6. Results

### 6.1 Training Outcome Summary

Before examining each model family in detail, the following table summarizes the principal completed training outcomes across all learned models. It is intended as a compact overview of scale, convergence point, and validation statistic. The diffusion validation objective should still be interpreted separately from deterministic or cVAE reconstruction losses.

| Model | Parameters | Best Validation Statistic | Best Epoch / Logged Epochs | Interpretation |
|-------|-----------:|---------------------------|----------------------------|----------------|
| CNN | 140,588 | 0.0196 | 131 / 161 | Deterministic regression baseline |
| CNN+LSTM | 246,787 | 0.0054 | 53 / 83 | Strongest deterministic baseline |
| CNN+GRU | 188,931 | 0.0061 | 26 / 56 | Recurrent baseline slightly behind LSTM |
| cVAE | 637,350 | 0.000774 (reconstruction) | 426 / 486 | Best reconstruction-oriented learned result and current strongest full-test generative artifact |
| Diffusion | 1,425,606 | 0.007965 (diffusion loss) | 414 / 494 | Stable training, but objective not directly comparable to cVAE reconstruction loss |

### 6.2 Analytical IK Reference

The evaluation pipeline includes an analytical inverse-kinematics reference method as a non-learned comparator. On the completed subset artifact, this reference yields effectively zero geometric reconstruction and tracking error, which is expected in the current synthetic setting because the dataset is derived from the same kinematic model. Its main role in the paper is therefore interpretive: it provides a reference ceiling for what the learned models are trying to approximate under branch-consistent trajectory generation.

### 6.3 Deterministic Baselines

The current best validation losses for the deterministic baselines are:

| Model | Best Validation Loss | Best-Epoch Index in Logged History |
|-------|----------------------|------------------------------------|
| CNN | 0.0196 | 131 / 161 |
| CNN+LSTM | **0.0054** | 53 / 83 |
| CNN+GRU | 0.0061 | 26 / 56 |

These results show a clear gain from temporal modeling. The recurrent baselines outperform the pure CNN by roughly a factor of three to four, indicating that sequence structure matters strongly in this task.

The broader evaluation pipeline also reports an analytical inverse-kinematics reference method when test-time artifacts are available. That reference should be interpreted as a kinematic ceiling for this controlled synthetic setting rather than as a directly comparable learned baseline.

Suggested final figure: `figures/baselines/training_curves.png`
Use this figure to visualize the recurrent-model advantage in convergence and final validation loss.

### 6.4 cVAE Performance

The cVAE now reaches a **best validation reconstruction loss of 0.000774** at epoch 426 and runs for 486 logged epochs before early termination. Relative to the strongest deterministic baseline (CNN+LSTM), this is a **6.98× reduction in validation reconstruction loss**.

This result is notable because the cVAE is not merely matching deterministic performance while adding stochasticity; it is improving the underlying reconstruction objective while simultaneously enabling multi-sample generation.

The latent regularization behavior remains healthy: the KL term stays non-zero throughout training and is approximately **0.028** on the best validation-reconstruction epoch, suggesting that the model avoids posterior collapse and uses the latent variable meaningfully.

The strongest new evidence comes from the completed **4,500-sample test evaluation**. On that run, the cVAE achieves:

- **0.02012** joint RMSE,
- **0.02911** end-effector position error,
- **0.01493** end-effector orientation error,
- **0.02882** path-tracking error,
- **0.000427** energy proxy,
- **9.10e-06** smoothness jerk,
- **0.0** constraint-violation rate,
- **0.02157** diversity score,
- **1.886 ms** reported inference time in the Colab evaluation environment.

The oracle-versus-mean-sample gap is also informative: mean-sample joint RMSE is **0.03436**, while oracle selection reduces this to **0.02012**, a **41.43%** improvement. This confirms that the model is generating multiple distinct candidates and that a downstream selector could exploit that diversity.

Per-shape results show the strongest cVAE performance on regular geometric trajectories and on the held-out generalization class:

- pentagon: **0.01311** joint RMSE
- circle: **0.01420**
- square: **0.01465**
- hexagon: **0.01543**
- line: **0.02245**
- random_smooth: **0.05495**

The held-out hexagon result is especially important because that shape family is excluded from training. Its low error suggests that the cVAE is learning a trajectory prior that generalizes beyond memorized training geometry.

In terms of model scale, the cVAE is noticeably larger than the deterministic baselines but remains below one million trainable parameters. This places it in a practical middle ground: expressive enough to capture multimodal behavior, yet still compact enough to remain plausible for repeated experimentation and downstream deployment-oriented studies.

Suggested final figure: `results/plots/cvae_full_4500_reconstruction_samples.png`
Companion figure worth adding if space allows: `results/plots/cvae_full_4500_latent_tsne.png`
Use the latent-space figure if the final paper wants qualitative support for the claim that the learned latent variable is structured and non-degenerate.

### 6.5 Diffusion Training Performance

The diffusion model reaches a **best validation loss of 0.007965** at epoch 414 with 494 logged epochs. This demonstrates successful training convergence and confirms that the conditioning and denoising pipeline is operational. However, the diffusion validation objective is not directly comparable to the cVAE reconstruction objective, so this result should primarily be interpreted as evidence that the diffusion training procedure is stable rather than as a head-to-head performance ranking.

The diffusion network is also the largest learned model in the repository at **1.43M parameters**. Even before full-scale inference benchmarking, this already foreshadows a central trade-off in the project: diffusion offers a flexible generative formulation, but likely at a higher runtime and systems cost than latent-variable alternatives such as the cVAE.

Suggested final figure: `figures/diffusion/denoising_progression.png`
Use this figure to explain the iterative refinement process rather than only reporting the final loss value.

### 6.6 Preliminary Comparative Evaluation

The completed subset-level evaluation currently yields the following overall metrics:

| Model | Joint RMSE | EE Pos | EE Orient | Path | Energy | Smoothness | Inference (ms) | Diversity |
|-------|-----------:|-------:|----------:|-----:|-------:|-----------:|---------------:|----------:|
| analytical_ik | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0006 | 0.0000 | 5.14 | N/A |
| cnn | 0.0556 | 0.0715 | 0.0227 | 0.0399 | 0.0004 | 0.0000 | 1.22 | N/A |
| cnn_lstm | 0.0314 | 0.0414 | 0.0262 | 0.0218 | 0.0006 | 0.0003 | 2.55 | N/A |
| cnn_gru | 0.0384 | 0.0454 | 0.0218 | 0.0266 | 0.0006 | 0.0001 | 27.05 | N/A |
| cvae | **0.0291** | **0.0394** | 0.0248 | **0.0208** | 0.0005 | 0.0000 | 13.77 | **0.3211** |
| diffusion_ddim | 0.0997 | 0.1172 | **0.0121** | 0.0675 | 0.0007 | 0.0015 | 785.66 | 0.1639 |

Within this completed subset report, the cVAE is the strongest learned model overall. It outperforms the deterministic baselines on joint RMSE and path-tracking error while also providing a non-zero diversity score. The diffusion model exhibits stochastic diversity but is currently much slower and less accurate than the cVAE under the completed subset benchmark. This table should now be read specifically as a **preliminary cross-model snapshot** rather than as the main cVAE evidence source, because the cVAE itself now also has a separate full 4,500-sample evaluation artifact.

Suggested final figure: `results/plots/accuracy_latency_tradeoff.png`
Use this figure as the compact visual summary of practical trade-offs across methods.

### 6.7 Current Findings Summary

The strongest current findings can be summarized as follows:

| Finding | Status | Basis in Current Artifacts |
|---------|--------|----------------------------|
| Temporal sequence modeling is important for learned trajectory prediction | Strong | CNN+LSTM and CNN+GRU substantially outperform the pure CNN on validation loss |
| cVAE is currently the strongest learned model in reconstruction-oriented evidence | Strong | Best validation reconstruction loss among learned models and stable non-zero KL behavior |
| cVAE now has strong full-scale test evidence | Strong | 4,500-sample integrated evaluation with 0.02012 joint RMSE, 0.0 constraint violations, and strong held-out hexagon performance |
| Cross-model ranking still favors the cVAE, but remains provisional | Preliminary | Best learned subset-level joint RMSE / path-tracking behavior plus stronger standalone full-test cVAE evidence, but no uniform full-test benchmark yet for all models |
| Diffusion training is stable, but its final comparative standing remains unresolved | Strong for training stability, preliminary for ranking | Best diffusion validation loss is established, but full-test comparative evidence is still incomplete |
| Deployment-oriented conclusions are not yet ready | Preliminary / incomplete | Edge-deployment code exists, but benchmark artifacts are still missing |

### 6.8 Per-Shape Behavior

The new full-scale cVAE evaluation gives a much stronger per-shape picture than the earlier one-sample-per-shape subset:

1. the cVAE performs strongest on pentagon, circle, square, and the held-out hexagon class, all with joint RMSE around **0.013-0.015**,
2. line trajectories are moderately harder at **0.02245** joint RMSE,
3. random smooth trajectories are the clearest remaining challenge at **0.05495** joint RMSE.

This pattern is scientifically useful. It suggests that the model generalizes well across regular geometric families, including unseen hexagons, but remains more sensitive to the curvature variability and branch-selection complexity of random smooth trajectories.

Cross-model per-shape comparison is still only available from the earlier subset artifact, so claims about whether diffusion or the deterministic baselines share the same per-shape failure modes should still be treated as provisional.

Suggested final figure: `results/plots/cvae_full_4500_evaluation_dashboard.png`
Use this figure as the compact full-scale per-shape cVAE summary, while retaining the subset heatmap only as an auxiliary diagnostic if needed.

---

## 7. Discussion

### 7.1 Answering the Research Questions

The currently completed evidence allows provisional answers to the three research questions stated in the introduction. For **RQ1**, the answer is clearly yes: deterministic neural baselines can model the trajectory-generation task effectively, and recurrent sequence models substantially outperform a pure CNN regressor. For **RQ2**, the strongest available evidence now points even more clearly to yes: the cVAE not only improves validation reconstruction quality beyond the deterministic baselines, but also achieves strong full-test performance on 4,500 samples while retaining multi-sample generation. For **RQ3**, the current answer is more qualified: the completed artifacts suggest that the most favorable learned trade-off presently belongs to the cVAE, while diffusion remains promising but computationally expensive and not yet fully characterized on a uniform full benchmark.

### 7.2 Why the cVAE Currently Leads

The cVAE currently occupies the most favorable position in the available evidence for four reasons:

1. It achieves the best reconstruction-oriented validation result among all learned models.
2. It now has a completed 4,500-sample evaluation with strong accuracy and zero constraint violations.
3. It provides explicit multi-sample generation without the severe inference-time penalty of diffusion.
4. Its latent-variable structure appears stable and meaningful under the observed KL behavior.

This makes the cVAE an attractive compromise between accuracy, diversity, and deployability.

### 7.3 What the Diffusion Results Mean

The diffusion results should not be interpreted as a failure of the diffusion paradigm. Instead, they suggest that in the present configuration:

- inference remains expensive even with DDIM acceleration,
- the current sampling budget may be insufficient for this task-quality target,
- additional ablations on step count, guidance scale, and denoiser design are still needed.

The diffusion implementation is therefore a meaningful research component of the project, but its final comparative position remains unresolved until the full evaluation and ablation suite are run.

### 7.4 Scientific Caution

From a publication standpoint, the most important caveat is now narrower than before: the **cross-model test-time comparison is still preliminary**, even though the cVAE itself now has strong full-scale evidence. The training and validation results are strong enough to support the paper's methodological narrative, but broad claims such as "the cVAE is definitively best overall" or "diffusion underperforms on the full benchmark" would still be premature at this stage. A publishable final version of the paper should therefore include:

1. uniform full-test evaluation for deterministic, cVAE, and diffusion models,
2. complete per-shape statistical summaries,
3. trajectory-analysis outputs,
4. ablation studies for the generative models.

### 7.5 Implications for Final Submission Framing

The strongest defensible framing at the current stage is a paper centered on a reproducible methodology for one-to-many manipulator trajectory generation, supported by strong validation evidence, a full-test cVAE result, and preliminary comparative evaluation across the broader model family. That is already sufficient for a credible draft submission narrative. The stronger framing of a full benchmark paper comparing deterministic, latent-variable, and diffusion models on complete test-time evidence should be reserved for the next revision, after the remaining model families are evaluated under the same 4,500-sample protocol.

### 7.6 Practical Relevance and Broader Impact

Even in its current incomplete empirical state, the project suggests a practically relevant message: when inverse-kinematics ambiguity matters, a carefully designed generative model can provide meaningful diversity without necessarily sacrificing predictive quality. If this pattern holds under the full benchmark, it would support broader use of conditional generative models in robotics settings where multiple feasible motions exist and secondary decision criteria must be applied after prediction. At the same time, the present study remains deliberately conservative about impact claims because deployment benchmarking, controller-in-the-loop validation, and higher-dimensional robot experiments are not yet complete.

---

## 8. Limitations

This study currently has several limitations:

1. The manipulator is planar and low-dimensional, so conclusions may not transfer directly to high-DOF spatial manipulators.
2. The current task formulation does not include obstacles, collision constraints, or full rigid-body dynamics.
3. The strongest completed **cross-model** comparative evaluation is still subset-based rather than full-test-based.
4. The edge-deployment phase is implemented at the code level but not yet completed with final benchmark artifacts.
5. Multi-seed statistical analysis and ablation studies remain pending.

These limitations do not invalidate the current work; rather, they define the scope of the claims that can be made in the present manuscript version.

### 8.1 Threats to Validity

Several threats to validity should be kept in mind when interpreting the present manuscript. The most important internal-validity threat is no longer the absence of full cVAE evidence, but rather the absence of a **uniform full-test comparison across all learned models**, which means the relative positions of cVAE, diffusion, and the deterministic baselines could still shift when the same protocol is applied to every method. A second internal-validity threat is that the completed evidence is primarily single-run rather than multi-seed, so sensitivity to random initialization has not yet been fully characterized.

The main external-validity threat is task scope. The robot is planar, low-dimensional, and free of obstacles and full rigid-body dynamics, so conclusions should be interpreted as evidence about controlled manipulator trajectory generation rather than immediate proof of transfer to high-DOF or contact-rich systems. Finally, there is a construct-validity threat around evaluation emphasis: the current metrics cover reconstruction, tracking, smoothness, energy proxy, diversity, and latency, but the eventual practical value of a model may also depend on downstream controller integration and deployment constraints that are not yet fully benchmarked.

---

## 9. Future Work

The most important next step is to complete a uniform full evaluation on the 4,500-sample test set for the deterministic baselines and diffusion model, using the already prepared evaluation pipeline and the now-validated portable notebook workflow where needed. Once those runs are available, the manuscript should be updated with final full-test comparison tables, per-shape generalization summaries, and final oracle-versus-mean-sample reporting for stochastic models.

Beyond the main benchmark completion, the project is well positioned for three follow-on directions. First, trajectory-analysis outputs should be generated and integrated so that smoothness, energy, feasibility, and robustness trends can be discussed with the same level of rigor as reconstruction accuracy. Second, ablation studies should be executed for latent size, KL scheduling, decoder structure, diffusion step count, DDIM step count, and guidance scale to better explain why the current models behave as observed. Third, the edge-deployment toolchain should be completed with ONNX export and runtime benchmarking so that the paper can make evidence-backed claims about practical deployment trade-offs.

---

## 10. Reproducibility Statement

This work emphasizes reproducibility in both implementation and reporting. The repository contains:

1. explicit dataset-generation logic,
2. configuration-driven model definitions,
3. stored training histories,
4. evaluation-report generation scripts,
5. manuscript drafts aligned with current artifacts.

At the present stage, the main reproducibility gap is not missing code but incomplete execution of the final large-scale generative benchmark and downstream trajectory-analysis/deployment phases. The manuscript is therefore reproducible as a methods-and-preliminary-results paper now, and is positioned to become fully reproducible as a final comparison paper once those remaining runs are completed.

---

## 11. Data and Code Availability

The work is grounded in a repository that already contains the simulation pipeline, model implementations, training configurations, stored histories, evaluation scripts, manuscript drafts, and figure assets described in this paper. In the current project state, the main availability limitation is not missing source material but incomplete execution of the final large-scale generative benchmark and the still-pending trajectory-analysis and deployment-result artifacts. The final submission version should therefore cite the repository commit or archived release used for the completed benchmark so that the paper and code snapshot remain aligned.

---

## 12. Conclusion

This paper presents a reproducible framework for trajectory generation in a 3-link planar robotic manipulator, with emphasis on the one-to-many nature of the mapping from task-space paths to joint-space trajectories. Deterministic baselines establish that temporal sequence modeling is important, with CNN+LSTM and CNN+GRU clearly outperforming a pure CNN regressor. Building on this foundation, the cVAE demonstrates that a conditional latent-variable model can improve reconstruction quality substantially while also enabling diverse trajectory generation. A conditional diffusion model is also developed and trained successfully, extending the project into a second modern generative family.

Based on the currently completed artifacts, the cVAE is the most promising learned model in the project, offering the strongest balance between accuracy, diversity, and computational practicality. The diffusion model remains an important component of the study, especially for future research on richer multimodal generation, but currently requires more evaluation and optimization before stronger comparative claims are justified.

The next steps toward a final publishable paper are clear: complete the full generative test-set evaluation, run the trajectory-analysis framework, perform targeted ablations, and consolidate the results into final comparison tables and discussion. With those additions, the present draft can be elevated from a strong methods-and-preliminary-results manuscript into a complete empirical paper.

---

## 13. References

The following references are the primary sources that should anchor the final bibliography. The exact BibTeX entries can be added in the final manuscript preparation stage.

1. D. P. Kingma and M. Welling, *Auto-Encoding Variational Bayes*, arXiv:1312.6114, 2013.  
2. J. Ho, A. Jain, and P. Abbeel, *Denoising Diffusion Probabilistic Models*, arXiv:2006.11239, 2020.  
3. J. Song, C. Meng, and S. Ermon, *Denoising Diffusion Implicit Models*, arXiv:2010.02502, 2020.  
4. J. Ho and T. Salimans, *Classifier-Free Diffusion Guidance*, arXiv:2207.12598, 2022.  
5. C. Chi, Z. Xu, S. Feng, B. Burchfiel, and S. Song, *Diffusion Policy: Visuomotor Policy Learning via Action Diffusion*, Robotics: Science and Systems, 2023.  
6. B. Ames, J. Morgan, and G. Konidaris, *IKFlow: Generating Diverse Inverse Kinematics Solutions*, arXiv:2111.08933, 2021.  
7. J. Carvalho, A. T. Le, M. Baierl, D. Koert, and J. Peters, *Motion Planning Diffusion: Learning and Planning of Robot Motions with Diffusion Models*, arXiv:2308.01557, 2023.  
8. N. Ratliff, M. Zucker, J. Bagnell, and S. Srinivasa, *CHOMP: Gradient Optimization Techniques for Efficient Motion Planning*, ICRA, 2009.  
9. M. Kalakrishnan, S. Chitta, E. Theodorou, P. Pastor, and S. Schaal, *STOMP: Stochastic Trajectory Optimization for Motion Planning*, ICRA, 2011.  
10. J. Schulman, J. Ho, A. Lee, I. Awwal, H. Bradlow, and P. Abbeel, *Finding Locally Optimal, Collision-Free Trajectories with Sequential Convex Optimization*, RSS, 2013.  

---

## 14. Appendix A: Recommended Final Paper Figures

The following figure set is recommended for the final publishable version:

| Paper Role | Candidate File | Intended Use |
|------------|----------------|--------------|
| Robot/system overview | `figures/simulation/arm_diagram.png` | Introduce the 3-link planar manipulator and geometric setup |
| Dataset overview | `figures/simulation/shape_trajectories_grid.png` | Show the trajectory families used for dataset generation |
| IK ambiguity evidence | `figures/simulation/ik_branches.png` | Support the one-to-many inverse-kinematics motivation |
| Baseline evidence | `figures/baselines/training_curves.png` | Visualize deterministic convergence and recurrent-model advantage |
| cVAE qualitative result | `results/plots/cvae_full_4500_reconstruction_samples.png` | Show reconstruction quality and branch-consistent generation behavior from the integrated full-scale run |
| cVAE latent analysis | `results/plots/cvae_full_4500_latent_tsne.png` | Provide qualitative evidence of learned latent structure from the integrated full-scale run |
| Diffusion explanation | `figures/diffusion/denoising_progression.png` | Illustrate iterative denoising behavior |
| Final comparison | `results/plots/accuracy_latency_tradeoff.png` | Summarize the practical trade-off between prediction quality and runtime |

## 15. Appendix B: Recommended Final Tables

| Table Role | Current Source / Inputs |
|------------|-------------------------|
| dataset summary and splits | dataset generation outputs and Section 3 split counts |
| model architecture and parameter counts | configs, model definitions, and verified parameter-count scripts |
| main comparison metrics across all models | `results/metrics/summary.json` and `results/metrics/summary.md` |
| per-shape or generalization-specific breakdown | `results/plots/per_shape_joint_rmse_heatmap.png` plus final full-test summaries |
