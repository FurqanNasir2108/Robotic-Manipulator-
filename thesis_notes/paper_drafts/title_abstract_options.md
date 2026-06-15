# Title and Abstract Options

## Purpose

This note provides alternate title and abstract framings that are consistent with the current repository evidence. The goal is to make final manuscript positioning easier once the remaining experiments are complete.

---

## 1. Current Best-Evidence Framing

### Recommended title

**Generative Trajectory Modeling for a 3-Link Planar Robotic Manipulator: Comparing Deterministic Baselines, Conditional VAEs, and Diffusion Models**

### Recommended abstract

Trajectory generation for robotic manipulators is commonly formulated as a mapping from task-space motion requirements to joint-space trajectories. Even for low-degree-of-freedom systems, this mapping can be intrinsically one-to-many because multiple joint configurations can realize the same end-effector pose sequence. Deterministic regression models therefore face a structural limitation: they can recover a single representative trajectory, but they cannot naturally model the diversity of valid solutions. This work studies trajectory generation for a 3-link planar revolute manipulator as a conditional sequence generation problem. We build a reproducible research pipeline that includes analytical inverse kinematics, deterministic neural baselines (CNN, CNN+LSTM, CNN+GRU), a conditional variational autoencoder (cVAE), and a conditional diffusion model with DDPM/DDIM-style sampling. The dataset contains 27,000 trajectories across six geometric shape families, with a held-out hexagon split used for generalization analysis. The deterministic baselines achieve best validation losses of 0.0196 (CNN), 0.0054 (CNN+LSTM), and 0.0061 (CNN+GRU), while the cVAE attains a best validation reconstruction loss of 0.001590 and enables multi-sample generation. A conditional diffusion model is also implemented and trained to a best validation loss of 0.007965. Preliminary subset-level evaluation indicates that the cVAE currently provides the strongest learned trade-off between reconstruction quality, diversity, and inference speed among the completed artifacts, while the diffusion model remains substantially slower at inference time. These findings support the use of latent-variable generative models for manipulator trajectory generation, while also highlighting the importance of scalable evaluation and accelerated sampling for diffusion-based approaches.

---

## 2. Methods-and-Benchmark Framing

Use this if the final paper should emphasize reproducible comparison infrastructure even before the benchmark is fully expanded.

### Candidate titles

1. **A Reproducible Benchmark Pipeline for Generative Trajectory Modeling in a 3-Link Planar Manipulator**
2. **Benchmarking Deterministic and Generative Models for One-to-Many Manipulator Trajectory Generation**

### Abstract option

We present a reproducible benchmark pipeline for trajectory generation in a 3-link planar robotic manipulator, focusing on the one-to-many mapping from task-space waypoint trajectories to joint-space motion. The repository integrates analytical inverse kinematics, dataset generation, deterministic neural baselines, a conditional variational autoencoder (cVAE), and a conditional diffusion model under a shared training and evaluation framework. The dataset contains 27,000 trajectories spanning six shape families, including a held-out hexagon split for generalization analysis. Within the completed artifacts, recurrent deterministic baselines outperform a pure CNN regressor, the cVAE reaches a best validation reconstruction loss of 0.001590, and the diffusion model trains stably to a best validation loss of 0.007965. A completed stratified subset evaluation further suggests that the cVAE currently offers the best learned trade-off between accuracy, diversity, and inference cost, although full generative evaluation on the 4,500-sample test set is still pending. The main contribution of the study is therefore an evidence-backed and extensible experimental framework for comparing deterministic and modern generative approaches to manipulator trajectory generation.

---

## 3. cVAE-Leaning Framing

Use this only if the final full benchmark continues to support cVAE as the strongest practical learned model.

### Candidate titles

1. **Conditional Variational Autoencoders for Diverse Trajectory Generation in a 3-Link Planar Manipulator**
2. **Learning Diverse Manipulator Trajectories with Conditional Variational Autoencoders**

### Abstract option

This work studies diverse joint-space trajectory generation for a 3-link planar robotic manipulator under the inherently one-to-many mapping induced by inverse kinematics. We compare deterministic neural baselines with conditional generative models and find that a conditional variational autoencoder (cVAE) currently provides the most promising learned balance between reconstruction quality, diversity, and inference efficiency in the completed artifacts. The project is implemented as a reproducible pipeline containing analytical inverse kinematics, normalized dataset generation, deterministic baselines, a cVAE, and a conditional diffusion model. On a dataset of 27,000 trajectories across six shape families, the cVAE achieves a best validation reconstruction loss of 0.001590, improving substantially over the strongest deterministic baseline, while also enabling multi-sample trajectory generation. Preliminary subset-level evaluation further indicates that the cVAE outperforms the completed learned alternatives on joint-space and path-tracking quality while maintaining non-zero diversity. These findings motivate conditional latent-variable modeling as a practical approach for manipulator trajectory generation, while full-test comparative conclusions remain subject to the pending large-scale benchmark.

---

## 4. Full-Benchmark Framing Placeholder

Use this only after the final 4,500-sample benchmark is complete.

### Candidate titles

1. **Comparing Deterministic, Latent-Variable, and Diffusion Models for Manipulator Trajectory Generation**
2. **Full-Benchmark Evaluation of Generative Trajectory Models for a 3-Link Planar Manipulator**

### Abstract guidance

Once the full benchmark is complete:

1. replace all subset-level wording with full-test evidence
2. decide whether the headline result is:
   - cVAE practical superiority,
   - a more nuanced accuracy-diversity-latency trade-off,
   - or a stronger benchmark narrative rather than a single winner claim
3. update the abstract only after the discussion and final comparison tables are frozen

---

## 5. Recommendation

At the current project state, the safest and strongest choice remains the title and abstract in Section 1 of this file. It best matches the evidence already present in the repository and does not depend on future results being stronger than they are today.
