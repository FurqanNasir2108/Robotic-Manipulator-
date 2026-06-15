# Chapter 8: Conclusion and Future Work

## 8.1 Thesis Summary

This thesis studies trajectory generation for a 3-link planar robotic manipulator from the perspective of **one-to-many conditional generation**. The core motivation is that a single task-space trajectory may correspond to multiple valid joint-space trajectories because inverse kinematics has multiple solution branches. A deterministic regressor can approximate one trajectory, but it cannot naturally represent the space of alternatives. This motivated the comparison between deterministic baselines and conditional generative models.

The project produced a reproducible research codebase containing:

1. a full analytical kinematics and dataset-generation pipeline,
2. deterministic neural baselines,
3. a conditional VAE,
4. a conditional diffusion model,
5. an evaluation pipeline for accuracy, diversity, smoothness, energy, and latency,
6. draft thesis and paper materials grounded in actual repository evidence.

---

## 8.2 Main Technical Contributions

The strongest completed contributions of the project are:

### 8.2.1 Reproducible 3-Link Manipulator Pipeline

The thesis establishes a complete trajectory-generation stack for a 3-link planar manipulator, including:

- forward and inverse kinematics,
- trajectory generation for six shape families,
- branch-consistent joint-trajectory construction,
- dataset normalization and split control,
- figure generation and test coverage.

### 8.2.2 Deterministic Baseline Reproduction

Three deterministic baselines were implemented and benchmarked:

- CNN,
- CNN+LSTM,
- CNN+GRU.

These models show that sequence modeling is important, with the recurrent baselines clearly outperforming the pure CNN.

### 8.2.3 Conditional VAE as the Strongest Current Learned Model

The most important completed generative result is the conditional VAE. It now has:

- the best validation reconstruction result among learned models,
- a completed 4,500-sample test evaluation,
- zero constraint violations,
- strong held-out hexagon generalization,
- meaningful oracle-versus-mean-sample diversity behavior.

### 8.2.4 Diffusion as a Complete but Still Provisional Alternative

The thesis also implements a conditional diffusion model with:

- 1D U-Net denoiser,
- cosine noise scheduling,
- classifier-free guidance,
- DDPM and DDIM sampling,
- EMA checkpointing.

This is a meaningful contribution in itself, even though its final comparative standing remains unresolved.

---

## 8.3 Main Empirical Conclusions

Based on the strongest current repository evidence, the thesis can already support the following conclusions.

### 8.3.1 Deterministic Sequence Modeling Matters

CNN+LSTM and CNN+GRU substantially outperform the pure CNN baseline. This confirms that temporal structure is essential for accurate trajectory generation.

### 8.3.2 The cVAE Is a Stronger Approach Than the Current Baselines

The cVAE achieves:

- best validation reconstruction loss: **0.000774**,
- full-test joint RMSE: **0.02012**,
- held-out hexagon RMSE: **0.01543**,
- zero constraint violations.

Relative to the strongest deterministic baseline, the cVAE is clearly better on the shared validation objective and appears better on the currently available test-time evidence as well.

### 8.3.3 The One-to-Many Formulation Is Useful

The cVAE does not simply reconstruct one trajectory accurately. It also produces multiple candidates, and oracle selection over those samples gives a substantial improvement over average sample quality. This supports the thesis premise that joint-space trajectory generation for ambiguous task-space conditions benefits from a generative formulation.

### 8.3.4 Diffusion Is Scientifically Valuable but Not Yet the Practical Winner

The diffusion model is a serious research component and broadens the thesis beyond a single generative approach. However, on the currently available evidence it is slower and less compelling than the cVAE, and it still lacks a full integrated test evaluation.

---

## 8.4 What the Results Say About the Proposed Approach

The current approach should be considered **successful**.

That judgment is justified because:

1. the codebase achieves the original goal of moving beyond deterministic trajectory regression,
2. the cVAE improves on deterministic baselines rather than merely matching them,
3. the approach generalizes to an unseen shape family,
4. the generated trajectories remain feasible under the implemented constraints.

The main scientific caution is not that the approach looks weak, but that the **full all-model comparison has not been completed yet**. In other words, the thesis already has strong evidence that the cVAE approach works well, but it does not yet have final evidence for the exact ranking of every learned method under the same large-scale protocol.

---

## 8.5 Limitations

The main limitations of the current thesis state are:

1. the robot is planar and low-dimensional,
2. the task does not include obstacles or full rigid-body dynamics,
3. only the cVAE currently has integrated full-scale test evidence,
4. ablation studies are still pending,
5. trajectory-analysis artifacts and deployment benchmarks have not yet been fully executed.

These limitations do not invalidate the completed work, but they do narrow the claims that can be made in the final thesis draft.

---

## 8.6 Future Work

The next phase of the project should proceed in this order:

1. run uniform 4,500-sample evaluation for deterministic baselines and diffusion,
2. execute the trajectory-analysis pipeline and generate its reports and figures,
3. run the main ablation studies,
4. select a deployment candidate from the accuracy-latency trade-off,
5. complete ONNX export and edge benchmarking,
6. finalize the thesis discussion and conclusion around the completed evidence.

If these steps are completed, the thesis can move from a strong partial-results manuscript to a fully benchmarked generative trajectory modeling study.

---

## 8.7 Final Conclusion

The thesis has already reached an important milestone: the implemented generative approach is not only conceptually justified, but empirically strong. The cVAE now stands as the strongest learned model in the repository and provides convincing evidence that conditional generative modeling is a promising solution for one-to-many manipulator trajectory generation.

The remaining work is therefore no longer about proving that the idea is viable. It is about completing the final comparative, analytical, and deployment phases so that the thesis can make its conclusions with full benchmark-level confidence.
