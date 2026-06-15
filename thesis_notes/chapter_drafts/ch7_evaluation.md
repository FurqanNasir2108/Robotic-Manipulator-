# Chapter 7: Evaluation and Comparative Analysis

## 7.1 Introduction

This chapter summarizes the current evaluation state of the project and compares the implemented methods using the strongest evidence available in the repository. The evaluation phase is designed to answer two questions:

1. how accurately can each model reproduce or generate valid joint-space trajectories for the same task-space condition?
2. how favorable is the trade-off between accuracy, diversity, smoothness, and inference cost?

The key scientific caution in this chapter is that the codebase currently contains **two different levels of evidence**:

- a completed **4,500-sample full-test evaluation** for the cVAE,
- an earlier **6-sample stratified cross-model comparison** for analytical IK, the deterministic baselines, the cVAE, and diffusion.

This means the cVAE can now be judged strongly on full-test evidence, but the final cross-model ranking is still provisional until the remaining learned models are evaluated under the same full protocol.

---

## 7.2 Evaluation Protocol

### 7.2.1 Shared Metrics

The implemented evaluation pipeline computes the following metrics:

1. joint-space RMSE,
2. end-effector position error,
3. end-effector orientation error,
4. path tracking error,
5. energy proxy,
6. smoothness (jerk-based),
7. inference latency,
8. diversity score for generative models,
9. constraint violation rate.

For generative models, the pipeline draws $K = 10$ samples per condition and reports both:

- **oracle metrics**: best sample among the $K$ candidates,
- **mean-sample metrics**: average quality across the $K$ samples.

Oracle selection uses the lowest joint-space RMSE. Diversity is computed as a mean pairwise trajectory-distance statistic over the generated set.

### 7.2.2 Artifact Layers

The current repository state contains two evaluation artifact layers:

| Artifact | Scope | Status |
|----------|-------|--------|
| `results/metrics/summary.md` | 6-sample stratified cross-model report | Complete |
| `results/metrics/cvae/cvae_full_4500_evaluation_summary.json` | 4,500-sample cVAE full-test report | Complete |

The subset artifact remains valuable because it places all currently trained model families in one table. The full cVAE artifact is more important scientifically because it provides the first thesis-grade large-scale generative evaluation in the repository.

---

## 7.3 Deterministic Baseline Status

The deterministic baselines establish the main non-generative reference point for the project.

### 7.3.1 Validation Evidence

| Model | Best Validation Loss | Best Epoch / Logged Epochs |
|-------|---------------------:|---------------------------:|
| CNN | 0.0196 | 131 / 161 |
| CNN+LSTM | **0.0054** | 53 / 83 |
| CNN+GRU | 0.0061 | 26 / 56 |

These results clearly show that temporal sequence modeling matters. The recurrent baselines strongly outperform the pure CNN, with CNN+LSTM providing the best deterministic validation performance.

### 7.3.2 Current Cross-Model Subset Metrics

The currently archived cross-model subset report gives the following overall results:

| Model | Joint RMSE | EE Pos | EE Orient | Path | Energy | Smoothness | Inference (ms) |
|-------|-----------:|-------:|----------:|-----:|-------:|-----------:|---------------:|
| CNN | 0.0556 | 0.0715 | 0.0227 | 0.0399 | 0.0004 | 0.0000 | **1.22** |
| CNN+LSTM | **0.0314** | 0.0414 | 0.0262 | 0.0218 | 0.0006 | 0.0003 | 2.55 |
| CNN+GRU | 0.0384 | 0.0454 | 0.0218 | 0.0266 | 0.0006 | 0.0001 | 27.05 |

Within the deterministic family, CNN+LSTM is the strongest current accuracy baseline, while the pure CNN is the fastest.

---

## 7.4 cVAE Evaluation Results

### 7.4.1 Validation Comparison Against Baselines

The cVAE achieves a best validation reconstruction loss of **0.000774**, which is:

- **25.32× better** than CNN,
- **6.98× better** than CNN+LSTM,
- **7.88× better** than CNN+GRU.

This is already strong evidence that the generative approach is not hurting the underlying prediction objective.

### 7.4.2 Full 4,500-Sample cVAE Results

The integrated full-test cVAE artifact reports:

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

These are very strong results for the current project stage. In particular:

- the error levels are low,
- the constraint violation rate is zero,
- the inference cost is small in the Colab evaluation environment,
- the model still preserves a meaningful best-of-$K$ advantage.

### 7.4.3 Oracle Versus Mean-Sample Behavior

The cVAE full-test artifact also reports:

- mean-sample joint RMSE: **0.03436**
- oracle-sample joint RMSE: **0.02012**

This is a **41.43% improvement** from mean-sample to oracle-sample quality. That gap is scientifically important because it confirms that the model is producing multiple distinct candidates and that a downstream selector could exploit them.

### 7.4.4 Generalization to Held-Out Hexagons

Per-shape cVAE joint RMSE values show:

| Shape | Joint RMSE |
|-------|-----------:|
| Pentagon | 0.01311 |
| Circle | 0.01420 |
| Square | 0.01465 |
| Hexagon (held out) | **0.01543** |
| Line | 0.02245 |
| Random Smooth | 0.05495 |

The held-out hexagon result is one of the strongest findings of the current thesis evidence. It suggests that the cVAE is learning a trajectory prior that generalizes beyond the exact geometric families seen during training.

---

## 7.5 Diffusion Evaluation Status

The diffusion model is implemented, trained, and visually documented, but its evaluation status remains weaker than the cVAE.

On the current 6-sample cross-model subset:

| Metric | Value |
|--------|------:|
| Oracle joint RMSE | 0.0997 |
| Mean joint RMSE | 0.2629 |
| Diversity score | 0.1639 |
| Inference time | 785.66 ms |

These numbers show that diffusion is functioning as a stochastic model, but they are not currently competitive with the cVAE on the available evidence. More importantly, the diffusion model does **not yet** have the same integrated 4,500-sample evaluation that the cVAE now has, so its final ranking remains unresolved.

---

## 7.6 Are the Results Good Compared with the Baselines?

The answer is **yes**, with an important fairness caveat.

### 7.6.1 Why the cVAE Results Are Good

The cVAE is performing well relative to the deterministic baselines for three separate reasons:

1. it achieves the best validation objective in the entire learned model stack,
2. it now has strong full-test evidence with low error and zero violations,
3. it retains generative diversity and best-of-$K$ gains rather than collapsing to a purely deterministic predictor.

### 7.6.2 Quantitative Comparison

The strongest currently justified comparisons are:

- **validation:** cVAE clearly outperforms all deterministic baselines,
- **current test-time evidence:** cVAE full-run joint RMSE `0.02012` is lower than the best deterministic subset RMSE `0.0314`.

That provisional test-time margin is approximately **35.99%** relative to the best deterministic model currently recorded in the cross-model summary.

### 7.6.3 Fairness Caveat

The caution is that this is **not yet a uniform full-test comparison**:

- the cVAE has a full integrated 4,500-sample evaluation,
- the deterministic baselines and diffusion still rely mainly on the earlier subset-level cross-model report.

So the correct thesis wording is:

> The current results strongly suggest that the cVAE approach is better than the deterministic baselines and substantially stronger than the currently evaluated diffusion setup, but the final all-model ranking remains provisional until the remaining learned models are evaluated under the same full-test protocol.

---

## 7.7 Current Limits of the Evaluation Phase

Several important tasks remain before this chapter can be treated as final:

1. full integrated evaluation for the deterministic baselines,
2. full integrated evaluation for diffusion,
3. trajectory-analysis artifact generation,
4. ablation studies,
5. multi-seed statistical reporting.

Because of these remaining tasks, the current chapter is best read as a strong partial evaluation chapter rather than the final benchmark chapter of the thesis.

---

## 7.8 Summary

The evaluation codebase is mature enough to support a serious comparative thesis chapter, and the cVAE results are already strong. The most important conclusion from the current evidence is that the cVAE is the strongest learned approach in the repository at this stage:

- best validation reconstruction among learned models,
- strong full-test accuracy,
- zero constraint violations,
- strong held-out hexagon generalization,
- meaningful best-of-$K$ behavior.

The main remaining task is not to prove that the cVAE works, because that is already well supported, but to complete the **uniform full-test comparison** for the rest of the model family so that the final thesis ranking can be stated without reservation.
