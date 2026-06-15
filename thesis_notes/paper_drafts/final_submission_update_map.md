# Final Submission Update Map

## Purpose

This note tells us exactly how to convert the current methods-and-preliminary-results paper draft into a full submission-grade manuscript once the remaining evaluation artifacts return from the external GPU run.

---

## 1. Sections That Will Need Immediate Revision

### Abstract

Replace the current "preliminary subset-level evaluation" sentence with the final full-test conclusion. Keep the validated deterministic, cVAE, and diffusion training numbers unless new reruns supersede them.

### Results

Update the following parts first:

1. the preliminary comparative evaluation table
2. the per-shape behavior subsection
3. any text describing cVAE versus diffusion at test time

### Discussion

Revisit these subsections after the full benchmark:

1. Why the cVAE Currently Leads
2. What the Diffusion Results Mean
3. Scientific Caution
4. Implications for Final Submission Framing

The final framing may shift from "methods plus preliminary results" to a full benchmark comparison paper depending on the completed evidence.

---

## 2. Exact Artifact Inputs Needed

Before revising the manuscript, collect the final versions of:

1. `results/metrics/summary.json`
2. `results/metrics/summary.md`
3. `results/plots/accuracy_latency_tradeoff.png`
4. `results/plots/generative_oracle_gap.png`
5. `results/plots/metric_grid.png`
6. `results/plots/per_shape_joint_rmse_heatmap.png`
7. trajectory-analysis outputs from `results/metrics/trajectory_analysis/`
8. any final deployment artifacts from `results/metrics/edge_deployment/`

If the full benchmark produces additional comparison plots, those should be considered alongside the existing list before locking figure selection.

---

## 3. Figure Replacement Plan

The final paper should prioritize a compact figure set:

1. manipulator/workspace figure
2. trajectory-shape dataset figure
3. baseline training or comparison figure
4. cVAE qualitative generation figure
5. diffusion qualitative or denoising figure
6. final model-comparison figure
7. accuracy-latency trade-off figure

The current manuscript should keep using placeholders and narrative references until the final figure choices are confirmed.

---

## 4. Claim Escalation Rules

The manuscript may safely escalate its claims only when the following evidence exists:

1. full 4,500-sample generative evaluation has completed successfully
2. cVAE and diffusion use the same test conditions and reporting protocol
3. per-shape summaries are available
4. latency reporting reflects the finalized benchmarking path
5. any deployment claim is backed by actual export and benchmark artifacts

Until then, keep the paper phrased as a strong methods-and-preliminary-results manuscript.

---

## 5. Fast Revision Order

When the new results return, revise in this order:

1. results tables
2. abstract
3. discussion
4. conclusion
5. figure references and captions
6. title, if the final evidence suggests a sharper framing

---

## 6. Bottom Line

The current drafts already support serious paper writing. This map exists so that once the remaining experiments are complete, we can update the manuscript quickly without re-auditing the whole repository from scratch.
