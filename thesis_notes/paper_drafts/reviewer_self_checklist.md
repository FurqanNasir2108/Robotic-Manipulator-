# Reviewer Self-Checklist

## Purpose

This note is a final pre-submission self-review aid for the paper bundle. It is written from the perspective of common reviewer questions and is intended to catch overclaiming, weak evidence links, or presentation gaps before the manuscript is finalized.

---

## 1. Claim Discipline

- Does every strong quantitative claim in the manuscript map to a concrete repository artifact?
- Are cVAE-versus-diffusion comparisons still clearly marked as preliminary where they rely on the 6-sample subset artifact?
- Are validation losses, best-epoch indices, and logged epochs reported consistently across manuscript, notes, and slides?
- Have we avoided implying multi-seed robustness or significance testing where those artifacts do not yet exist?

---

## 2. Standard Paper Quality

- Does the paper contain all core sections expected in a publishable manuscript?
- Is the novelty statement explicit and appropriately scoped?
- Are the research questions clear and answerable from the presented evidence?
- Are limitations and threats to validity stated honestly?
- Are reproducibility and data/code availability statements present?

---

## 3. Evidence Strength

- Are the deterministic baseline results backed by the stored history files?
- Are the cVAE and diffusion training claims backed by their stored history files?
- Are figure references tied to assets that actually exist in the repository?
- Is the final comparison language aligned with the current `results/metrics/summary.*` artifacts?

---

## 4. Submission Readiness

- Is the LaTeX draft the current source of truth for the submission candidate?
- Have captions been adapted from `figure_caption_bank.md` for every inserted figure?
- Has `evidence_traceability_matrix.md` been checked after the last manuscript edit?
- Has the final repository commit or archive identifier been recorded for the submitted version?

---

## 5. Final Gate

Before calling the paper complete, make sure the following are true:

1. the final 4,500-sample generative benchmark has been integrated if the manuscript claims final comparative results
2. the abstract matches the final evidence level
3. discussion and conclusion language match the strongest actually completed artifacts
4. all tables, figures, and numeric claims have been cross-checked one last time

---

## Bottom Line

If this checklist can be answered cleanly, the manuscript is much less likely to drift beyond the evidence that the repository actually supports.
