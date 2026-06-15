# Reporting Standards Note

## Purpose

This note captures the current reporting conventions used across the paper drafts so that future edits stay numerically consistent and scientifically honest.

---

## 1. Validation Results

- Report deterministic baseline values from the stored training histories.
- Treat the listed validation numbers as best logged checkpoint results from the completed runs.
- Keep best-epoch index and total logged epochs distinct.

Current evidence-backed examples:

1. CNN: best validation loss `0.0196`, best-epoch index `131`, logged epochs `161`
2. CNN+LSTM: best validation loss `0.0054`, best-epoch index `53`, logged epochs `83`
3. CNN+GRU: best validation loss `0.0061`, best-epoch index `26`, logged epochs `56`
4. cVAE: best validation reconstruction loss `0.001590`, best epoch `226`, logged epochs `286`
5. Diffusion: best validation loss `0.007965`, best epoch `414`, logged epochs `494`

---

## 2. Test-Time Comparative Results

- Treat the currently completed generative comparison as preliminary because it is based on the 6-sample stratified subset artifact.
- Do not rewrite current drafts as though the full 4,500-sample comparison has already been completed.
- Any strong ranking claims should wait for the full benchmark.

---

## 3. Statistical Language

- Describe current numbers as descriptive and repository-evidence-backed.
- Do not imply confidence intervals, significance tests, or multi-seed robustness unless those artifacts have actually been generated.
- Use cautious wording such as `currently`, `preliminary`, `completed subset artifact`, and `based on available evidence` where appropriate.

---

## 4. Availability Language

- The repository already supports strong reproducibility claims for methods and completed artifacts.
- The main availability gap is incomplete execution of the final benchmark and downstream result-generation phases, not missing code.
- The final submission should cite a fixed repository snapshot, release, or commit hash once the benchmark is finalized.

---

## 5. Bottom Line

The paper is already strong enough to support a methods-and-preliminary-results narrative. This note exists to keep future revisions aligned with that evidence boundary until the remaining experiments are complete.
