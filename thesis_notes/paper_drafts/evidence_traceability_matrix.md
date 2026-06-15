# Evidence Traceability Matrix

## Purpose

This file maps the major sections and core quantitative claims of the paper drafts to the concrete repository artifacts that support them. It is intended to make future manuscript updates auditable and evidence-first.

---

## 1. Manuscript Section Traceability

| Paper Section | Main Claim / Purpose | Primary Evidence Sources |
|---------------|----------------------|--------------------------|
| Abstract | High-level summary of implemented pipeline and current results | `thesis_notes/project_status.md`, `results/metrics/baselines/*.json`, `results/metrics/cvae/cvae_history.json`, `results/metrics/diffusion/diffusion_history.json`, `results/metrics/summary.md` |
| Introduction | Motivation from one-to-many IK and need for generative modeling | `thesis_notes/chapter_drafts/ch3_robot_model.md`, `src/models/analytical_ik.py`, `figures/simulation/ik_branches.png` |
| Literature Survey / Related Work | Contextual framing against optimization, VAEs, diffusion, IKFlow, robotics diffusion | `thesis_notes/paper_drafts/references.bib`, manuscript text in both drafts |
| Robot Model and Dataset | 3-link planar manipulator, kinematics, shape families, split sizes | `thesis_notes/chapter_drafts/ch3_robot_model.md`, `thesis_notes/project_status.md`, `figures/simulation/arm_diagram.png`, `figures/simulation/shape_trajectories_grid.png`, `figures/simulation/dataset_distribution.png` |
| Deterministic Baselines | CNN, CNN+LSTM, CNN+GRU architectures and validation results | `src/models/cnn_baseline.py`, `src/models/cnn_lstm.py`, `src/models/cnn_gru.py`, `configs/baseline_cnn*.yaml`, `results/metrics/baselines/*.json`, `figures/baselines/*` |
| cVAE Method and Results | cVAE architecture, latent dimension, training objective, best validation reconstruction | `src/models/cvae.py`, `configs/vae.yaml`, `results/metrics/cvae/cvae_history.json`, `figures/vae/*`, `thesis_notes/chapter_drafts/ch5_vae.md` |
| Diffusion Method and Results | Diffusion architecture, denoising setup, DDIM usage, best validation loss | `src/models/diffusion.py`, `configs/diffusion.yaml`, `results/metrics/diffusion/diffusion_history.json`, `figures/diffusion/*`, `thesis_notes/chapter_drafts/ch6_diffusion.md` |
| Evaluation Protocol | Metrics, oracle-vs-mean reporting, subset/full-benchmark distinction | `scripts/evaluate_models.py`, `scripts/plot_results.py`, `src/evaluate/comparisons.py`, `results/metrics/summary.md`, `thesis_notes/project_status.md` |
| Preliminary Comparative Evaluation | Current 6-sample stratified subset comparison | `results/metrics/summary.json`, `results/metrics/summary.md`, `results/plots/*` |
| Discussion and Limitations | Current interpretation boundaries and unresolved empirical questions | `thesis_notes/project_status.md`, `submission_readiness_checklist.md`, `final_submission_update_map.md`, `reporting_standards_note.md` |
| Reproducibility / Availability | Repository-based reproducibility and current artifact availability | `paper_asset_manifest.md`, `submission_package_readme.md`, `reporting_standards_note.md`, repository file tree |

---

## 2. Core Quantitative Claim Traceability

| Claim in Drafts | Current Value | Evidence Source |
|-----------------|---------------|-----------------|
| Deterministic CNN best validation loss | `0.0196` | `results/metrics/baselines/cnn_history.json` |
| CNN+LSTM best validation loss | `0.0054` | `results/metrics/baselines/cnn_lstm_history.json` |
| CNN+GRU best validation loss | `0.0061` | `results/metrics/baselines/cnn_gru_history.json` |
| CNN best-epoch index / logged epochs | `131 / 161` | `results/metrics/baselines/cnn_history.json` |
| CNN+LSTM best-epoch index / logged epochs | `53 / 83` | `results/metrics/baselines/cnn_lstm_history.json` |
| CNN+GRU best-epoch index / logged epochs | `26 / 56` | `results/metrics/baselines/cnn_gru_history.json` |
| cVAE parameter count | `637,350` | verified parameter-count script output captured in manuscript work, `src/models/cvae.py`, `configs/vae.yaml` |
| cVAE best validation reconstruction loss | `0.001590` | `results/metrics/cvae/cvae_history.json` |
| cVAE best epoch / logged epochs | `226 / 286` | `results/metrics/cvae/cvae_history.json` |
| Diffusion parameter count | `1,425,606` | verified parameter-count script output captured in manuscript work, `src/models/diffusion.py`, `configs/diffusion.yaml` |
| Diffusion best validation loss | `0.007965` | `results/metrics/diffusion/diffusion_history.json` |
| Diffusion best epoch / logged epochs | `414 / 494` | `results/metrics/diffusion/diffusion_history.json` |
| Current comparative evaluation scope | `6-sample stratified subset` | `results/metrics/summary.md`, `thesis_notes/project_status.md` |
| Current best learned model on completed subset | `cVAE` | `results/metrics/summary.md` |

---

## 3. Claims That Must Stay Qualified

The following statements must remain explicitly qualified until new evidence is available:

1. any overall ranking between cVAE and diffusion on the full 4,500-sample test set
2. any per-shape generalization conclusion beyond the current subset artifact
3. any deployment-readiness conclusion
4. any multi-seed robustness claim
5. any significance-testing or confidence-interval language

Supporting caution files:

- [submission_readiness_checklist.md](/c:/3_Link_Manipulator/thesis_notes/paper_drafts/submission_readiness_checklist.md)
- [final_submission_update_map.md](/c:/3_Link_Manipulator/thesis_notes/paper_drafts/final_submission_update_map.md)
- [reporting_standards_note.md](/c:/3_Link_Manipulator/thesis_notes/paper_drafts/reporting_standards_note.md)

---

## 4. Recommended Use

When editing the paper after new experiments:

1. update the raw result artifacts first
2. verify the affected manuscript claims against those artifacts
3. revise the relevant manuscript sections
4. update this traceability matrix if any source-of-truth files change

---

## 5. Bottom Line

The paper drafts are now supported by a concrete evidence map rather than only narrative notes. This should make the final submission phase safer, faster, and easier to audit.
