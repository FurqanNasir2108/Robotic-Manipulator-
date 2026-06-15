# Report Generation Guide

This document provides a structured template for generating the thesis report and conference paper from this project's code and results.

---

## Report Types

| Type | Length | Format | Audience |
|------|--------|--------|----------|
| Master's Thesis | 60–100 pages | University LaTeX template | Committee |
| Conference Paper | 6–8 pages | IEEE/Springer template | Reviewers |
| Technical Report | 15–25 pages | Free format | Lab/advisor |
| Experiment Log | Running doc | Markdown | Self |

---

## Thesis Report Structure

### Chapter 1: Introduction (5–8 pages)

**Content to include:**
- Motivation for trajectory generation in robotics
- Limitations of classical methods (analytical IK, optimization-based)
- Why learning-based and generative approaches are promising
- Problem statement: one-to-many trajectory mapping
- Research questions:
  1. Can generative models produce diverse, valid manipulator trajectories?
  2. How do cVAE and diffusion models compare against supervised baselines?
  3. What is the trade-off between trajectory quality, diversity, and inference speed?
- Thesis contributions (3–4 bullet points)
- Thesis organization (brief chapter summary)

**Data sources:** `docs/01_project_overview.md`, `docs/02_problem_formulation.md`

### Chapter 2: Literature Review (10–15 pages)

**Sections to cover:**
1. Manipulator trajectory generation (classical methods)
2. Inverse kinematics (analytical, numerical, learning-based)
3. Deep learning for trajectory prediction (CNN, RNN, Transformer)
4. Variational Autoencoders (theory, conditional variants)
5. Diffusion models (DDPM, DDIM, applications in robotics)
6. Flow matching and other generative alternatives
7. Evaluation metrics for trajectory generation
8. Summary and gap analysis

**Data sources:** Literature search, `docs/04_baselines.md`, `docs/05_vae_method.md`, `docs/06_diffusion_method.md`

### Chapter 3: Robot Model and Data Generation (8–12 pages)

**Content to include:**
- 3-link planar manipulator description
  - DH parameters, link lengths, joint limits
  - Forward kinematics derivation
  - Inverse kinematics derivation (all solution branches)
  - Workspace analysis
- Trajectory shape definitions (circle, square, pentagon, hexagon, line, random)
- Dataset construction process
  - Number of samples, time steps, shapes
  - Train/val/test split strategy
  - Normalization approach
  - Data quality checks
- Figures: arm diagram, workspace plot, sample trajectories

**Data sources:** `src/simulation/`, `src/data/`, `configs/data.yaml`, `configs/simulation.yaml`

**Code to generate figures:**
```bash
python scripts/generate_data.py --config configs/data.yaml --visualize
```

### Chapter 4: Supervised Baseline Models (8–12 pages)

**Content per baseline:**
1. Architecture description with diagram
2. Input/output specification
3. Training details (loss function, optimizer, epochs, hyperparameters)
4. Results on test set (all metrics)
5. Sample trajectory visualizations
6. Discussion of strengths and limitations

**Models to cover:**
- Analytical IK reference
- CNN regressor
- CNN + LSTM
- CNN + GRU

**Data sources:** `src/models/`, `configs/baseline_*.yaml`, `results/metrics/baselines/`

**Code to generate results:**
```bash
python scripts/train_baselines.py --config configs/baseline_cnn.yaml
python scripts/evaluate_models.py --model baseline_cnn --output results/metrics/baselines/
```

### Chapter 5: Conditional VAE for Trajectory Generation (10–15 pages)

**Content to include:**
- VAE theory (ELBO, reparameterization trick)
- Conditional VAE formulation
- Architecture: encoder, decoder, conditioning
- Loss function: reconstruction + KL + energy + smoothness
- KL annealing strategy and justification
- Training procedure
- Results:
  - Reconstruction quality vs baselines
  - Generation diversity (sampling multiple trajectories)
  - Latent space visualization (t-SNE/UMAP)
  - Ablation: latent dimension, KL weight, decoder type
- Discussion: advantages over deterministic methods

**Data sources:** `src/models/cvae.py`, `configs/vae.yaml`, `results/metrics/`, `figures/vae/`

**Code to generate results:**
```bash
python scripts/train_vae.py --config configs/vae.yaml
python scripts/evaluate_models.py --model cvae --output results/metrics/vae/
```

### Chapter 6: Diffusion-Based Trajectory Generation (10–15 pages)

**Content to include:**
- Diffusion model theory (forward process, reverse process, noise prediction)
- Conditioning: classifier-free guidance
- Denoiser architecture (1D U-Net or Transformer)
- Sampling strategies: DDPM, DDIM, DPM-Solver
- Training procedure
- Results:
  - Sample quality vs VAE and baselines
  - Diversity analysis
  - Speed vs quality trade-off (sampling steps ablation)
  - Guidance scale ablation
  - Denoising process visualization
- Discussion: when diffusion outperforms and when it's overkill

**Data sources:** `src/models/diffusion.py`, `configs/diffusion.yaml`, `results/metrics/`, `figures/diffusion/`

**Code to generate results:**
```bash
python scripts/train_diffusion.py --config configs/diffusion.yaml
python scripts/evaluate_models.py --model diffusion --output results/metrics/diffusion/
```

### Chapter 7: Evaluation and Discussion (10–15 pages)

**Content to include:**
- Evaluation protocol (same test set, same seeds, statistical rigor)
- Main comparison table (all models × all metrics)
- Per-shape breakdown
- Generalization to unseen shapes
- Ablation study summary
- Key findings:
  - Which method wins on which metric?
  - Trade-offs between quality, diversity, and speed
  - When are generative models worth the complexity?
- Limitations of the study
- Threats to validity

**Data sources:** `results/metrics/summary.md`, `figures/comparison/`

**Code to generate summary:**
```bash
python scripts/evaluate_models.py --all --output results/metrics/
python scripts/plot_results.py --comparison --output figures/comparison/
```

### Chapter 8: Conclusion and Future Work (3–5 pages)

**Content:**
- Summary of contributions
- Answers to research questions
- Key takeaways
- Future work:
  - Higher-DOF manipulators (6-DOF, 7-DOF)
  - Real hardware validation
  - Obstacle avoidance integration
  - Online trajectory adaptation
  - Sim-to-real transfer

---

## Conference Paper Structure (6–8 pages)

### Recommended Focus
Pick **one** strong angle:
- **Option A:** cVAE vs supervised baselines (if VAE results are strong)
- **Option B:** Diffusion for trajectory generation (if diffusion results are strong)
- **Option C:** Comparative study: deterministic vs generative (if comparison tells a compelling story)

### Paper Sections

| Section | Length | Key Content |
|---------|--------|-------------|
| Abstract | 150–200 words | Problem, method, key result, conclusion |
| Introduction | 0.75 page | Motivation, gap, contribution |
| Related Work | 0.75 page | Position against closest work |
| Problem Formulation | 0.5 page | Robot model, I/O, challenge |
| Method | 1.5–2 pages | Architecture, loss, training details |
| Experiments | 1.5–2 pages | Setup, metrics, results tables, figures |
| Discussion | 0.5 page | Analysis, limitations |
| Conclusion | 0.25 page | Summary, future work |
| References | 0.5–1 page | 20–35 references |

### Key Figures for Paper (max 6–8)
1. Robot arm + task visualization
2. Model architecture diagram
3. Main comparison table
4. Trajectory overlay plot (best result)
5. Diversity visualization
6. Ablation plot (most important)
7. (Optional) Latent space or denoising visualization
8. (Optional) Generalization result

### Submission Targets

| Venue | Deadline | Type | Fit |
|-------|----------|------|-----|
| ICRA | Sep/Oct | Conference | Strong (robotics + learning) |
| IROS | Mar | Conference | Strong (robotics + learning) |
| CoRL | Jun | Conference | Good (robot learning focus) |
| RSS | Jan | Conference | Good (if strong results) |
| RA-L | Rolling | Journal | Good (extended version) |
| NeurIPS Workshop | Sep | Workshop | Good (generative modeling angle) |

---

## Automated Report Asset Generation

### Generate All Tables
```bash
python scripts/evaluate_models.py --all --format latex --output results/tables/
python scripts/evaluate_models.py --all --format markdown --output results/tables/
```

### Generate All Figures
```bash
python scripts/plot_results.py --all --dpi 300 --format pdf --output figures/
python scripts/export_figures.py --thesis --output figures/thesis/
```

### Generate Experiment Summary
```bash
python scripts/evaluate_models.py --summary --output results/metrics/summary.md
```

---

## Writing Tools

| Tool | Purpose | When to Use |
|------|---------|-------------|
| Overleaf | LaTeX editing | Thesis and paper writing |
| Zotero / Mendeley | Reference management | Literature review |
| draw.io | Architecture diagrams | Method figures |
| Matplotlib (code) | Result plots | Auto-generated from scripts |
| Grammarly | Proofreading | Before submission |

---

## Report Quality Checklist

- [ ] All figures are vector (PDF) or high-res (300 DPI PNG)
- [ ] All tables have consistent formatting and units
- [ ] All metrics are reported with mean ± std
- [ ] Equations are numbered and referenced in text
- [ ] All figures have captions and are referenced in text
- [ ] Code/config for every result is documented
- [ ] References are complete and consistently formatted
- [ ] No placeholder text remains
- [ ] Page count within limits (thesis: 60–100, paper: 6–8)
- [ ] Advisor has reviewed before submission
