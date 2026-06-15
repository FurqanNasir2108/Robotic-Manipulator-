# Phase 9: Thesis and Paper Packaging

> **Prerequisite:** Phase 8 complete. Read `00_main_orchestrator.instructions.md` first.

---

## Goal

Convert all code, results, and analysis into thesis-ready, paper-ready, and presentation-ready forms.

---

## Tasks

### 6.1 Repository Polish
- Verify all scripts run end-to-end from a clean clone
- Update `README.md` with:
  - Project overview
  - Setup instructions
  - How to generate data
  - How to train each model
  - How to evaluate
  - How to produce figures
  - License information
- Clean up dead code, unused files
- Verify `.gitignore` covers all generated artifacts
- Run linting (`flake8` or `ruff`) and fix issues

### 6.2 Thesis Preparation
Update `thesis_notes/outline.md` with:
- Chapter-by-chapter outline mapped to code modules
- Key results per chapter
- Figure references per chapter

Create chapter draft notes in `thesis_notes/chapter_drafts/`:
```
ch1_introduction.md     → motivation, contributions
ch2_literature.md       → related work references
ch3_robot_model.md      → kinematics, simulation details
ch4_baselines.md        → supervised methods, results
ch5_vae.md              → cVAE method, results, analysis
ch6_diffusion.md        → diffusion method, results, analysis
ch7_evaluation.md       → comparative evaluation, ablations
ch8_conclusion.md       → summary, contributions, future work
```

### 6.3 Paper Preparation
Create `thesis_notes/paper_drafts/paper_outline.md`:
- Title, abstract sketch
- Section mapping to thesis chapters
- Key figures to include (max 6-8 for conference paper)
- Key tables to include (max 3-4)
- Contribution bullet points

### 6.4 Presentation Preparation
- See `docs/PRESENTATION_GUIDE.md` for detailed slide structure
- Export key figures to `figures/` with publication resolution (300 DPI, PDF + PNG)
- Create a `scripts/export_figures.py` script that regenerates all figures from saved results

### 6.5 Experiment Log
Create/update `results/logs/experiment_log.md`:
- Date, experiment name, config used, result summary
- What worked, what didn't
- Hyperparameter choices and justification

### 6.6 Final Checklist Verification
Complete all items in `docs/14_repo_completion_checklist.md`

---

## Deliverables

- [ ] Polished `README.md`
- [ ] Clean, runnable repository
- [ ] Thesis outline with chapter drafts
- [ ] Paper outline with key figures/tables identified
- [ ] Presentation guide populated
- [ ] Experiment log up to date
- [ ] All figures exported at publication quality
- [ ] `CHANGELOG.md` up to date
- [ ] Repo completion checklist 100% done

---

## Go/No-Go Checks

| Check | Action if Failed |
|-------|-----------------|
| Scripts fail on clean clone | Fix missing dependencies, hardcoded paths, missing files |
| Figures look bad at publication size | Increase DPI, adjust font sizes, use vector format (PDF/SVG) |
| Thesis outline has empty chapters | Fill with bullet points from experimental notes; mark incomplete sections |
| README instructions unclear | Have someone else follow them; iterate on clarity |
