# Submission Package README

## Purpose

This note describes the current paper-writing bundle in practical terms: which files matter, how they relate to one another, and how to turn them into a final manuscript package once the remaining experiments are complete.

---

## Primary Draft Files

Use these as the main manuscript sources:

1. [paper_manuscript.tex](/c:/3_Link_Manipulator/thesis_notes/paper_drafts/paper_manuscript.tex)
2. [research_paper_draft.md](/c:/3_Link_Manipulator/thesis_notes/paper_drafts/research_paper_draft.md)
3. [references.bib](/c:/3_Link_Manipulator/thesis_notes/paper_drafts/references.bib)

The LaTeX draft should be treated as the primary submission-oriented manuscript. The Markdown draft is the easier working draft for fast editing and high-level review.

---

## Support Files

Use these files to guide final assembly and evidence control:

1. [paper_progress_notes.md](/c:/3_Link_Manipulator/thesis_notes/paper_drafts/paper_progress_notes.md)
2. [submission_readiness_checklist.md](/c:/3_Link_Manipulator/thesis_notes/paper_drafts/submission_readiness_checklist.md)
3. [final_submission_update_map.md](/c:/3_Link_Manipulator/thesis_notes/paper_drafts/final_submission_update_map.md)
4. [paper_asset_manifest.md](/c:/3_Link_Manipulator/thesis_notes/paper_drafts/paper_asset_manifest.md)
5. [figure_caption_bank.md](/c:/3_Link_Manipulator/thesis_notes/paper_drafts/figure_caption_bank.md)
6. [reporting_standards_note.md](/c:/3_Link_Manipulator/thesis_notes/paper_drafts/reporting_standards_note.md)
7. [evidence_traceability_matrix.md](/c:/3_Link_Manipulator/thesis_notes/paper_drafts/evidence_traceability_matrix.md)
8. [reviewer_self_checklist.md](/c:/3_Link_Manipulator/thesis_notes/paper_drafts/reviewer_self_checklist.md)
9. [title_abstract_options.md](/c:/3_Link_Manipulator/thesis_notes/paper_drafts/title_abstract_options.md)

Together, these files define:

- what the paper can already claim honestly,
- which figures and tables are already supported,
- what still depends on pending experiments,
- how to update the manuscript quickly once new results arrive,
- and exactly which repository artifacts back the main manuscript claims.

---

## Current Status of the Bundle

The package already supports a strong methods-and-preliminary-results paper with:

1. abstract, introduction, literature survey, and contributions
2. robot model, dataset, methodology, and training protocol
3. deterministic, cVAE, and diffusion validation evidence
4. preliminary subset-level generative comparison
5. limitations, threats to validity, future work, and reproducibility framing

What is still missing for a full final submission-grade empirical paper is:

1. full 4,500-sample cVAE generative evaluation
2. full 4,500-sample diffusion generative evaluation
3. final per-shape and comparison tables
4. trajectory-analysis outputs
5. deployment benchmark outputs, if deployment claims are retained

---

## Recommended Finalization Workflow

When the remaining results are available, use this order:

1. update `results/metrics/summary.*`-derived tables and plots
2. revise the Results section in both drafts
3. revise the Abstract and Discussion
4. update figure references using `paper_asset_manifest.md`
5. adapt captions using `figure_caption_bank.md`
6. perform a final claim audit using `submission_readiness_checklist.md`
7. freeze the LaTeX manuscript as the submission candidate

---

## Bottom Line

The paper bundle is now organized enough that the final phase should be mostly evidence integration, not manuscript reconstruction.
