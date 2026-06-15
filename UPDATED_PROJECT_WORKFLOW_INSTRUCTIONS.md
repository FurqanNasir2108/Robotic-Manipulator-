# UPDATED PROJECT WORKFLOW INSTRUCTIONS

Version: 2026-05-10
Project: Generative Trajectory Modeling and Analysis for a 3-Link Planar Robotic Manipulator with Edge Deployment

---

## 1. Purpose and Scope

This document is the consolidated operational workflow for the full thesis codebase after restructuring to a 9+T phase pipeline.

Primary goals:
- Preserve scientific correctness and reproducibility.
- Integrate trajectory analysis as a first-class stage before final evaluation and deployment.
- Integrate edge deployment as a first-class stage with explicit quality validation.
- Keep implementation, testing, and reporting synchronized.

This workflow supersedes the older 6+T phase flow.

---

## 2. Canonical Phase Architecture (9+T)

Phase flow:
1. Phase 1: Foundation and Simulation
2. Phase 2: Baselines (CNN, CNN+LSTM, CNN+GRU)
3. Phase 3: Conditional VAE
4. Phase 4: Diffusion
5. Phase 5: Intermediate Evaluation and Trajectory Caching
6. Phase 6: Trajectory Analysis Framework
7. Phase 7: Evaluation and Comparison (final comparative evaluation)
8. Phase 8: Edge Deployment and IoT Integration
9. Phase 9: Thesis and Paper Packaging
10. Phase T: Testing (continuous across all phases)

Strict dependency chain:
- Phase 6 requires raw predicted trajectories produced by Phase 5.
- Phase 7 must consume Phase 6 trajectory-analysis outputs.
- Phase 8 model selection must use Phase 6 + Phase 7 outcomes (Pareto viewpoint).
- Phase 9 must only package validated and tested outputs.

---

## 3. Entry Point and Instruction Loading Protocol

Always follow this exact reading order before making implementation changes:
1. docs/instructions/00_main_orchestrator.instructions.md
2. The active phase instruction file for your current task.
3. docs/instructions/phase_T_testing.instructions.md when adding or changing code.

Phase instruction files currently used:
- docs/instructions/phase_1_foundation_and_simulation.instructions.md
- docs/instructions/phase_2_baselines.instructions.md
- docs/instructions/phase_3_conditional_vae.instructions.md
- docs/instructions/phase_4_diffusion.instructions.md
- docs/instructions/phase_5_evaluation.instructions.md
- docs/instructions/phase_6_trajectory_analysis.instructions.md
- docs/instructions/phase_7_evaluation.instructions.md
- docs/instructions/phase_8_edge_deployment.instructions.md
- docs/instructions/phase_9_packaging.instructions.md
- docs/instructions/phase_T_testing.instructions.md

---

## 4. Data and Artifact Contracts

Core trajectory contract (per-sample semantics):
- condition: input task condition (waypoints and/or pose descriptors)
- shape_type: trajectory class label
- q_reference: ground-truth joint trajectory, shape (T, 3)
- q_predicted: model output trajectory, shape (T, 3) or (K, T, 3)
- x_reference and x_predicted: FK-mapped task-space trajectories

Primary artifact locations:
- data/processed/: training and test datasets
- results/checkpoints/: PyTorch training checkpoints
- results/generated_trajectories/: cached per-model predicted trajectories
- results/metrics/: model-wise and phase-wise metric outputs
- results/plots/: generated plot artifacts
- figures/: thesis-grade figure outputs

Critical rule:
- Never run Phase 6 analysis against ad-hoc in-memory predictions only. Cache and load from results/generated_trajectories/ to guarantee reproducibility.

---

## 5. Configuration Authority and File Ownership

Configuration files are the single source of truth for runtime behavior:
- configs/simulation.yaml: kinematic and dynamic limits
- configs/data.yaml: data generation and normalization behavior
- configs/baseline_cnn.yaml
- configs/baseline_cnn_lstm.yaml
- configs/baseline_cnn_gru.yaml
- configs/vae.yaml
- configs/diffusion.yaml
- configs/evaluation.yaml
- configs/trajectory_analysis.yaml
- configs/edge_deployment.yaml

Configuration policy:
- Do not hardcode model/device/limits where config entries exist.
- Add new parameters in config first, then wire into code.
- Keep default values safe for CPU fallback.

---

## 6. Source Module Boundaries

Simulation and data:
- src/simulation/: manipulator model, trajectory generation, optional physics adapters
- src/data/: datasets, loaders, normalization, augmentation

Models and training:
- src/models/: analytical IK, baselines, cVAE, diffusion, optional flow matching
- src/train/: trainer, losses, schedulers

Evaluation and analysis:
- src/evaluate/: common metrics, model comparison, ablations
- src/trajectory_analysis/: deep quality analysis modules

Deployment:
- src/edge_deployment/: export, optimization, runtime, benchmark, monitoring, sensor and communication interfaces

Utility and plotting:
- src/utils/
- src/visualization/

Boundary rule:
- Reuse existing metrics in src/evaluate/metrics.py where definitions overlap; extend without duplicating formulas.

---

## 7. Script-Orchestrated Execution Plan

Recommended order for full pipeline runs:
1. scripts/generate_data.py
2. scripts/train_baselines.py
3. scripts/train_vae.py
4. scripts/train_diffusion.py
5. scripts/evaluate_models.py
6. scripts/analyze_trajectories.py
7. scripts/compare_trajectory_methods.py
8. scripts/generate_trajectory_report.py
9. scripts/plot_trajectory_analysis.py
10. scripts/plot_results.py
11. scripts/generate_thesis_figures.py
12. scripts/export_model_to_onnx.py
13. scripts/benchmark_edge_inference.py
14. scripts/compare_pytorch_vs_onnx.py
15. scripts/profile_memory_latency.py

One-command orchestrator:
- scripts/run_all_experiments.py

Execution policy:
- Treat nonzero exit code as phase-blocking unless explicitly marked optional.
- Persist logs and metrics after each major stage.

---

## 8. Phase 6 Trajectory Analysis Operating Rules

Scope of analysis categories:
- Kinematic accuracy
- Smoothness and derivatives
- Dynamics and energy proxies
- Feasibility and constraint violations
- Diversity for generative models
- Robustness under perturbations
- Statistical confidence and pairwise significance

Expected outputs:
- Per-trajectory CSV
- Aggregate JSON
- Summary Markdown report
- Comparison tables and figures under figures/trajectory_analysis/

Mandatory numerical practices:
- Use consistent dt from config.
- Apply smoothing before higher-order differentiation.
- Compute best-of-K and mean-of-K separately for stochastic models.
- Mark failed trajectories explicitly.

---

## 9. Phase 7 Evaluation and Comparison Rules

Purpose separation:
- Phase 6 answers: Are outputs physically meaningful?
- Phase 7 answers: Which model is best under thesis objectives?

Evaluation must include:
- Overall metrics table per model
- Per-shape breakdown
- Accuracy vs efficiency tradeoffs
- Clear discussion of deterministic vs generative behavior

Use only:
- Common test split
- Common normalization
- Common metric definitions

---

## 10. Phase 8 Edge Deployment and IoT Rules

Primary deployment targets:
- cVAE as first deployment candidate
- Baselines as low-complexity references
- Diffusion as optional advanced path

Export constraints:
- cVAE: export decode(z, condition), not stochastic generate loop
- Diffusion: export denoiser step, keep iterative schedule in Python

Edge metrics to report:
- mean and percentile latency
- throughput
- model size
- RAM/GPU memory usage
- utilization metrics (if available)
- PyTorch vs ONNX output deviation
- trajectory quality retention post-export/quantization

Fallback constraints:
- CPU-only ONNX Runtime must always work.
- TensorRT is optional and must be guarded.

---

## 11. Model Selection Policy (Scientific and Deployment)

Selection is multi-objective and phase-coupled.

Use these objective groups:
- Accuracy: joint/task-space fidelity
- Physical quality: smoothness, feasibility, energy
- Diversity and robustness: for generative candidates
- Runtime constraints: latency, memory, size

Decision process:
1. Reject models violating feasibility/safety thresholds.
2. Rank remaining models using weighted comparison from thesis priorities.
3. Build Pareto front on accuracy-latency-size for deployment.
4. Choose deployment candidate with best practical tradeoff, not only best raw RMSE.

---

## 12. Testing Strategy (Phase T Across All Phases)

Test layers:
- Unit tests for metric correctness, export wrappers, utilities.
- Integration tests for full analysis and deployment pipelines.
- Regression tests for known trajectory behavior.

Mandatory new tests already mapped:
- tests/unit/test_trajectory_analysis.py
- tests/unit/test_edge_deployment.py
- tests/integration/test_trajectory_analysis_pipeline.py
- tests/integration/test_edge_pipeline.py
- tests/integration/test_realtime_loop.py

Gate rule:
- No phase considered complete without passing relevant tests.

---

## 13. Reproducibility and Experiment Hygiene

Required controls:
- Seed all random sources (numpy, torch, python random where used).
- Log config snapshots per run.
- Keep deterministic preprocessing pipelines.
- Save raw metrics before aggregation.

Tracking:
- Update CHANGELOG.md after meaningful modifications.
- Keep run metadata traceable to exact config + checkpoint + script versions.

---

## 14. Documentation and Figure Production Workflow

Documentation outputs:
- results/metrics/* summary artifacts
- thesis_notes/outline.md and chapter drafts
- docs/PRESENTATION_GUIDE.md and docs/REPORT_GENERATION_GUIDE.md alignment

Figure outputs should remain method-scoped:
- figures/simulation/
- figures/baselines/
- figures/vae/
- figures/diffusion/
- figures/comparison/
- figures/trajectory_analysis/
- figures/edge_deployment/

Quality rule:
- Every thesis figure must be reproducible from scripts and source metrics.

---

## 15. Risk Register and Mitigation Defaults

Common risks and defaults:
- Export incompatibility: isolate unsupported ops and simplify wrappers.
- Quantization degradation: fall back from INT8 to FP16 and report delta.
- Hardware unavailability: run CPU fallback path and document as simulated edge.
- Sensor/communication instability: degrade to local simulated stream and REST baseline.

Scientific integrity rule:
- Never hide failure cases. Report limitations and mitigation attempts explicitly.

---

## 16. Definition of Done per Phase

A phase is complete only if all are true:
- Implemented code exists in expected module paths.
- Config entries are present and used.
- Outputs are generated in expected results/figures directories.
- Tests pass at relevant layer(s).
- Changelog and reporting artifacts are updated.

For cross-phase dependencies:
- Upstream artifacts must be version-consistent with downstream runs.

---

## 17. Operational Checklist (Daily and Full-Run)

Daily cycle:
1. Pull latest workspace state.
2. Confirm active phase and instruction file.
3. Implement smallest coherent change set.
4. Run targeted tests.
5. Run affected scripts.
6. Verify output files and metrics.
7. Update changelog and notes.

Full-run cycle:
1. Execute scripts/run_all_experiments.py.
2. Review failures and rerun failed stages after fixes.
3. Regenerate summary reports and figures.
4. Validate final artifact completeness against phase deliverables.

---

End of document.
