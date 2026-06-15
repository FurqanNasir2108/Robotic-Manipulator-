# Phase 6: Trajectory Analysis Framework

> **Prerequisite:** Phases 1–5 complete. Read `00_main_orchestrator.instructions.md` first.

---

## Goal

Implement a comprehensive trajectory analysis framework that scientifically evaluates whether generated trajectories are accurate, smooth, feasible, energy-efficient, diverse, and suitable for deployment.

This phase answers: **"Are these trajectories physically meaningful?"** — distinct from Phase 7 (Evaluation) which answers **"Which model is best?"**

---

## Key Definitions

| Term | Definition |
|------|-----------|
| **Path** | Geometric curve in space without time parameterization |
| **Trajectory** | Time-parameterized path: position as a function of time |
| **Joint-space trajectory** | `q(t) = [q1(t), q2(t), q3(t)]` — joint angles over time |
| **Task-space trajectory** | `x(t) = [x_e(t), y_e(t), θ(t)]` — end-effector pose over time |
| **Kinematic trajectory** | Trajectory described by position, velocity, acceleration (no forces) |
| **Dynamic trajectory** | Trajectory that includes torques/forces via inverse dynamics |

### Notation

- `q(t) = [q1(t), q2(t), q3(t)]` — joint angles at time t
- `Q` — full joint trajectory sequence, shape `(T, 3)`
- `x(t) = [x_e(t), y_e(t), θ(t)]` — end-effector pose at time t
- `C` — task condition (shape type, waypoints, goal pose, or shape parameters)

---

## Relationship to Overall Workflow

```
Phase 5 (Diffusion) → Phase 6 (Trajectory Analysis) → Phase 7 (Evaluation)
                            ↓
            Produces per-trajectory quality reports
            that feed into Phase 7 comparison tables
            and Phase 8 edge deployment model selection
```

Trajectory analysis operates on **generated trajectories from all models**: IK, CNN, CNN+LSTM, CNN+GRU, cVAE, and Diffusion. It requires raw predicted trajectories to be cached on disk by the evaluation pipeline.

---

## Tasks

### 6.1 Trajectory Data Format and Loading

Define a standard trajectory sample format:

```python
{
    "sample_id": int,
    "condition": np.ndarray,        # waypoints or goal pose
    "shape_type": str,              # circle, square, pentagon, hexagon, line, random
    "waypoints": np.ndarray,        # (W, 3) waypoint sequence
    "q_reference": np.ndarray,      # (T, 3) ground-truth joint trajectory
    "q_predicted": np.ndarray,      # (T, 3) or (K, T, 3) for generative models
    "x_reference": np.ndarray,      # (T, 3) GT task-space trajectory via FK
    "x_predicted": np.ndarray,      # (T, 3) or (K, T, 3) predicted task-space via FK
    "model_name": str,
    "split": str,                   # test
    "metadata": dict,               # any additional info
}
```

Supported file formats:
- NumPy `.npz` (primary)
- CSV (for per-trajectory metrics export)
- JSON (for metadata and summary reports)
- PyTorch tensors (for direct model output)

Implement in `src/trajectory_analysis/utils.py`:
- `load_generated_trajectories(model_name, results_dir) -> dict`
- `compute_task_space_trajectory(q_trajectory, fk_fn) -> np.ndarray`
- `finite_difference(signal, dt, order) -> np.ndarray` — velocity, acceleration, jerk
- `smooth_trajectory(trajectory, window, method) -> np.ndarray` — Savitzky-Golay or moving average
- `validate_trajectory_format(sample) -> bool`

### 6.2 Kinematics Metrics

Implement in `src/trajectory_analysis/kinematics_metrics.py`:

**Task-space accuracy:**
```python
def final_ee_position_error(q_pred, q_ref, fk_fn) -> float
def final_ee_orientation_error(q_pred, q_ref, fk_fn) -> float
def path_rmse_taskspace(q_pred, q_ref, fk_fn) -> float
def max_path_deviation(q_pred, q_ref, fk_fn) -> float
```

**Joint-space accuracy:**
```python
def joint_angle_rmse(q_pred, q_ref) -> float
def joint_velocity_rmse(q_pred, q_ref, dt) -> float
def joint_acceleration_rmse(q_pred, q_ref, dt) -> float
```

Reuse `joint_rmse`, `ee_position_error`, `ee_orientation_error` from `src/evaluate/metrics.py` where applicable. Extend — do not duplicate.

### 6.3 Smoothness Metrics

Implement in `src/trajectory_analysis/smoothness_metrics.py`:

```python
def velocity_profile(trajectory, dt) -> np.ndarray          # (T-1, 3)
def acceleration_profile(trajectory, dt) -> np.ndarray       # (T-2, 3)
def jerk_profile(trajectory, dt) -> np.ndarray               # (T-3, 3)
def integrated_squared_acceleration(trajectory, dt) -> float # ISA
def integrated_squared_jerk(trajectory, dt) -> float         # ISJ
def peak_acceleration(trajectory, dt) -> float
def peak_jerk(trajectory, dt) -> float
def smoothness_summary(trajectory, dt) -> dict               # all metrics in one call
```

### 6.4 Energy and Dynamics Metrics

Implement in `src/trajectory_analysis/energy_metrics.py` and `dynamics_metrics.py`:

```python
# energy_metrics.py
def velocity_squared_cost(trajectory, dt) -> float
def torque_squared_cost(trajectory, dt, dynamics_fn) -> float
def mechanical_power(trajectory, dt, dynamics_fn) -> np.ndarray  # power(t)
def positive_mechanical_work(trajectory, dt, dynamics_fn) -> float
def peak_power(trajectory, dt, dynamics_fn) -> float
def energy_per_trajectory(trajectory, dt, dynamics_fn) -> float

# dynamics_metrics.py
def compute_gravity_torque(q, manipulator) -> np.ndarray
def compute_inertia_matrix(q, manipulator) -> np.ndarray
def inverse_dynamics_proxy(q, qd, qdd, manipulator) -> np.ndarray
```

Note: For a planar horizontal manipulator, gravity torque is zero (g acts perpendicular to plane). The dynamics model uses simplified rigid-body parameters from `configs/trajectory_analysis.yaml`.

### 6.5 Feasibility Checks

Implement in `src/trajectory_analysis/feasibility_checks.py`:

```python
def joint_limit_violation_rate(trajectory, joint_limits) -> float
def velocity_limit_violation_rate(trajectory, dt, velocity_limits) -> float
def acceleration_limit_violation_rate(trajectory, dt, accel_limits) -> float
def torque_limit_violation_rate(trajectory, dt, torque_limits, dynamics_fn) -> float
def workspace_reachability_rate(trajectory, fk_fn, workspace_bounds) -> float
def trajectory_success_rate(trajectories, checks_config) -> float
def feasibility_summary(trajectory, dt, config) -> dict
```

### 6.6 Diversity Metrics (Generative Models Only)

Implement in `src/trajectory_analysis/diversity_metrics.py`:

```python
def pairwise_trajectory_diversity(samples, method='dtw') -> float
def best_of_k_accuracy(samples, reference, k_values, metric_fn) -> dict
def feasible_diversity(samples, feasibility_fn, method='dtw') -> float
def diversity_vs_accuracy_tradeoff(samples, reference, metric_fn) -> dict
def repeatability_score(model, condition, n_runs, metric_fn) -> float
```

Reuse `diversity_score` from `src/evaluate/metrics.py` as the base pairwise computation.

### 6.7 Robustness Tests

Implement in `src/trajectory_analysis/robustness_tests.py`:

```python
def evaluate_noisy_input(model, condition, noise_levels, metric_fn) -> dict
def evaluate_perturbed_start(model, condition, perturbation, metric_fn) -> dict
def evaluate_changed_shape_params(model, conditions_variants, metric_fn) -> dict
def robustness_summary(model, test_config) -> dict
```

These tests require re-running model inference with perturbed inputs and comparing metric degradation.

### 6.8 Statistical Analysis

Implement in `src/trajectory_analysis/statistical_tests.py`:

```python
def compute_statistics(values) -> dict  # mean, std, median, P50, P95, P99, CI
def wilcoxon_signed_rank(values_a, values_b) -> dict  # p-value, statistic
def paired_comparison(model_a_metrics, model_b_metrics) -> dict
def confidence_interval(values, level=0.95) -> tuple
```

### 6.9 Report Generation

Implement in `src/trajectory_analysis/trajectory_report.py`:

```python
def generate_per_trajectory_csv(all_metrics, output_path)
def generate_aggregate_json(all_metrics, output_path)
def generate_per_shape_report(all_metrics, output_path)
def generate_per_model_report(all_metrics, output_path)
def generate_comparison_table(all_metrics, output_path)
def generate_analysis_summary_md(all_metrics, output_path)
```

All raw per-trajectory metrics must be saved before computing averages.

---

## Required Visualizations

Generate and save to `figures/trajectory_analysis/`:

| Figure | Description |
|--------|-------------|
| `desired_vs_generated_taskspace.png` | Desired vs generated task-space path per model |
| `joint_angle_vs_time.png` | q1, q2, q3 time series for sample trajectories |
| `velocity_vs_time.png` | Joint velocity profiles |
| `acceleration_vs_time.png` | Joint acceleration profiles |
| `jerk_vs_time.png` | Joint jerk profiles |
| `energy_power_vs_time.png` | Power(t) and cumulative energy |
| `accuracy_vs_energy_scatter.png` | Accuracy-energy trade-off per model |
| `smoothness_vs_accuracy.png` | ISJ vs RMSE scatter |
| `diversity_histogram.png` | Distribution of pairwise distances |
| `feasibility_violation_barchart.png` | Violation rate per model per type |
| `model_comparison_table.png` | Rendered comparison table |
| `robustness_degradation.png` | Metric vs noise level |
| `per_shape_analysis_grid.png` | Shape-wise metric breakdown |

---

## Phase-Wise Sub-Tasks

### Sub-Phase 6.1: Data Format and Loading
- **Goal:** Define and implement trajectory data loading
- **Tasks:** Implement `utils.py` with loading, FK wrapper, finite differences, smoothing
- **Output:** Loadable trajectory data from all models
- **Completion check:** Can load trajectories from all 6 models, compute FK, compute derivatives

### Sub-Phase 6.2: Kinematics Accuracy Metrics
- **Goal:** Implement task-space and joint-space accuracy metrics
- **Tasks:** Implement `kinematics_metrics.py`
- **Output:** 7 accuracy metrics per trajectory
- **Completion check:** Metrics match expected values on known test cases

### Sub-Phase 6.3: Smoothness and Derivative Metrics
- **Goal:** Implement smoothness analysis
- **Tasks:** Implement `smoothness_metrics.py`
- **Output:** 7 smoothness metrics + profiles per trajectory
- **Completion check:** ISA/ISJ correct for synthetic smooth vs jerky trajectories

### Sub-Phase 6.4: Energy and Dynamics Metrics
- **Goal:** Implement energy and torque-based metrics
- **Tasks:** Implement `energy_metrics.py` and `dynamics_metrics.py`
- **Output:** 6 energy metrics per trajectory
- **Completion check:** Energy metrics are non-negative; power integral matches work

### Sub-Phase 6.5: Feasibility Checks
- **Goal:** Implement constraint violation detection
- **Tasks:** Implement `feasibility_checks.py`
- **Output:** 6 feasibility metrics per trajectory
- **Completion check:** Analytical IK trajectories have 0% violation; deliberately out-of-bounds trajectories have >0%

### Sub-Phase 6.6: Diversity Metrics
- **Goal:** Implement generative-model diversity analysis
- **Tasks:** Implement `diversity_metrics.py`
- **Output:** 5 diversity metrics for generative models
- **Completion check:** Deterministic models return diversity=0; generative models return diversity>0

### Sub-Phase 6.7: Robustness Tests
- **Goal:** Implement input perturbation experiments
- **Tasks:** Implement `robustness_tests.py`
- **Output:** Robustness degradation curves per model
- **Completion check:** Metrics degrade monotonically with increasing noise

### Sub-Phase 6.8: Statistical Analysis and Reporting
- **Goal:** Implement statistical tests and report generation
- **Tasks:** Implement `statistical_tests.py` and `trajectory_report.py`
- **Output:** Per-trajectory CSV, aggregate JSON, summary MD, comparison tables
- **Completion check:** Reports contain all metrics for all models with statistical annotations

### Sub-Phase 6.9: Visualizations
- **Goal:** Generate all trajectory analysis figures
- **Tasks:** Implement plotting in scripts, generate 13 figure types
- **Output:** 13+ publication-quality figures in `figures/trajectory_analysis/`
- **Completion check:** All figures render correctly with proper labels, legends, and axes

### Sub-Phase 6.10: Integration
- **Goal:** Integrate with evaluation pipeline
- **Tasks:** Add `--run-analysis` flag to `scripts/evaluate_models.py`, update `run_all_experiments.py`
- **Output:** Trajectory analysis runs as part of full pipeline
- **Completion check:** `python scripts/run_all_experiments.py` runs analysis automatically

---

## Strict Rules

- Use radians for all joint angles internally.
- Use consistent time step `dt` from config.
- Do not compare models on different test sets.
- Use the same conditions for all models.
- Save raw per-trajectory metrics before computing averages.
- Report mean, standard deviation, median, and 95th percentile.
- Do not claim improvement unless metrics support it.
- Mark failed trajectories explicitly (feasibility check failed).
- Support both deterministic models (single output) and stochastic models (K samples per condition).
- For stochastic models, compute metrics on each sample individually, then report best-of-K and mean.
- Smooth trajectories before differentiation to avoid numerical noise amplification.

---

## Deliverables

- [ ] `src/trajectory_analysis/` module (11 files)
- [ ] `configs/trajectory_analysis.yaml`
- [ ] `scripts/analyze_trajectories.py`
- [ ] `scripts/compare_trajectory_methods.py`
- [ ] `scripts/plot_trajectory_analysis.py`
- [ ] `scripts/generate_trajectory_report.py`
- [ ] `results/metrics/trajectory_analysis/` with CSV, JSON, MD outputs
- [ ] `figures/trajectory_analysis/` with 13+ figures
- [ ] `tests/unit/test_trajectory_analysis.py`
- [ ] `tests/integration/test_trajectory_analysis_pipeline.py`

---

## Go/No-Go Checks

| Check | Action if Failed |
|-------|-----------------|
| Analytical IK has non-zero feasibility violations | Verify joint limits in config match simulation |
| Smoothness metrics are NaN or Inf | Check finite-difference implementation; add epsilon guards |
| Energy metrics are negative | Verify power computation sign convention |
| Diversity is zero for generative models | Check that K>1 samples are generated; verify sampling is stochastic |
| Robustness tests show no degradation | Verify noise is actually being applied to inputs |
| Statistical tests fail due to small sample size | Increase test set or reduce percentile requirements |

---
