# Phase T: Testing Strategy

> **This phase runs continuously alongside all other phases.** Read `00_main_orchestrator.instructions.md` first.

---

## Goal

Ensure every component is correct, reliable, and regression-free through systematic testing.

---

## Testing Framework

- **Framework:** `pytest` (with `pytest-cov` for coverage)
- **Run all tests:** `pytest tests/ -v --tb=short`
- **Run with coverage:** `pytest tests/ --cov=src --cov-report=html`
- **Minimum coverage target:** 80% for `src/simulation/`, `src/data/`, `src/evaluate/`

---

## Test Hierarchy

### 1. Unit Tests (`tests/unit/`)

#### `test_kinematics.py` — 3-Link Manipulator Math
```python
# FK consistency
- test_fk_zero_angles()              # all zeros → known position
- test_fk_known_configurations()     # manually computed reference poses
- test_fk_joint_limits()             # angles at limits produce valid positions

# IK consistency
- test_ik_roundtrip()                # FK(IK(pose)) ≈ pose for reachable poses
- test_ik_unreachable_pose()         # raises error or returns None for out-of-workspace
- test_ik_multiple_solutions()       # returns all valid solution branches
- test_ik_elbow_up_down()            # distinct solutions for same EE pose

# Jacobian
- test_jacobian_numerical_vs_analytical()  # finite-difference ≈ analytical
- test_jacobian_singularity()              # detect singular configurations
```

#### `test_trajectory_generator.py` — Shape Trajectory Generation
```python
- test_circle_trajectory_endpoints()      # starts and ends at correct position
- test_circle_trajectory_shape()          # all points at correct radius ± tolerance
- test_square_trajectory_corners()        # passes through 4 corners
- test_polygon_trajectory_symmetry()      # N-gon has N-fold symmetry
- test_line_trajectory_straight()         # all points on straight line
- test_bezier_trajectory_smoothness()     # no discontinuities in velocity
- test_trajectory_length_consistency()    # output length matches requested N
- test_trajectory_joint_limits()          # all joint angles within limits
- test_trajectory_reachability()          # all waypoints in workspace
```

#### `test_dataset.py` — Data Pipeline
```python
- test_dataset_load_save_roundtrip()      # save → load → identical data
- test_dataset_split_ratios()             # train/val/test match config ratios
- test_dataset_no_data_leakage()          # no sample appears in multiple splits
- test_dataset_shape_distribution()       # shapes are represented as expected
- test_dataset_metadata_consistency()     # metadata matches actual data dims
- test_dataset_schema_validation()        # all required fields present
```

#### `test_normalization.py` — Data Normalization
```python
- test_normalize_denormalize_roundtrip()  # denorm(norm(x)) ≈ x
- test_zscore_zero_mean_unit_var()        # normalized data has μ≈0, σ≈1
- test_minmax_range()                     # normalized data in [0, 1] or [-1, 1]
- test_normalization_stats_saved()        # stats file exists after fitting
- test_normalization_no_nan_inf()         # no NaN/Inf in output
```

#### `test_metrics.py` — Evaluation Metrics
```python
- test_rmse_zero_for_identical()          # RMSE(x, x) = 0
- test_rmse_symmetric()                   # RMSE(x, y) = RMSE(y, x)
- test_ee_error_zero_for_same_traj()      # zero error when pred = gt
- test_energy_proxy_nonnegative()         # energy ≥ 0
- test_smoothness_constant_trajectory()   # zero jerk for constant trajectory
- test_diversity_single_sample()          # diversity = 0 for one sample
- test_diversity_increases_with_spread()  # more spread → higher diversity
- test_constraint_violation_within_limits()  # 0% violation for valid trajectory
- test_constraint_violation_outside_limits() # >0% violation for invalid trajectory
```

#### `test_losses.py` — Loss Functions
```python
- test_mse_loss_zero_for_identical()      # MSE(x, x) = 0
- test_kl_divergence_standard_normal()    # KL(N(0,1) || N(0,1)) = 0
- test_kl_divergence_positive()           # KL ≥ 0 for any distribution
- test_energy_loss_gradient_exists()      # loss.backward() succeeds
- test_smoothness_loss_gradient_exists()  # loss.backward() succeeds
- test_diffusion_noise_loss_shape()       # output shape matches input
```

#### `test_models.py` — Model Architecture Validation
```python
# Shape tests (all models)
- test_cnn_output_shape()                 # correct (batch, N, 3)
- test_cnn_lstm_output_shape()
- test_cnn_gru_output_shape()
- test_cvae_output_shape()
- test_cvae_latent_shape()                # mu, log_var have correct dim
- test_diffusion_denoiser_output_shape()  # same shape as input

# Gradient tests
- test_cnn_backward_pass()                # no NaN gradients
- test_cvae_backward_pass()
- test_diffusion_backward_pass()

# Determinism
- test_model_deterministic_with_seed()    # same seed → same output

# Device compatibility
- test_model_runs_on_cpu()                # all models work on CPU
```

---

### 2. Integration Tests (`tests/integration/`)

#### `test_data_pipeline.py`
```python
- test_generate_and_load_dataset()        # generate → save → load → train split works
- test_dataloader_batch_shapes()          # DataLoader yields correct batch dimensions
- test_dataloader_normalization_applied()  # data is normalized when loaded
- test_augmentation_preserves_shape()     # augmented data has same structure
```

#### `test_training_loop.py`
```python
- test_baseline_overfits_single_batch()   # 1 batch, many epochs → loss near 0
- test_vae_loss_decreases()               # loss decreases over 10 epochs on small data
- test_diffusion_loss_decreases()         # loss decreases over 10 epochs on small data
- test_checkpoint_save_and_resume()       # save → load → resume training → loss continues decreasing
- test_tensorboard_logging()              # log files created during training
- test_early_stopping_triggers()          # training stops when val loss plateaus
```

#### `test_evaluation_pipeline.py`
```python
- test_evaluate_baseline_produces_metrics()    # all 9 metrics computed
- test_evaluate_vae_produces_metrics()
- test_evaluate_diffusion_produces_metrics()
- test_comparison_table_generation()           # generates markdown/CSV table
- test_figure_generation()                     # figures saved to correct directory
```

---

### 3. Regression Tests (`tests/regression/`)

#### `test_known_trajectories.py`
```python
- test_circle_trajectory_matches_reference()   # known circle → known joint trajectory
- test_ik_known_solution()                     # specific pose → known joint angles
- test_fk_known_solution()                     # specific joints → known EE pose
```

Store reference values in `tests/regression/fixtures/`:
```
fixtures/
├── circle_trajectory_ref.npz
├── ik_solutions_ref.json
└── fk_solutions_ref.json
```

---

### 4. Smoke Tests (Quick Validation)

Create `tests/test_smoke.py` for fast CI-style checks:
```python
- test_import_all_modules()               # all src modules importable
- test_config_files_valid_yaml()          # all configs parse without error
- test_small_dataset_generation()         # generate 10 samples → no crash
- test_small_training_run()              # 2 epochs on 10 samples → no crash
- test_evaluation_on_random_data()        # metrics compute on random tensors
```

---

## Testing by Phase

| Phase | Required Tests Before Proceeding |
|-------|--------------------------------|
| Phase 1 | All unit tests for kinematics, trajectory, dataset, normalization |
| Phase 2 | Model shape tests, single-batch overfit test, baseline evaluation test |
| Phase 3 | cVAE shape/gradient tests, loss decrease test, KL annealing correctness |
| Phase 4 | Diffusion shape/gradient tests, loss decrease test, sampling produces valid shapes |
| Phase 5 | All metric tests, comparison table test, figure generation test |
| Phase 6 | Smoke tests, all integration tests, full pipeline test |

---

## Test Configuration

`pyproject.toml` test section:
```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-v --tb=short --strict-markers"
markers = [
    "slow: marks tests as slow (deselect with '-m \"not slow\"')",
    "gpu: marks tests that require GPU",
    "integration: marks integration tests",
    "regression: marks regression tests",
]

[tool.coverage.run]
source = ["src"]
omit = ["src/visualization/*"]

[tool.coverage.report]
fail_under = 70
show_missing = true
```

---

## Continuous Validation Commands

```bash
# Quick check (unit tests only, ~30 seconds)
pytest tests/unit/ -v --tb=short

# Full check (all tests, ~5 minutes)
pytest tests/ -v --tb=short

# With coverage report
pytest tests/ --cov=src --cov-report=html --cov-report=term

# Only smoke tests (for quick CI)
pytest tests/test_smoke.py -v

# Skip slow tests
pytest tests/ -v -m "not slow"
```

---

## Deliverables

- [ ] `tests/conftest.py` with shared fixtures
- [ ] All unit test files listed above
- [ ] All integration test files listed above
- [ ] Regression test with reference fixtures
- [ ] Smoke test file
- [ ] `pyproject.toml` pytest configuration
- [ ] Coverage report showing ≥70% on `src/`
- [ ] All tests passing before final delivery

---

## Fallbacks

| Risk | Alternative |
|------|-------------|
| Some model tests too slow | Mark with `@pytest.mark.slow` and exclude from quick runs |
| GPU tests fail on CPU machine | Mark with `@pytest.mark.gpu` and use `@pytest.mark.skipif` |
| Coverage below target | Focus on critical paths (kinematics, metrics, data pipeline) over visualization |
| Reference fixtures need updating | Regenerate with `scripts/generate_test_fixtures.py` and commit |
