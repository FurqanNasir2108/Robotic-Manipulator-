# Testing Strategy

> Full details in `docs/instructions/phase_T_testing.instructions.md`

## Overview

This project uses a 3-tier testing strategy:

| Tier | Purpose | Location | Run Time |
|------|---------|----------|----------|
| Unit | Verify individual functions/classes | `tests/unit/` | ~30s |
| Integration | Verify end-to-end pipelines | `tests/integration/` | ~2min |
| Regression | Verify against known reference outputs | `tests/regression/` | ~30s |

## Quick Reference

```bash
# Run all tests
pytest tests/ -v

# Unit tests only (fast)
pytest tests/unit/ -v

# With coverage
pytest tests/ --cov=src --cov-report=html

# Smoke tests (CI)
pytest tests/test_smoke.py -v
```

## What to Test Per Phase

- **Phase 1:** FK/IK correctness, trajectory shapes, dataset I/O, normalization
- **Phase 2:** Model output shapes, gradient flow, single-batch overfit, baseline metrics
- **Phase 3:** VAE latent shapes, KL computation, reconstruction, sampling
- **Phase 4:** Diffusion noise schedule, denoiser shapes, sampling correctness
- **Phase 5:** All metrics, comparison pipeline, figure generation
- **Phase 6:** Full pipeline smoke test, import checks, config validation

See `phase_T_testing.instructions.md` for exhaustive test specifications.
