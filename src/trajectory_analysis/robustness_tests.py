"""Robustness tests: evaluate models under noisy/perturbed inputs."""

from __future__ import annotations

from typing import Callable

import numpy as np

from src.trajectory_analysis.utils import _to_numpy


def evaluate_noisy_input(model_fn: Callable, condition: np.ndarray,
                         noise_levels: list[float], metric_fn: Callable,
                         reference: np.ndarray, n_repeats: int = 5) -> dict:
    """Evaluate model performance under Gaussian noise on the input condition.

    Parameters
    ----------
    model_fn : callable
        (condition) -> np.ndarray of shape (T, 3)
    condition : np.ndarray
        Original input condition.
    noise_levels : list of float
        Standard deviations of additive Gaussian noise.
    metric_fn : callable
        (pred, ref) -> float
    reference : np.ndarray
        Ground-truth trajectory.
    n_repeats : int
        Number of repetitions per noise level.

    Returns
    -------
    dict
        {noise_level: {"mean": float, "std": float}} for each level.
    """
    condition = _to_numpy(condition)
    reference = _to_numpy(reference)
    results = {}
    for sigma in noise_levels:
        scores = []
        for _ in range(n_repeats):
            noisy = condition + np.random.normal(0, sigma, condition.shape)
            pred = _to_numpy(model_fn(noisy))
            scores.append(metric_fn(pred, reference))
        results[sigma] = {"mean": float(np.mean(scores)), "std": float(np.std(scores))}
    return results


def evaluate_perturbed_start(model_fn: Callable, condition: np.ndarray,
                             perturbation: float, metric_fn: Callable,
                             reference: np.ndarray, n_repeats: int = 5) -> dict:
    """Evaluate model when the start pose is perturbed.

    Adds uniform noise of magnitude `perturbation` to the first waypoint.

    Returns
    -------
    dict
        {"mean": float, "std": float}
    """
    condition = _to_numpy(condition)
    reference = _to_numpy(reference)
    scores = []
    for _ in range(n_repeats):
        perturbed = condition.copy()
        perturbed[0] += np.random.uniform(-perturbation, perturbation, perturbed[0].shape)
        pred = _to_numpy(model_fn(perturbed))
        scores.append(metric_fn(pred, reference))
    return {"mean": float(np.mean(scores)), "std": float(np.std(scores))}


def evaluate_changed_shape_params(model_fn: Callable, conditions_variants: list[np.ndarray],
                                   metric_fn: Callable,
                                   references: list[np.ndarray]) -> dict:
    """Evaluate model on conditions with changed shape parameters.

    Parameters
    ----------
    conditions_variants : list of np.ndarray
        Modified conditions (e.g. different radii, side lengths).
    references : list of np.ndarray
        Corresponding ground-truth trajectories.

    Returns
    -------
    dict
        {"scores": list, "mean": float, "std": float}
    """
    scores = []
    for cond, ref in zip(conditions_variants, references):
        pred = _to_numpy(model_fn(_to_numpy(cond)))
        scores.append(metric_fn(pred, _to_numpy(ref)))
    return {
        "scores": [float(s) for s in scores],
        "mean": float(np.mean(scores)),
        "std": float(np.std(scores)),
    }


def robustness_summary(results: dict) -> dict:
    """Aggregate robustness test results.

    Parameters
    ----------
    results : dict
        Keys: "noisy_input", "perturbed_start", "changed_shape".

    Returns
    -------
    dict
        Summary with degradation factors.
    """
    summary = {}
    if "noisy_input" in results:
        noise_results = results["noisy_input"]
        levels = sorted(noise_results.keys())
        if len(levels) >= 2:
            baseline = noise_results[levels[0]]["mean"]
            worst = noise_results[levels[-1]]["mean"]
            summary["noise_degradation_factor"] = worst / max(baseline, 1e-12)
        summary["noisy_input"] = noise_results

    if "perturbed_start" in results:
        summary["perturbed_start"] = results["perturbed_start"]

    if "changed_shape" in results:
        summary["changed_shape"] = results["changed_shape"]

    return summary
