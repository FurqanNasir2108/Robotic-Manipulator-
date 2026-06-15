"""Diversity metrics for generative trajectory models."""

from __future__ import annotations

from typing import Callable

import numpy as np

from src.evaluate.metrics import diversity_score, path_tracking_error
from src.trajectory_analysis.utils import _to_numpy


def pairwise_trajectory_diversity(samples: np.ndarray, method: str = "dtw") -> float:
    """Mean pairwise diversity across K samples for the same condition.

    Delegates to evaluate.metrics.diversity_score.

    Parameters
    ----------
    samples : np.ndarray of shape (K, T, 3)
    method : str

    Returns
    -------
    float
    """
    return diversity_score(samples, method=method)


def best_of_k_accuracy(samples: np.ndarray, reference: np.ndarray,
                        k_values: list[int], metric_fn: Callable) -> dict:
    """Best-of-K accuracy: for each K, pick the sample closest to reference.

    Parameters
    ----------
    samples : np.ndarray of shape (K_max, T, 3)
    reference : np.ndarray of shape (T, 3)
    k_values : list of int
        K values to evaluate (e.g. [1, 5, 10]).
    metric_fn : callable
        (pred, ref) -> float, lower is better.

    Returns
    -------
    dict
        {k: best_metric_value} for each k in k_values.
    """
    samples = _to_numpy(samples)
    reference = _to_numpy(reference)
    all_scores = [metric_fn(samples[i], reference) for i in range(len(samples))]
    sorted_scores = sorted(all_scores)

    result = {}
    for k in k_values:
        k_clamped = min(k, len(sorted_scores))
        result[k] = float(sorted_scores[0]) if k_clamped > 0 else float("inf")
        # Best among first k samples (random order → take overall best of k)
        result[k] = float(min(sorted_scores[:k_clamped]))
    return result


def feasible_diversity(samples: np.ndarray, feasibility_fn: Callable,
                       method: str = "dtw") -> float:
    """Diversity computed only among feasible samples.

    Parameters
    ----------
    samples : np.ndarray of shape (K, T, 3)
    feasibility_fn : callable
        (trajectory) -> bool, True if feasible.
    method : str

    Returns
    -------
    float
        Diversity among feasible samples. 0.0 if <2 feasible.
    """
    samples = _to_numpy(samples)
    feasible = [s for s in samples if feasibility_fn(s)]
    if len(feasible) < 2:
        return 0.0
    return diversity_score(np.array(feasible), method=method)


def diversity_vs_accuracy_tradeoff(samples: np.ndarray, reference: np.ndarray,
                                    metric_fn: Callable, method: str = "dtw") -> dict:
    """Compute both diversity and best accuracy for a set of samples.

    Returns
    -------
    dict
        Keys: diversity, best_accuracy, mean_accuracy, n_samples.
    """
    samples = _to_numpy(samples)
    reference = _to_numpy(reference)
    scores = [metric_fn(s, reference) for s in samples]
    div = diversity_score(samples, method=method) if len(samples) >= 2 else 0.0
    return {
        "diversity": div,
        "best_accuracy": float(min(scores)) if scores else float("inf"),
        "mean_accuracy": float(np.mean(scores)) if scores else float("inf"),
        "n_samples": len(samples),
    }


def repeatability_score(model_fn: Callable, condition: np.ndarray,
                        n_runs: int, metric_fn: Callable) -> float:
    """Measure how consistent a stochastic model is across repeated runs.

    Parameters
    ----------
    model_fn : callable
        (condition) -> np.ndarray of shape (T, 3)
    condition : np.ndarray
    n_runs : int
    metric_fn : callable
        (pred_a, pred_b) -> float

    Returns
    -------
    float
        Mean pairwise distance across runs (lower = more repeatable).
    """
    trajectories = [_to_numpy(model_fn(condition)) for _ in range(n_runs)]
    if len(trajectories) < 2:
        return 0.0
    distances = []
    for i in range(len(trajectories)):
        for j in range(i + 1, len(trajectories)):
            distances.append(metric_fn(trajectories[i], trajectories[j]))
    return float(np.mean(distances))
