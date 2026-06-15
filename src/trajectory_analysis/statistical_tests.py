"""Statistical tests and summary statistics for trajectory analysis."""

from __future__ import annotations

import numpy as np
from scipy import stats


def compute_statistics(values: list | np.ndarray) -> dict:
    """Compute summary statistics for a set of metric values.

    Parameters
    ----------
    values : array-like

    Returns
    -------
    dict
        Keys: mean, std, median, min, max, p50, p95, p99, ci_lower, ci_upper.
    """
    values = np.asarray(values, dtype=np.float64)
    if len(values) == 0:
        return {k: 0.0 for k in ["mean", "std", "median", "min", "max",
                                   "p50", "p95", "p99", "ci_lower", "ci_upper"]}
    ci_lower, ci_upper = confidence_interval(values, level=0.95)
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
        "median": float(np.median(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "p50": float(np.percentile(values, 50)),
        "p95": float(np.percentile(values, 95)),
        "p99": float(np.percentile(values, 99)),
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
    }


def confidence_interval(values: np.ndarray, level: float = 0.95) -> tuple[float, float]:
    """Compute confidence interval for the mean.

    Parameters
    ----------
    values : np.ndarray
    level : float

    Returns
    -------
    tuple
        (lower, upper) bounds.
    """
    values = np.asarray(values, dtype=np.float64)
    n = len(values)
    if n < 2:
        m = float(np.mean(values)) if n == 1 else 0.0
        return (m, m)
    mean = float(np.mean(values))
    se = float(stats.sem(values))
    alpha = 1.0 - level
    t_crit = float(stats.t.ppf(1.0 - alpha / 2.0, df=n - 1))
    margin = t_crit * se
    return (mean - margin, mean + margin)


def wilcoxon_signed_rank(values_a: np.ndarray, values_b: np.ndarray) -> dict:
    """Wilcoxon signed-rank test for paired samples.

    Parameters
    ----------
    values_a, values_b : np.ndarray
        Paired metric values for two models.

    Returns
    -------
    dict
        Keys: statistic, p_value, significant_at_005.
    """
    values_a = np.asarray(values_a, dtype=np.float64)
    values_b = np.asarray(values_b, dtype=np.float64)
    diff = values_a - values_b
    # Remove zero differences
    nonzero = diff[diff != 0]
    if len(nonzero) < 10:
        return {"statistic": float("nan"), "p_value": 1.0, "significant_at_005": False}
    stat, p_value = stats.wilcoxon(nonzero)
    return {
        "statistic": float(stat),
        "p_value": float(p_value),
        "significant_at_005": p_value < 0.05,
    }


def paired_comparison(model_a_metrics: dict, model_b_metrics: dict,
                       metric_names: list[str] | None = None) -> dict:
    """Compare two models across multiple metrics using Wilcoxon tests.

    Parameters
    ----------
    model_a_metrics : dict
        {metric_name: list_of_values}
    model_b_metrics : dict
        {metric_name: list_of_values}
    metric_names : list of str or None
        Metrics to compare. If None, compare all shared keys.

    Returns
    -------
    dict
        {metric_name: wilcoxon_result}
    """
    if metric_names is None:
        metric_names = sorted(set(model_a_metrics.keys()) & set(model_b_metrics.keys()))

    results = {}
    for name in metric_names:
        a = np.asarray(model_a_metrics[name])
        b = np.asarray(model_b_metrics[name])
        if len(a) != len(b):
            results[name] = {"error": "mismatched lengths"}
            continue
        results[name] = wilcoxon_signed_rank(a, b)
    return results
