"""Edge benchmarking: latency, throughput, memory, model size, Pareto analysis."""

from __future__ import annotations

import os
import time

import numpy as np


def benchmark_latency(session, test_inputs: list[np.ndarray],
                      n_runs: int = 1000, warmup: int = 100) -> dict:
    """Benchmark inference latency.

    Parameters
    ----------
    session : ONNXInferenceSession or similar
    test_inputs : list of np.ndarray
    n_runs : int
    warmup : int

    Returns
    -------
    dict
        Keys: mean_ms, median_ms, p50_ms, p95_ms, p99_ms, std_ms.
    """
    # Warmup
    for i in range(min(warmup, len(test_inputs))):
        session.predict(test_inputs[i % len(test_inputs)])

    latencies = []
    for i in range(n_runs):
        inp = test_inputs[i % len(test_inputs)]
        start = time.perf_counter()
        session.predict(inp)
        end = time.perf_counter()
        latencies.append((end - start) * 1000.0)

    latencies = np.array(latencies)
    return {
        "mean_ms": float(np.mean(latencies)),
        "median_ms": float(np.median(latencies)),
        "p50_ms": float(np.percentile(latencies, 50)),
        "p95_ms": float(np.percentile(latencies, 95)),
        "p99_ms": float(np.percentile(latencies, 99)),
        "std_ms": float(np.std(latencies)),
        "min_ms": float(np.min(latencies)),
        "max_ms": float(np.max(latencies)),
    }


def benchmark_throughput(session, test_inputs: list[np.ndarray],
                         duration_sec: float = 10.0) -> float:
    """Measure throughput in trajectories per second.

    Parameters
    ----------
    session : inference session
    test_inputs : list of np.ndarray
    duration_sec : float

    Returns
    -------
    float
        Trajectories per second.
    """
    count = 0
    start = time.perf_counter()
    while time.perf_counter() - start < duration_sec:
        inp = test_inputs[count % len(test_inputs)]
        session.predict(inp)
        count += 1
    elapsed = time.perf_counter() - start
    return float(count / elapsed)


def benchmark_memory(session, test_input: np.ndarray) -> dict:
    """Measure peak memory during inference.

    Parameters
    ----------
    session : inference session
    test_input : np.ndarray

    Returns
    -------
    dict
        Keys: peak_ram_mb, peak_gpu_mb (if available).
    """
    result = {}
    try:
        import psutil
        process = psutil.Process()
        mem_before = process.memory_info().rss / (1024 * 1024)
        session.predict(test_input)
        mem_after = process.memory_info().rss / (1024 * 1024)
        result["peak_ram_mb"] = float(max(mem_before, mem_after))
        result["ram_delta_mb"] = float(mem_after - mem_before)
    except ImportError:
        result["peak_ram_mb"] = -1.0
        result["note"] = "psutil not available"

    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            session.predict(test_input)
            result["peak_gpu_mb"] = float(torch.cuda.max_memory_allocated() / (1024 * 1024))
    except Exception:
        pass

    return result


def model_size_mb(model_path: str) -> float:
    """Get model file size in megabytes.

    Parameters
    ----------
    model_path : str

    Returns
    -------
    float
    """
    return float(os.path.getsize(model_path) / (1024 * 1024))


def pareto_analysis(benchmark_results: dict, accuracy_results: dict) -> dict:
    """Find Pareto-optimal models on accuracy-latency-size front.

    Parameters
    ----------
    benchmark_results : dict
        {model_name: {"mean_ms": float, "size_mb": float}}
    accuracy_results : dict
        {model_name: {"joint_rmse": float}}

    Returns
    -------
    dict
        Keys: pareto_models (list), all_models (list of dicts).
    """
    models = []
    for name in benchmark_results:
        if name not in accuracy_results:
            continue
        models.append({
            "name": name,
            "latency_ms": benchmark_results[name].get("mean_ms", float("inf")),
            "size_mb": benchmark_results[name].get("size_mb", float("inf")),
            "accuracy": accuracy_results[name].get("joint_rmse", float("inf")),
        })

    # Find Pareto front (minimize all three objectives)
    pareto = []
    for m in models:
        dominated = False
        for other in models:
            if (other["latency_ms"] <= m["latency_ms"] and
                other["size_mb"] <= m["size_mb"] and
                other["accuracy"] <= m["accuracy"] and
                (other["latency_ms"] < m["latency_ms"] or
                 other["size_mb"] < m["size_mb"] or
                 other["accuracy"] < m["accuracy"])):
                dominated = True
                break
        if not dominated:
            pareto.append(m["name"])

    return {"pareto_models": pareto, "all_models": models}
