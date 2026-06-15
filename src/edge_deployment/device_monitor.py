"""Device monitoring: CPU, GPU, RAM usage during inference."""

from __future__ import annotations

import time
import warnings


class DeviceMonitor:
    """Monitor system resources during inference.

    Tracks CPU, RAM, and optionally GPU usage.
    """

    def __init__(self):
        self._running = False
        self._samples = []
        self._start_time = None

        try:
            import psutil
            self._psutil = psutil
        except ImportError:
            self._psutil = None
            warnings.warn("psutil not available; resource monitoring limited.")

    def start(self):
        """Start monitoring."""
        self._running = True
        self._start_time = time.perf_counter()
        self._samples = []

    def stop(self):
        """Stop monitoring."""
        self._running = False

    def sample(self):
        """Take a single resource sample."""
        sample = {"timestamp": time.perf_counter()}

        if self._psutil is not None:
            process = self._psutil.Process()
            sample["cpu_percent"] = process.cpu_percent(interval=None)
            sample["ram_mb"] = process.memory_info().rss / (1024 * 1024)

        try:
            import torch
            if torch.cuda.is_available():
                sample["gpu_memory_mb"] = torch.cuda.memory_allocated() / (1024 * 1024)
                sample["gpu_memory_peak_mb"] = torch.cuda.max_memory_allocated() / (1024 * 1024)
        except Exception:
            pass

        self._samples.append(sample)

    def get_cpu_usage(self) -> float:
        """Get latest CPU usage percentage."""
        if self._psutil is not None:
            return float(self._psutil.Process().cpu_percent(interval=0.1))
        return -1.0

    def get_ram_usage(self) -> float:
        """Get current RAM usage in MB."""
        if self._psutil is not None:
            return float(self._psutil.Process().memory_info().rss / (1024 * 1024))
        return -1.0

    def get_gpu_usage(self) -> float:
        """Get GPU memory usage in MB (if available)."""
        try:
            import torch
            if torch.cuda.is_available():
                return float(torch.cuda.memory_allocated() / (1024 * 1024))
        except Exception:
            pass
        return -1.0

    def get_gpu_memory(self) -> float:
        """Get peak GPU memory in MB (if available)."""
        try:
            import torch
            if torch.cuda.is_available():
                return float(torch.cuda.max_memory_allocated() / (1024 * 1024))
        except Exception:
            pass
        return -1.0

    def get_power_estimate(self) -> float:
        """Get power estimate (placeholder — requires hardware-specific API)."""
        return -1.0

    def get_summary(self) -> dict:
        """Get summary of all collected samples.

        Returns
        -------
        dict
        """
        if not self._samples:
            return {"n_samples": 0}

        import numpy as np
        ram_values = [s.get("ram_mb", 0) for s in self._samples]
        cpu_values = [s.get("cpu_percent", 0) for s in self._samples]

        summary = {
            "n_samples": len(self._samples),
            "peak_ram_mb": float(max(ram_values)) if ram_values else -1.0,
            "mean_cpu_percent": float(np.mean(cpu_values)) if cpu_values else -1.0,
        }

        gpu_values = [s.get("gpu_memory_mb", -1) for s in self._samples if "gpu_memory_mb" in s]
        if gpu_values:
            summary["peak_gpu_mb"] = float(max(gpu_values))

        return summary
