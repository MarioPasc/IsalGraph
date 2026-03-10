# ruff: noqa: N803, N806
"""Timing utilities for computational benchmarking.

Follows Weber et al. (2019), Genome Biology 20:125 and
Bartz-Beielstein et al. (2020) guidelines:
- CPU time (not wall clock) via time.process_time()
- 25 repetitions per measurement
- Report median and IQR
- Single-threaded execution
"""

from __future__ import annotations

import os
import platform
import time
from collections.abc import Callable
from typing import Any

import numpy as np
from scipy import stats


def time_function(
    func: Callable[..., Any],
    args: tuple = (),
    kwargs: dict[str, Any] | None = None,
    n_reps: int = 25,
    warmup: int = 1,
) -> dict[str, Any]:
    """Time a function with multiple repetitions using CPU time.

    Args:
        func: Function to time.
        args: Positional arguments for func.
        kwargs: Keyword arguments for func.
        n_reps: Number of timed repetitions.
        warmup: Number of warmup calls before timing.

    Returns:
        Dict with median_s, iqr_s, mean_s, std_s, min_s, max_s,
        all_times_s, n_reps, and result (from last call).
    """
    kwargs = kwargs or {}

    # Warmup runs (excluded from timing)
    result: Any = None
    for _ in range(warmup):
        result = func(*args, **kwargs)

    # Timed runs
    times: list[float] = []
    for _ in range(n_reps):
        t0 = time.process_time()
        result = func(*args, **kwargs)
        t1 = time.process_time()
        times.append(t1 - t0)

    times_arr = np.array(times)
    return {
        "median_s": float(np.median(times_arr)),
        "iqr_s": float(np.percentile(times_arr, 75) - np.percentile(times_arr, 25)),
        "mean_s": float(np.mean(times_arr)),
        "std_s": float(np.std(times_arr)),
        "min_s": float(np.min(times_arr)),
        "max_s": float(np.max(times_arr)),
        "all_times_s": times,
        "n_reps": n_reps,
        "result": result,
    }


def time_function_batch(
    func: Callable[..., Any],
    args: tuple = (),
    kwargs: dict[str, Any] | None = None,
    n_reps: int = 25,
    batch_size: int = 1000,
    warmup: int = 1,
) -> dict[str, Any]:
    """Time a very fast function by batching calls.

    For functions that complete in < 1 microsecond, process_time()
    may return 0. This variant times `batch_size` consecutive calls
    and divides by batch_size.

    Args:
        func: Function to time.
        args: Positional arguments for func.
        kwargs: Keyword arguments for func.
        n_reps: Number of timed batches.
        batch_size: Calls per batch.
        warmup: Number of warmup calls before timing.

    Returns:
        Same structure as time_function, with per-call times.
    """
    kwargs = kwargs or {}

    result: Any = None
    for _ in range(warmup):
        result = func(*args, **kwargs)

    times: list[float] = []
    for _ in range(n_reps):
        t0 = time.process_time()
        for _ in range(batch_size):
            result = func(*args, **kwargs)
        t1 = time.process_time()
        times.append((t1 - t0) / batch_size)

    times_arr = np.array(times)
    return {
        "median_s": float(np.median(times_arr)),
        "iqr_s": float(np.percentile(times_arr, 75) - np.percentile(times_arr, 25)),
        "mean_s": float(np.mean(times_arr)),
        "std_s": float(np.std(times_arr)),
        "min_s": float(np.min(times_arr)),
        "max_s": float(np.max(times_arr)),
        "all_times_s": times,
        "n_reps": n_reps,
        "batch_size": batch_size,
        "result": result,
    }


def get_hardware_info() -> dict[str, Any]:
    """Collect hardware and software specification for reproducibility.

    Returns:
        Dict with platform, processor, python_version, cpu_count,
        and optionally ram_gb and cpu_freq_mhz (if psutil installed).
    """
    info: dict[str, Any] = {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "machine": platform.machine(),
        "python_version": platform.python_version(),
        "cpu_count": os.cpu_count(),
    }
    try:
        import psutil

        info["ram_gb"] = round(psutil.virtual_memory().total / (1024**3), 1)
        freq = psutil.cpu_freq()
        info["cpu_freq_mhz"] = freq.max if freq else None
    except ImportError:
        info["ram_gb"] = "unknown (psutil not installed)"
        info["cpu_freq_mhz"] = None
    return info


def fit_scaling_exponent(
    sizes: np.ndarray,
    times: np.ndarray,
) -> dict[str, float]:
    """Fit scaling exponent via OLS on log-log data.

    Model: log(T) = alpha * log(n) + log(c), i.e. T(n) ~ c * n^alpha.

    Args:
        sizes: Array of problem sizes (e.g. node counts).
        times: Array of corresponding median times.

    Returns:
        Dict with alpha, c, r_squared, p_value, std_err.
        Returns NaN values if regression fails (e.g. too few points).
    """
    # Filter out non-positive entries (can't take log)
    mask = (sizes > 0) & (times > 0) & np.isfinite(times)
    if mask.sum() < 2:
        return {
            "alpha": float("nan"),
            "c": float("nan"),
            "r_squared": float("nan"),
            "p_value": float("nan"),
            "std_err": float("nan"),
            "n_points": int(mask.sum()),
        }

    log_n = np.log(sizes[mask].astype(np.float64))
    log_t = np.log(times[mask].astype(np.float64))

    slope, intercept, r_value, p_value, std_err = stats.linregress(log_n, log_t)

    return {
        "alpha": float(slope),
        "c": float(np.exp(intercept)),
        "r_squared": float(r_value**2),
        "p_value": float(p_value),
        "std_err": float(std_err),
        "n_points": int(mask.sum()),
    }
