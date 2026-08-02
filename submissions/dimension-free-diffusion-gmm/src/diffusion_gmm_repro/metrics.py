"""Calibrated discrepancy metrics for high-dimensional mixture analysis."""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .model import IsotropicGMM

DEFAULT_SEED: int = 42


class MetricRecord(dict):
    """Dictionary subclass for metric evaluation records supporting direct estimate comparisons."""

    def __gt__(self, other: object) -> bool:
        if isinstance(other, dict):
            return self["estimate"] > float(other["estimate"])
        return self["estimate"] > float(other)

    def __lt__(self, other: object) -> bool:
        if isinstance(other, dict):
            return self["estimate"] < float(other["estimate"])
        return self["estimate"] < float(other)

    def __ge__(self, other: object) -> bool:
        if isinstance(other, dict):
            return self["estimate"] >= float(other["estimate"])
        return self["estimate"] >= float(other)

    def __le__(self, other: object) -> bool:
        if isinstance(other, dict):
            return self["estimate"] <= float(other["estimate"])
        return self["estimate"] <= float(other)


def _probabilities(arr: ArrayLike) -> NDArray[np.float64]:
    a = np.asarray(arr, dtype=np.float64)
    if not np.all(np.isfinite(a)):
        raise ValueError("probabilities must be finite")
    if np.any(a < 0):
        raise ValueError("probabilities must be non-negative")
    s = np.sum(a)
    if s <= 0:
        raise ValueError("sum of probabilities must be positive")
    return a / s


def partition_tv_lower_bound(
    observed_probabilities: ArrayLike,
    target_probabilities: ArrayLike,
) -> float:
    """Compute exact total variation lower bound between discrete partition probabilities."""
    observed = _probabilities(observed_probabilities)
    target = _probabilities(target_probabilities)
    if observed.shape != target.shape:
        raise ValueError("probability vectors must have the same shape")
    return float(0.5 * np.sum(np.abs(observed - target)))


def bootstrap_interval(
    data: ArrayLike,
    *,
    statistic: Callable[[NDArray[np.float64]], float] | None = None,
    confidence_level: float = 0.95,
    seed: int | None = None,
    replicates: int = 1000,
) -> tuple[float, float]:
    """Compute percentile bootstrap confidence interval for a scalar statistic."""
    if confidence_level != 0.95:
        raise ValueError(
            "only 0.95 confidence_level is supported for lower_95/upper_95 record contracts"
        )
    arr = np.asarray(data, dtype=np.float64)
    if not np.all(np.isfinite(arr)):
        raise ValueError("data must be finite")
    if arr.size == 0:
        raise ValueError("data must not be empty")

    effective_seed = DEFAULT_SEED if seed is None else int(seed)
    rng = np.random.default_rng(effective_seed)
    n = len(arr)
    indices = rng.integers(0, n, size=(replicates, n))
    boot_samples = arr[indices]

    if statistic is None:
        boot_stats = np.mean(boot_samples, axis=1)
    else:
        boot_stats = np.array([statistic(row) for row in boot_samples])

    alpha = (1.0 - confidence_level) / 2.0
    lower = float(np.percentile(boot_stats, 100.0 * alpha))
    upper = float(np.percentile(boot_stats, 100.0 * (1.0 - alpha)))
    return (lower, upper)


def linear_mmd(
    x: ArrayLike,
    y: ArrayLike,
    *,
    bandwidth: float = 1.0,
    seed: int | None = None,
    replicates: int = 1000,
) -> MetricRecord:
    """Compute linear-time MMD with paired samples, calibrated against self-split noise floor."""
    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)

    if not (np.all(np.isfinite(x_arr)) and np.all(np.isfinite(y_arr))):
        raise ValueError("inputs must be finite")
    if x_arr.ndim != 2 or y_arr.ndim != 2:
        raise ValueError("x and y must be 2D arrays")
    if x_arr.shape[1] != y_arr.shape[1]:
        raise ValueError("feature dimensions must match")
    if not (np.isfinite(bandwidth) and bandwidth > 0.0):
        raise ValueError("bandwidth must be finite and positive")

    n = min(len(x_arr), len(y_arr))
    if n < 2:
        raise ValueError("at least 2 samples are required")

    effective_seed = DEFAULT_SEED if seed is None else int(seed)
    m = n // 2
    x1 = x_arr[: 2 * m : 2]
    x2 = x_arr[1 : 2 * m : 2]
    y1 = y_arr[: 2 * m : 2]
    y2 = y_arr[1 : 2 * m : 2]

    gamma = 0.5 / (bandwidth**2)

    def rbf(a: NDArray[np.float64], b: NDArray[np.float64]) -> NDArray[np.float64]:
        diff = a - b
        dist_sq = np.sum(diff * diff, axis=-1)
        return np.exp(-gamma * dist_sq)

    h = rbf(x1, x2) + rbf(y1, y2) - rbf(x1, y2) - rbf(x2, y1)

    raw_estimate = float(np.mean(h))
    raw_lower_95, raw_upper_95 = bootstrap_interval(
        h, seed=effective_seed, replicates=replicates
    )

    m_half = m // 2
    if m_half >= 1:
        h_floor = (
            rbf(y_arr[: 2 * m_half : 2], y_arr[1 : 2 * m_half : 2])
            + rbf(
                y_arr[2 * m_half : 4 * m_half : 2],
                y_arr[2 * m_half + 1 : 4 * m_half : 2],
            )
            - rbf(
                y_arr[: 2 * m_half : 2], y_arr[2 * m_half + 1 : 4 * m_half : 2]
            )
            - rbf(
                y_arr[1 : 2 * m_half : 2], y_arr[2 * m_half : 4 * m_half : 2]
            )
        )
        calibration_floor = max(0.0, float(np.mean(h_floor)))
    else:
        calibration_floor = 0.0

    estimate = max(0.0, raw_estimate - calibration_floor)
    lower_95 = max(0.0, raw_lower_95 - calibration_floor)
    upper_95 = max(0.0, raw_upper_95 - calibration_floor)

    return MetricRecord(
        {
            "metric_kind": "linear-mmd",
            "estimate": estimate,
            "lower_95": lower_95,
            "upper_95": upper_95,
            "calibration_floor": calibration_floor,
            "samples": n,
            "seed": effective_seed,
        }
    )


def classifier_tv_lower_bound(
    x: ArrayLike,
    y: ArrayLike,
    *,
    seed: int | None = None,
    num_features: int = 256,
    ridge_alpha: float = 1.0,
    replicates: int = 1000,
) -> MetricRecord:
    """Compute classifier TV lower bound via Random Fourier Features and ridge regression."""
    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)

    if not (np.all(np.isfinite(x_arr)) and np.all(np.isfinite(y_arr))):
        raise ValueError("inputs must be finite")
    if x_arr.ndim != 2 or y_arr.ndim != 2:
        raise ValueError("x and y must be 2D arrays")
    if x_arr.shape[1] != y_arr.shape[1]:
        raise ValueError("feature dimensions must match")

    n = min(len(x_arr), len(y_arr))
    if n < 4:
        raise ValueError(
            "at least 4 samples are required for classifier fold split"
        )

    effective_seed = DEFAULT_SEED if seed is None else int(seed)
    d = x_arr.shape[1]
    rng = np.random.default_rng(effective_seed)

    W = rng.normal(size=(d, num_features))
    b = rng.uniform(0.0, 2.0 * np.pi, size=num_features)

    def phi(data: NDArray[np.float64]) -> NDArray[np.float64]:
        projection = data @ W + b[None, :]
        return np.sqrt(2.0 / num_features) * np.cos(projection)

    half = n // 2
    x_train, x_test = x_arr[:half], x_arr[half:n]
    y_train, y_test = y_arr[:half], y_arr[half:n]

    phi_x_train = phi(x_train)
    phi_y_train = phi(y_train)

    Z_train = np.vstack([phi_x_train, phi_y_train])
    labels_train = np.concatenate(
        [np.ones(len(x_train)), -np.ones(len(y_train))]
    )

    A = Z_train.T @ Z_train + ridge_alpha * np.eye(num_features)
    weights = np.linalg.solve(A, Z_train.T @ labels_train)

    phi_x_test = phi(x_test)
    phi_y_test = phi(y_test)

    pred_x = phi_x_test @ weights
    pred_y = phi_y_test @ weights

    correct_x = np.where(pred_x > 0, 1.0, np.where(pred_x == 0, 0.5, 0.0))
    correct_y = np.where(pred_y < 0, 1.0, np.where(pred_y == 0, 0.5, 0.0))
    correct_array = np.concatenate([correct_x, correct_y])

    accuracy = float(np.mean(correct_array))
    raw_estimate = max(0.0, 2.0 * accuracy - 1.0)

    raw_lower_95, raw_upper_95 = bootstrap_interval(
        correct_array,
        statistic=lambda arr: max(0.0, 2.0 * float(np.mean(arr)) - 1.0),
        seed=effective_seed,
        replicates=replicates,
    )

    y_half = len(y_arr) // 2
    if y_half >= 4:
        y_f1, y_f2 = y_arr[:y_half], y_arr[y_half : 2 * y_half]
        floor_n = min(len(y_f1), len(y_f2))
        f_half = floor_n // 2
        f_phi_x_train = phi(y_f1[:f_half])
        f_phi_y_train = phi(y_f2[:f_half])
        fZ_train = np.vstack([f_phi_x_train, f_phi_y_train])
        flabels_train = np.concatenate([np.ones(f_half), -np.ones(f_half)])
        fA = fZ_train.T @ fZ_train + ridge_alpha * np.eye(num_features)
        fweights = np.linalg.solve(fA, fZ_train.T @ flabels_train)

        fpred_x = phi(y_f1[f_half:floor_n]) @ fweights
        fpred_y = phi(y_f2[f_half:floor_n]) @ fweights
        fc_x = np.where(fpred_x > 0, 1.0, np.where(fpred_x == 0, 0.5, 0.0))
        fc_y = np.where(fpred_y < 0, 1.0, np.where(fpred_y == 0, 0.5, 0.0))
        facc = float(np.mean(np.concatenate([fc_x, fc_y])))
        calibration_floor = max(0.0, 2.0 * facc - 1.0)
    else:
        calibration_floor = 0.0

    estimate = max(0.0, raw_estimate - calibration_floor)
    lower_95 = max(0.0, raw_lower_95 - calibration_floor)
    upper_95 = max(0.0, raw_upper_95 - calibration_floor)

    return MetricRecord(
        {
            "metric_kind": "classifier-induced-tv-lower-bound",
            "estimate": estimate,
            "lower_95": lower_95,
            "upper_95": upper_95,
            "calibration_floor": calibration_floor,
            "samples": n,
            "seed": effective_seed,
        }
    )


def _histogram_gaussian_convolution(
    samples_1d: NDArray[np.float64],
    bin_edges: NDArray[np.float64],
    dx: float,
    bandwidth: float,
) -> NDArray[np.float64]:
    """Convolve a 1D sample histogram on fixed bin_edges with a discrete Gaussian kernel."""
    grid_points = len(bin_edges) - 1
    n = len(samples_1d)
    counts, _ = np.histogram(samples_1d, bins=bin_edges)
    empirical_density = counts / (n * dx)

    radius = int(np.ceil(4.0 * bandwidth / dx))
    j = np.arange(-radius, radius + 1)
    kernel = np.exp(-0.5 * (j * dx / bandwidth) ** 2) / (
        np.sqrt(2.0 * np.pi) * bandwidth
    )
    kernel_norm = kernel / (np.sum(kernel) * dx)
    full_conv = np.convolve(empirical_density, kernel_norm * dx, mode="full")
    start_idx = (len(kernel) - 1) // 2
    return full_conv[start_idx : start_idx + grid_points]


def plugin_tv_diagnostic(
    samples: ArrayLike,
    target: IsotropicGMM,
    *,
    grid_points: int = 1000,
    seed: int | None = None,
    replicates: int = 1000,
) -> MetricRecord:
    """Compute 1D plug-in TV diagnostic via fixed-grid histogram and discrete Gaussian convolution."""
    if grid_points <= 1:
        raise ValueError("grid_points must be at least 2")

    samples_arr = np.asarray(samples, dtype=np.float64)
    if not np.all(np.isfinite(samples_arr)):
        raise ValueError("samples must be finite")

    if samples_arr.ndim == 2:
        samples_1d = samples_arr[:, 0]
    elif samples_arr.ndim == 1:
        samples_1d = samples_arr
    else:
        raise ValueError("samples must be 1D or 2D array")

    n = len(samples_1d)
    if n == 0:
        raise ValueError("samples must not be empty")

    effective_seed = DEFAULT_SEED if seed is None else int(seed)

    target_1d = IsotropicGMM(
        target.weights,
        target.means[:, 0:1],
        target.variances,
    )

    std_max = float(np.sqrt(np.max(target.variances)))
    min_val = (
        min(float(np.min(samples_1d)), float(np.min(target.means[:, 0])))
        - 4.0 * std_max
    )
    max_val = (
        max(float(np.max(samples_1d)), float(np.max(target.means[:, 0])))
        + 4.0 * std_max
    )

    grid = np.linspace(min_val, max_val, grid_points)
    dx = (max_val - min_val) / (grid_points - 1)
    bin_edges = np.linspace(
        min_val - 0.5 * dx, max_val + 0.5 * dx, grid_points + 1
    )

    target_pdf = np.exp(target_1d.log_density(grid[:, None]))

    sample_std = float(np.std(samples_1d))
    if sample_std == 0.0:
        sample_std = 1.0
    h_base = 1.06 * sample_std * (n ** (-0.2))

    multipliers = [0.75, 1.0, 1.5]

    def calc_tv_for_samples(s: NDArray[np.float64]) -> float:
        tv_vals = []
        for mult in multipliers:
            h = mult * h_base
            conv_density = _histogram_gaussian_convolution(
                s, bin_edges, dx, h
            )
            tv = 0.5 * np.sum(np.abs(conv_density - target_pdf)) * dx
            tv_vals.append(tv)
        return float(np.mean(tv_vals))

    raw_estimate = calc_tv_for_samples(samples_1d)

    raw_lower_95, raw_upper_95 = bootstrap_interval(
        samples_1d,
        statistic=calc_tv_for_samples,
        seed=effective_seed,
        replicates=replicates,
    )

    rng = np.random.default_rng(effective_seed)
    comp_idx = rng.choice(len(target_1d.weights), size=n, p=target_1d.weights)
    target_draws = target_1d.means[comp_idx, 0] + rng.normal(
        scale=np.sqrt(target_1d.variances[comp_idx]), size=n
    )
    calibration_floor = calc_tv_for_samples(target_draws)

    estimate = max(0.0, raw_estimate - calibration_floor)
    lower_95 = max(0.0, raw_lower_95 - calibration_floor)
    upper_95 = max(0.0, raw_upper_95 - calibration_floor)

    return MetricRecord(
        {
            "metric_kind": "plugin-tv-diagnostic",
            "estimate": estimate,
            "lower_95": lower_95,
            "upper_95": upper_95,
            "calibration_floor": calibration_floor,
            "samples": n,
            "seed": effective_seed,
        }
    )


def log_slope_interval(
    steps: ArrayLike,
    metrics: ArrayLike,
    *,
    confidence_level: float = 0.95,
    seed: int | None = None,
    replicates: int = 1000,
) -> MetricRecord:
    """Compute log-log slope and bootstrap confidence interval for convergence rates."""
    if confidence_level != 0.95:
        raise ValueError(
            "only 0.95 confidence_level is supported for lower_95/upper_95 record contracts"
        )

    steps_arr = np.asarray(steps, dtype=np.float64)
    metrics_arr = np.asarray(metrics, dtype=np.float64)

    if not (
        np.all(np.isfinite(steps_arr)) and np.all(np.isfinite(metrics_arr))
    ):
        raise ValueError("steps and metrics must be finite")
    if steps_arr.ndim != 1 or metrics_arr.ndim != 1:
        raise ValueError("steps and metrics must be 1D arrays")
    if len(steps_arr) != len(metrics_arr):
        raise ValueError("steps and metrics must have the same length")
    if len(steps_arr) < 2:
        raise ValueError("at least 2 data points are required")
    if np.any(steps_arr <= 0) or np.any(metrics_arr <= 0):
        raise ValueError("steps and metrics must be strictly positive")

    effective_seed = DEFAULT_SEED if seed is None else int(seed)
    log_x = np.log(steps_arr)
    log_y = np.log(metrics_arr)

    def fit_slope(idx: NDArray[np.int64]) -> float:
        lx = log_x[idx]
        ly = log_y[idx]
        if np.all(lx == lx[0]):
            return 0.0
        slope, _ = np.polyfit(lx, ly, 1)
        return float(slope)

    estimate = fit_slope(np.arange(len(steps_arr)))

    rng = np.random.default_rng(effective_seed)
    n = len(steps_arr)
    boot_indices = rng.integers(0, n, size=(replicates, n))
    boot_slopes = np.array([fit_slope(idx) for idx in boot_indices])

    alpha = (1.0 - confidence_level) / 2.0
    lower_95 = float(np.percentile(boot_slopes, 100.0 * alpha))
    upper_95 = float(np.percentile(boot_slopes, 100.0 * (1.0 - alpha)))

    return MetricRecord(
        {
            "metric_kind": "log-slope-interval",
            "estimate": estimate,
            "lower_95": lower_95,
            "upper_95": upper_95,
            "calibration_floor": 0.0,
            "samples": n,
            "seed": effective_seed,
        }
    )
