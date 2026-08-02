"""Deterministic CPU diagnostics for the selected diffusion/GMM claims."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from diffusion_gmm_repro.model import IsotropicGMM


def _positive_dimensions(dimensions: Iterable[int]) -> tuple[int, ...]:
    values = tuple(int(value) for value in dimensions)
    if not values or any(value <= 0 for value in values):
        raise ValueError("dimensions must contain positive integers")
    return values


def run_dimension_audit(
    *,
    dimensions: Iterable[int] = (1, 4, 16, 64),
    step_size: float = 0.125,
    seed: int = 2026,
    samples: int = 2048,
) -> list[dict[str, object]]:
    """Audit one Euler/DDPM step at the stationary standard Gaussian.

    With exact score ``-x``, ``x'=(1-h)x+sqrt(2h)z`` has coordinate
    variance ``1+h**2``. The per-coordinate analytic bias is therefore
    independent of ambient dimension; sampled values are diagnostics only.
    """

    dims = _positive_dimensions(dimensions)
    if not 0.0 < step_size < 1.0:
        raise ValueError("step_size must be between zero and one")
    if samples <= 0:
        raise ValueError("samples must be positive")

    rng = np.random.default_rng(seed)
    records: list[dict[str, object]] = []
    for dimension in dims:
        initial = rng.normal(size=(samples, dimension))
        noise = rng.normal(size=(samples, dimension))
        updated = (1.0 - step_size) * initial + np.sqrt(2.0 * step_size) * noise
        records.append(
            {
                "dimension": dimension,
                "step_size": float(step_size),
                "samples": samples,
                "seed": seed,
                "analytic_per_coordinate_variance_error": float(step_size**2),
                "empirical_per_coordinate_variance_error": float(
                    abs(np.mean(updated**2) - 1.0)
                ),
                "sample_checksum": float(np.round(np.sum(updated), 12)),
                "assumption_satisfied": True,
                "control": None,
            }
        )
    return records


def run_score_error_audit(
    *,
    error_levels: Iterable[float] = (0.0, 0.05, 0.1, 0.25),
    dimension: int = 8,
    step_size: float = 0.125,
    seed: int = 2026,
    samples: int = 2048,
) -> list[dict[str, object]]:
    """Audit the response to a controlled additive score-vector error."""

    levels = tuple(float(level) for level in error_levels)
    if not levels or any(level < 0.0 or not np.isfinite(level) for level in levels):
        raise ValueError("error_levels must be finite and nonnegative")
    if dimension <= 0 or samples <= 0:
        raise ValueError("dimension and samples must be positive")
    if not 0.0 < step_size < 1.0:
        raise ValueError("step_size must be between zero and one")

    rng = np.random.default_rng(seed)
    initial = rng.normal(size=(samples, dimension))
    noise = rng.normal(size=(samples, dimension))
    direction = rng.normal(size=dimension)
    direction /= np.linalg.norm(direction)
    records: list[dict[str, object]] = []
    for level in levels:
        updated = (
            (1.0 - step_size) * initial
            + step_size * level * direction
            + np.sqrt(2.0 * step_size) * noise
        )
        records.append(
            {
                "dimension": dimension,
                "step_size": float(step_size),
                "score_error_l2": level,
                "analytic_mean_shift_l2": float(step_size * level),
                "empirical_mean_l2": float(np.linalg.norm(np.mean(updated, axis=0))),
                "samples": samples,
                "seed": seed,
                "assumption_satisfied": True,
                "control": None,
            }
        )
    misspecified = (
        (1.0 - step_size) * initial
        + step_size * 0.5 * initial**3
        + np.sqrt(2.0 * step_size) * noise
    )
    records.append(
        {
            "dimension": dimension,
            "step_size": float(step_size),
            "score_error_l2": None,
            "analytic_mean_shift_l2": None,
            "empirical_mean_l2": float(np.linalg.norm(np.mean(misspecified, axis=0))),
            "empirical_second_moment": float(np.mean(misspecified**2)),
            "samples": samples,
            "seed": seed,
            "assumption_satisfied": False,
            "control": "state-dependent-misspecified-score",
        }
    )
    return records


def run_jacobian_audit(
    *,
    dimensions: Iterable[int] = (1, 4, 16, 64),
    seed: int = 2026,
    samples: int = 512,
    variance_floor: float = 0.5,
) -> list[dict[str, object]]:
    """Audit normalized score-Jacobian traces and a variance-floor control."""

    dims = _positive_dimensions(dimensions)
    if samples <= 0 or variance_floor <= 0.0:
        raise ValueError("samples and variance_floor must be positive")
    records: list[dict[str, object]] = []
    for variance, assumption_satisfied, control in (
        (1.0, True, None),
        (variance_floor / 10.0, False, "unit-covariance-violation"),
    ):
        for dimension in dims:
            means = np.zeros((2, dimension))
            means[:, 0] = (-2.0, 2.0)
            model = IsotropicGMM(
                weights=np.array([0.5, 0.5]),
                means=means,
                variances=np.array([variance, variance]),
            )
            points = np.zeros((samples, dimension))
            points[:, 0] = np.linspace(-4.0, 4.0, samples)
            trace_i_plus_j = dimension + model.score_jacobian_trace(points)
            epsilon = 1e-5
            finite_difference_trace = np.zeros(samples)
            for axis in range(dimension):
                offset = np.zeros(dimension)
                offset[axis] = epsilon
                finite_difference_trace += (
                    model.score(points + offset)[:, axis]
                    - model.score(points - offset)[:, axis]
                ) / (2.0 * epsilon)
            finite_difference_error = np.max(
                np.abs(finite_difference_trace - model.score_jacobian_trace(points))
            )
            records.append(
                {
                    "dimension": dimension,
                    "components": 2,
                    "variance": float(variance),
                    "variance_floor": float(variance_floor),
                    "max_trace_i_plus_j": float(np.max(trace_i_plus_j)),
                    "mean_trace_i_plus_j": float(np.mean(trace_i_plus_j)),
                    "finite_difference_error": float(finite_difference_error),
                    "samples": samples,
                    "seed": seed,
                    "assumption_satisfied": assumption_satisfied,
                    "control": control,
                }
            )
    return records
