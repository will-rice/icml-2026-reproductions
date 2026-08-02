"""Computed audits for the Gaussian-mixture assumptions."""

from __future__ import annotations

from math import erfc, sqrt

import numpy as np
from numpy.typing import ArrayLike

from .model import IsotropicGMM

_CERTIFICATE_TOLERANCE = 1e-6
_LAPLACE_SCALE = 1.0


def _symmetric_tail_bound(model: IsotropicGMM, radius: float) -> float:
    """Bound the model mass outside ``[-radius, radius]``."""
    means = np.abs(model.means[:, 0])
    scales = np.sqrt(model.variances)
    distances = np.maximum(0.0, radius - means)
    component_bounds = np.array(
        [
            erfc(float(distance / (sqrt(2.0) * scale)))
            for distance, scale in zip(distances, scales, strict=True)
        ]
    )
    return float(model.weights @ component_bounds)


def _density_difference(
    model: IsotropicGMM, points: np.ndarray
) -> np.ndarray:
    base_density = np.exp(model.log_density(points[:, None]))
    laplace_density = 0.5 * np.exp(-np.abs(points) / _LAPLACE_SCALE)
    return laplace_density - base_density


def _first_derivative_interval_bounds(
    model: IsotropicGMM,
    *,
    left: np.ndarray,
    right: np.ndarray,
) -> np.ndarray:
    """Bound the density-difference derivative separately on every cell."""
    bound = np.zeros_like(left)
    for weight, mean, variance in zip(
        model.weights, model.means[:, 0], model.variances, strict=True
    ):
        scale = np.sqrt(variance)
        contains_mean = (left <= mean) & (mean <= right)
        minimum_distance = np.where(
            contains_mean,
            0.0,
            np.minimum(np.abs(left - mean), np.abs(right - mean)),
        )
        maximum_distance = np.maximum(
            np.abs(left - mean), np.abs(right - mean)
        )

        def derivative_at(distance: np.ndarray) -> np.ndarray:
            return (
                distance
                * np.exp(-0.5 * distance**2 / variance)
                / (np.sqrt(2.0 * np.pi) * scale**3)
            )

        component_bound = np.maximum(
            derivative_at(minimum_distance),
            derivative_at(maximum_distance),
        )
        spans_maximum = (minimum_distance <= scale) & (
            scale <= maximum_distance
        )
        component_bound[spans_maximum] = np.exp(-0.5) / (
            np.sqrt(2.0 * np.pi) * variance
        )
        bound += weight * component_bound

    contains_zero = (left <= 0.0) & (0.0 <= right)
    minimum_abs = np.where(
        contains_zero, 0.0, np.minimum(np.abs(left), np.abs(right))
    )
    bound += (
        0.5
        * np.exp(-minimum_abs / _LAPLACE_SCALE)
        / _LAPLACE_SCALE**2
    )
    return bound


def _second_derivative_bound(model: IsotropicGMM) -> float:
    """Return a global second-derivative bound away from the grid point zero."""
    variances = model.variances
    gaussian_second = 1.0 / (
        np.sqrt(2.0 * np.pi) * variances ** 1.5
    )
    laplace_second = 0.5 / _LAPLACE_SCALE**3
    return float(model.weights @ gaussian_second + laplace_second)


def _certified_grid_integral(
    model: IsotropicGMM, *, radius: float, grid_points: int
) -> tuple[float, float]:
    """Enclose the truncated absolute-density integral on a symmetric grid.

    A global first-derivative bound certifies cells that may contain a sign
    change. On cells whose sign is fixed by that bound, the usual trapezoid
    remainder uses a global second-derivative bound. Zero is always a grid
    boundary, so the Laplace derivative cusp is never inside a trapezoid cell.
    """
    grid = np.linspace(-radius, radius, grid_points)
    midpoint = 0.5 * (grid[:-1] + grid[1:])
    width = float(grid[1] - grid[0])
    endpoint_difference = _density_difference(model, grid)
    midpoint_difference = _density_difference(model, midpoint)
    first_bound = _first_derivative_interval_bounds(
        model, left=grid[:-1], right=grid[1:]
    )
    second_bound = _second_derivative_bound(model)

    fixed_sign = np.abs(midpoint_difference) > first_bound * width / 2.0
    estimates = np.empty(grid_points - 1, dtype=np.float64)
    errors = np.empty(grid_points - 1, dtype=np.float64)
    estimates[fixed_sign] = np.abs(
        0.5
        * width
        * (
            endpoint_difference[:-1][fixed_sign]
            + endpoint_difference[1:][fixed_sign]
        )
    )
    errors[fixed_sign] = second_bound * width**3 / 12.0
    estimates[~fixed_sign] = (
        width * np.abs(midpoint_difference[~fixed_sign])
    )
    errors[~fixed_sign] = first_bound[~fixed_sign] * width**2 / 4.0
    return float(np.sum(estimates)), float(np.sum(errors))


def contaminated_target_tv(
    base: IsotropicGMM, *, epsilon: float, grid_points: int
) -> dict[str, object]:
    """Compute TV from ``base`` to its centered-Laplace contamination.

    The contaminated density is ``(1-epsilon) * base + epsilon * Laplace(0, 1)``.
    The omitted-tail contribution is certified analytically.
    """
    epsilon = float(epsilon)
    grid_points = int(grid_points)
    if base.dimension != 1:
        raise ValueError("certified quadrature requires a one-dimensional mixture")
    if not np.isfinite(epsilon) or not 0.0 <= epsilon <= 1.0:
        raise ValueError("epsilon must be finite and between zero and one")
    if grid_points < 3 or grid_points % 2 == 0:
        raise ValueError("grid_points must be an odd integer of at least 3")

    max_mean = float(np.max(np.abs(base.means[:, 0])))
    max_scale = float(np.sqrt(np.max(base.variances)))
    radius = max(1.0, max_mean + 8.0 * max_scale)
    while True:
        gaussian_tail = _symmetric_tail_bound(base, radius)
        laplace_tail = float(np.exp(-radius / _LAPLACE_SCALE))
        combined_tail = 0.5 * (gaussian_tail + laplace_tail)
        if combined_tail < _CERTIFICATE_TOLERANCE / 4.0:
            break
        radius *= 1.5

    initial_grid_points = grid_points
    while True:
        absolute_integral, absolute_integral_error = _certified_grid_integral(
            base, radius=radius, grid_points=grid_points
        )
        unit_integration_error = 0.5 * absolute_integral_error
        if unit_integration_error + combined_tail < _CERTIFICATE_TOLERANCE:
            break
        grid_points = 2 * (grid_points - 1) + 1

    numerical_integral = float(0.5 * epsilon * absolute_integral)
    tail_error_bound = float(epsilon * combined_tail)
    integration_error_bound = float(epsilon * unit_integration_error)
    total_error_bound = tail_error_bound + integration_error_bound
    tv = float(np.clip(numerical_integral, 0.0, 1.0))
    covariance_scope = (
        "theorem-family"
        if np.allclose(base.variances, 1.0, rtol=0.0, atol=0.0)
        else "out-of-scope-control"
    )
    return {
        "metric": "total-variation",
        "scope": covariance_scope,
        "covariance_scope": covariance_scope,
        "epsilon": epsilon,
        "contaminant": "centered-laplace-scale-1",
        "quadrature": "symmetric-trapezoid",
        "initial_grid_points": initial_grid_points,
        "grid_points": grid_points,
        "grid_points_used": grid_points,
        "integration_radius": radius,
        "numerical_integral": numerical_integral,
        "tail_error_bound": tail_error_bound,
        "integration_error_bound": integration_error_bound,
        "total_error_bound": total_error_bound,
        "certificate_tolerance": _CERTIFICATE_TOLERANCE,
        "tv": tv,
    }


def check_mean_growth(
    *, means: ArrayLike, steps: int, c_r: float
) -> dict[str, object]:
    """Check ``max_k ||mu_k|| <= T ** c_r`` and label controls."""
    mean_array = np.asarray(means, dtype=np.float64)
    steps = int(steps)
    c_r = float(c_r)
    if mean_array.ndim != 2 or len(mean_array) == 0:
        raise ValueError("means must be a nonempty two-dimensional array")
    if not np.all(np.isfinite(mean_array)):
        raise ValueError("means must be finite")
    if steps < 2:
        raise ValueError("steps must be at least 2")
    if not np.isfinite(c_r) or c_r <= 0.0:
        raise ValueError("c_r must be finite and positive")

    max_mean_norm = float(np.max(np.linalg.norm(mean_array, axis=1)))
    threshold = float(np.exp(c_r * np.log(float(steps))))
    if not np.isfinite(threshold):
        raise ValueError("steps ** c_r must be finite")
    satisfied = max_mean_norm <= threshold
    return {
        "assumption": "mean-growth",
        "scope": "theorem-family" if satisfied else "out-of-scope-control",
        "steps": steps,
        "c_r": c_r,
        "max_mean_norm": max_mean_norm,
        "threshold": threshold,
        "satisfied": bool(satisfied),
    }
