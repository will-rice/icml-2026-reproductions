"""Randomized posterior-covariance audits for score-Jacobian traces."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
from numpy.typing import NDArray

from .model import IsotropicGMM
from .schedule import paper_schedule

_THEOREM_GEOMETRIES = {"balanced", "imbalanced", "overlapping", "separated"}
_RESPONSIBILITY_MEMORY_CAP_BYTES = 64 * 1024 * 1024
_TEMPORARY_ARRAYS_PER_RESPONSIBILITY_ELEMENT = 6


def random_mixture_case(
    *,
    components: int,
    ambient_dimension: int,
    geometry: str,
    seed: int,
) -> dict[str, object]:
    """Construct a reproducible full-available-rank mixture geometry."""
    components = int(components)
    ambient_dimension = int(ambient_dimension)
    seed = int(seed)
    if components < 2:
        raise ValueError("components must be at least 2")
    if ambient_dimension < 1:
        raise ValueError("ambient_dimension must be positive")
    if geometry not in {*_THEOREM_GEOMETRIES, "anisotropic-control"}:
        raise ValueError("unknown mixture geometry")

    rng = np.random.default_rng(seed)
    means = rng.normal(size=(components, ambient_dimension))
    means -= means.mean(axis=0, keepdims=True)
    target_rank = min(components - 1, ambient_dimension)
    left, _, right_t = np.linalg.svd(means, full_matrices=False)
    means = left[:, :target_rank] @ right_t[:target_rank]

    scale = {
        "balanced": 2.0,
        "imbalanced": 2.0,
        "overlapping": 0.35,
        "separated": 4.0,
        "anisotropic-control": 2.0,
    }[geometry]
    means *= scale
    if geometry == "imbalanced":
        weights = np.exp(np.linspace(0.0, -2.0, components))
        weights /= weights.sum()
    else:
        weights = np.full(components, 1.0 / components)

    is_control = geometry == "anisotropic-control"
    coordinate_variances = (
        np.linspace(0.5, 1.5, ambient_dimension)
        if is_control
        else np.ones(ambient_dimension)
    )
    return {
        "components": components,
        "ambient_dimension": ambient_dimension,
        "geometry": geometry,
        "seed": seed,
        "active_rank": int(np.linalg.matrix_rank(means)),
        "scope": "out-of-scope-control" if is_control else "theorem-family",
        "component_covariance": "common-anisotropic" if is_control else "identity",
        "weights": weights.tolist(),
        "means": means.tolist(),
        "coordinate_variances": coordinate_variances.tolist(),
    }


def _responsibilities(
    points: NDArray[np.float64],
    *,
    weights: NDArray[np.float64],
    means: NDArray[np.float64],
    coordinate_variances: NDArray[np.float64],
) -> NDArray[np.float64]:
    differences = points[:, None, :] - means[None, :, :]
    component_logs = (
        np.log(weights)[None, :]
        - 0.5 * np.sum(np.log(2.0 * np.pi * coordinate_variances))
        - 0.5
        * np.sum(
            differences**2 / coordinate_variances[None, None, :], axis=2
        )
    )
    maxima = np.max(component_logs, axis=1, keepdims=True)
    unnormalized = np.exp(component_logs - maxima)
    return unnormalized / np.sum(unnormalized, axis=1, keepdims=True)


def _finite_difference_trace(
    model: IsotropicGMM, points: NDArray[np.float64], *, step: float = 1e-5
) -> NDArray[np.float64]:
    diagonal_sum = np.zeros(len(points), dtype=np.float64)
    for coordinate in range(model.dimension):
        plus = points.copy()
        minus = points.copy()
        plus[:, coordinate] += step
        minus[:, coordinate] -= step
        diagonal_sum += (
            model.score(plus)[:, coordinate] - model.score(minus)[:, coordinate]
        ) / (2.0 * step)
    return model.dimension + diagonal_sum


def run_jacobian_cell(
    *,
    components: int,
    ambient_dimension: int,
    geometry: str,
    steps: int,
    time: int,
    samples: int,
    seed: int,
    finite_difference_check: bool = False,
    batch_size: int | None = None,
) -> dict[str, object]:
    """Evaluate ``tr(I + Jacobian(score))`` on a randomized mixture cell."""
    steps = int(steps)
    time = int(time)
    samples = int(samples)
    seed = int(seed)
    if samples < 1:
        raise ValueError("samples must be positive")
    if batch_size is not None and int(batch_size) < 1:
        raise ValueError("batch_size must be positive")
    if time < 0 or time > steps:
        raise ValueError("time must be between zero and steps")

    case = random_mixture_case(
        components=components,
        ambient_dimension=ambient_dimension,
        geometry=geometry,
        seed=seed,
    )
    weights = np.asarray(case["weights"], dtype=np.float64)
    means = np.asarray(case["means"], dtype=np.float64)
    coordinate_variances = np.asarray(
        case["coordinate_variances"], dtype=np.float64
    )
    schedule = paper_schedule(steps=steps)
    alpha_bar_t = float(schedule.alpha_bar[time])
    forward_means = np.sqrt(alpha_bar_t) * means
    forward_variances = (
        alpha_bar_t * coordinate_variances + (1.0 - alpha_bar_t)
    )

    rng = np.random.default_rng(np.random.SeedSequence([seed, time, samples]))
    assignments = rng.choice(len(weights), size=samples, p=weights)
    points = forward_means[assignments] + rng.normal(
        size=(samples, int(case["ambient_dimension"]))
    ) * np.sqrt(forward_variances)
    responsibility_bytes_per_sample = max(
        1,
        int(components)
        * int(ambient_dimension)
        * np.dtype(np.float64).itemsize
        * _TEMPORARY_ARRAYS_PER_RESPONSIBILITY_ELEMENT,
    )
    memory_limited_batch_size = max(
        1,
        _RESPONSIBILITY_MEMORY_CAP_BYTES
        // responsibility_bytes_per_sample,
    )
    requested_batch_size = samples if batch_size is None else int(batch_size)
    responsibility_chunk_size = min(
        samples, requested_batch_size, memory_limited_batch_size
    )

    theorem_identity_applied = case["scope"] == "theorem-family"
    trace_values = np.empty(samples, dtype=np.float64)
    finite_difference_max_error: float | None = None
    forward_model = (
        IsotropicGMM(weights, forward_means, np.ones(len(weights)))
        if finite_difference_check and theorem_identity_applied
        else None
    )
    for start in range(0, samples, responsibility_chunk_size):
        stop = min(start + responsibility_chunk_size, samples)
        point_batch = points[start:stop]
        responsibilities = _responsibilities(
            point_batch,
            weights=weights,
            means=forward_means,
            coordinate_variances=forward_variances,
        )
        if theorem_identity_applied:
            posterior_mean = responsibilities @ means
            second_moment = np.einsum(
                "nk,kd,kd->n", responsibilities, means, means
            )
            trace_posterior_covariance = second_moment - np.sum(
                posterior_mean**2, axis=1
            )
            batch_trace = alpha_bar_t * trace_posterior_covariance
            batch_trace = np.maximum(batch_trace, 0.0)
        else:
            posterior_mean = responsibilities @ forward_means
            second_moment_coordinates = responsibilities @ (
                forward_means**2
            )
            posterior_variance = (
                second_moment_coordinates - posterior_mean**2
            )
            batch_trace = np.sum(
                1.0
                - 1.0 / forward_variances
                + posterior_variance / forward_variances**2,
                axis=1,
            )
        trace_values[start:stop] = batch_trace

        if finite_difference_check:
            if not theorem_identity_applied:
                raise ValueError(
                    "finite-difference certification is only for "
                    "theorem-family cells"
                )
            assert forward_model is not None
            finite_difference = _finite_difference_trace(
                forward_model, point_batch
            )
            batch_error = float(
                np.max(np.abs(batch_trace - finite_difference))
            )
            finite_difference_max_error = (
                batch_error
                if finite_difference_max_error is None
                else max(finite_difference_max_error, batch_error)
            )

    quantiles = {
        "q50": float(np.quantile(trace_values, 0.50)),
        "q90": float(np.quantile(trace_values, 0.90)),
        "q95": float(np.quantile(trace_values, 0.95)),
        "q99": float(np.quantile(trace_values, 0.99)),
        "max": float(np.max(trace_values)),
    }
    log_kt = float(np.log(float(int(components) * steps)))
    return {
        "claim": 4,
        "metric": "trace-i-plus-score-jacobian",
        "scope": case["scope"],
        "components": int(components),
        "ambient_dimension": int(ambient_dimension),
        "active_rank": case["active_rank"],
        "geometry": geometry,
        "steps": steps,
        "time": time,
        "samples": samples,
        "seed": seed,
        "alpha_bar_t": alpha_bar_t,
        "theorem_identity_applied": theorem_identity_applied,
        "resource_method": "bounded-responsibility-batches",
        "responsibility_memory_cap_bytes": _RESPONSIBILITY_MEMORY_CAP_BYTES,
        "responsibility_estimated_bytes_per_sample": (
            responsibility_bytes_per_sample
        ),
        "responsibility_chunk_size": responsibility_chunk_size,
        "responsibility_chunk_count": (
            samples + responsibility_chunk_size - 1
        )
        // responsibility_chunk_size,
        "quantiles": quantiles,
        "log_kt": log_kt,
        "q95_over_log_kt": quantiles["q95"] / log_kt,
        "finite_difference_max_error": finite_difference_max_error,
    }


def summarize_jacobian(
    rows: Iterable[dict[str, object]],
) -> dict[str, object]:
    """Summarize finite theorem-family ratios without pooling controls."""
    all_rows = list(rows)
    theorem_rows = [
        row for row in all_rows if row.get("scope") == "theorem-family"
    ]
    controls = [
        row
        for row in all_rows
        if row.get("scope") == "out-of-scope-control"
    ]
    excluded_scope_rows = [
        row
        for row in all_rows
        if row.get("scope") not in {"theorem-family", "out-of-scope-control"}
    ]
    ratios = np.asarray(
        [float(row["q95_over_log_kt"]) for row in theorem_rows],
        dtype=np.float64,
    )
    failed_rules = ["row-scope"] if excluded_scope_rows else []
    if len(ratios) == 0:
        failed_rules.append("theorem-rows")
    if not np.all(np.isfinite(ratios)):
        failed_rules.append("finite-ratios")
    finite_ratios = ratios[np.isfinite(ratios)]
    return {
        "claim": 4,
        "scope": "theorem-family-summary",
        "theorem_row_count": len(theorem_rows),
        "control_row_count": len(controls),
        "excluded_scope_row_count": len(excluded_scope_rows),
        "median_q95_over_log_kt": (
            float(np.median(finite_ratios)) if len(finite_ratios) else None
        ),
        "max_q95_over_log_kt": (
            float(np.max(finite_ratios)) if len(finite_ratios) else None
        ),
        "failed_rules": failed_rules,
        "finite_support": not failed_rules,
    }
