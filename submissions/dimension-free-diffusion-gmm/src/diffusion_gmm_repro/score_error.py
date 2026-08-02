"""Paired multi-step audits for time-averaged score error."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable

import numpy as np

from .convergence import gmm_family
from .metrics import bootstrap_interval
from .sampler import constant_rms_profile, ddpm_sample
from .schedule import paper_schedule


def run_score_error_cell(
    *,
    family: str,
    steps: int,
    seed: int,
    samples: int,
    epsilon_score: float,
    profile_shape: str,
) -> dict[str, object]:
    """Run exact and perturbed DDPM paths with identical random streams."""
    steps = int(steps)
    seed = int(seed)
    samples = int(samples)
    epsilon_score = float(epsilon_score)
    model = gmm_family(family, seed=seed)
    schedule = paper_schedule(steps=steps)
    profile = constant_rms_profile(
        steps=steps, rms=epsilon_score, shape=profile_shape
    )
    error_direction = np.ones(model.dimension, dtype=np.float64)
    error_direction /= np.linalg.norm(error_direction)

    exact = ddpm_sample(
        model, schedule=schedule, samples=samples, seed=seed
    )
    perturbed = ddpm_sample(
        model,
        schedule=schedule,
        samples=samples,
        seed=seed,
        score_profile=profile,
        error_direction=error_direction,
    )
    paired_distances = np.linalg.norm(perturbed - exact, axis=1)
    rms_score_error = float(np.sqrt(np.mean(profile[1:] ** 2)))
    bound_scale = float(rms_score_error * np.sqrt(np.log(float(steps))))
    rms_l2 = float(np.sqrt(np.mean(paired_distances**2)))
    return {
        "claim": 3,
        "scope": "theorem-family",
        "family": family,
        "steps": steps,
        "seed": seed,
        "samples": samples,
        "epsilon_score": epsilon_score,
        "profile_shape": profile_shape,
        "rms_score_error": rms_score_error,
        "bound_scale": bound_scale,
        "pairing": "identical-start-and-noise-streams",
        "paired_excess": {
            "mean_l2": float(np.mean(paired_distances)),
            "median_l2": float(np.median(paired_distances)),
            "q95_l2": float(np.quantile(paired_distances, 0.95)),
            "rms_l2": rms_l2,
        },
        "normalized_ratio": 0.0 if bound_scale == 0.0 else rms_l2 / bound_scale,
    }


def summarize_score_error(
    rows: Iterable[dict[str, object]],
    *,
    seed: int,
    bootstrap_replicates: int,
) -> dict[str, object]:
    """Check monotonic response and the pooled normalized-ratio rule."""
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
    if not theorem_rows:
        raise ValueError("at least one theorem-family row is required")
    if bootstrap_replicates < 1:
        raise ValueError("bootstrap_replicates must be positive")

    grouped: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in theorem_rows:
        grouped[(str(row["family"]), str(row["profile_shape"]))].append(row)

    failed_rules = ["row-scope"] if excluded_scope_rows else []
    series_records: list[dict[str, object]] = []
    bootstrap_seed = int(seed)
    for (family, shape), series_rows in sorted(grouped.items()):
        by_epsilon: dict[float, list[float]] = defaultdict(list)
        for row in series_rows:
            epsilon = float(row["epsilon_score"])
            excess = float(dict(row["paired_excess"])["rms_l2"])
            if not np.isfinite(epsilon) or not np.isfinite(excess):
                raise ValueError("score-error rows must contain finite metrics")
            by_epsilon[epsilon].append(excess)

        medians: list[float] = []
        points: list[dict[str, object]] = []
        for epsilon in sorted(by_epsilon):
            values = np.asarray(by_epsilon[epsilon], dtype=np.float64)
            median = float(np.median(values))
            medians.append(median)
            points.append(
                {
                    "epsilon_score": epsilon,
                    "median_paired_excess": median,
                }
            )
        if any(current < previous for previous, current in zip(medians, medians[1:])):
            failed_rules.append("monotonicity")
        series_records.append(
            {"family": family, "profile_shape": shape, "points": points}
        )

    ratio_cells: dict[tuple[str, str, float], list[float]] = defaultdict(list)
    for row in theorem_rows:
        epsilon = float(row["epsilon_score"])
        if epsilon > 0.0:
            ratio_cells[
                (
                    str(row["profile_shape"]),
                    str(row["family"]),
                    epsilon,
                )
            ].append(float(row["normalized_ratio"]))
    cell_medians = {
        cell: float(np.median(values)) for cell, values in ratio_cells.items()
    }
    pooled_ratios = np.asarray(list(cell_medians.values()), dtype=np.float64)
    if not np.all(np.isfinite(pooled_ratios)):
        raise ValueError("normalized ratios must be finite")
    pooled_ratio = (
        float(np.median(pooled_ratios)) if len(pooled_ratios) else 0.0
    )
    ratio_limit = 2.0 * pooled_ratio
    profile_cells: dict[str, list[float]] = defaultdict(list)
    for (profile_shape, _family, _epsilon), median in cell_medians.items():
        profile_cells[profile_shape].append(median)
    profile_ratio_bounds: list[dict[str, object]] = []
    for profile_index, (profile_shape, values) in enumerate(
        sorted(profile_cells.items())
    ):
        profile_values = np.asarray(values, dtype=np.float64)
        lower, upper = bootstrap_interval(
            profile_values,
            statistic=lambda sample: float(np.median(sample)),
            seed=bootstrap_seed + 10_000 + profile_index,
            replicates=bootstrap_replicates,
        )
        passed = upper <= ratio_limit
        if not passed:
            failed_rules.append("normalized-ratio-upper-95")
        profile_ratio_bounds.append(
            {
                "profile_shape": profile_shape,
                "cell_count": len(profile_values),
                "median_normalized_ratio": float(
                    np.median(profile_values)
                ),
                "lower_95": lower,
                "upper_95": upper,
                "limit": ratio_limit,
                "passed": passed,
            }
        )

    failed_rules = list(dict.fromkeys(failed_rules))
    return {
        "claim": 3,
        "scope": "theorem-family-summary",
        "acceptance_rule": "two-times-pooled-normalized-ratio",
        "pooled_normalized_ratio": pooled_ratio,
        "normalized_ratio_limit": ratio_limit,
        "profile_ratio_bounds": profile_ratio_bounds,
        "series": series_records,
        "theorem_row_count": len(theorem_rows),
        "control_row_count": len(controls),
        "excluded_scope_row_count": len(excluded_scope_rows),
        "failed_rules": failed_rules,
        "finite_support": not failed_rules,
    }
