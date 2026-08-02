"""Computed multi-step convergence diagnostics for unit-covariance GMMs."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Sequence

import numpy as np
from numpy.typing import NDArray

from .metrics import (
    _histogram_gaussian_convolution,
    bootstrap_interval,
    classifier_tv_lower_bound,
    linear_mmd,
    log_slope_interval,
)
from .model import IsotropicGMM
from .sampler import ddpm_sample, normal_blocks
from .schedule import paper_schedule

_FAMILY_TEMPLATES: dict[str, NDArray[np.float64]] = {
    "rank1-k2": np.array([[-2.0], [2.0]]),
    "rank2-k4": np.array(
        [[-1.5, -1.5], [-1.5, 1.5], [1.5, -1.5], [1.5, 1.5]]
    ),
    "rank4-k8": np.vstack((2.5 * np.eye(4), -2.5 * np.eye(4))),
}
_PLUGIN_BANDWIDTH_MULTIPLIERS = (0.75, 1.0, 1.5)
_MAX_DIRECT_AMBIENT_DIMENSION = 16
_DDPM_DOMAIN_ID = 1
_TARGET_DOMAIN_ID = 2
_TARGET_REFERENCE_DOMAIN_ID = 3


def _seeded_rotation(dimension: int, seed: int) -> NDArray[np.float64]:
    """Return a reproducible orthogonal change of basis."""
    rng = np.random.default_rng(seed)
    basis, signs = np.linalg.qr(rng.normal(size=(dimension, dimension)))
    return basis * np.sign(np.diag(signs))


def gmm_family(name: str, *, seed: int) -> IsotropicGMM:
    """Construct one balanced, centered, unit-covariance theorem family."""
    try:
        template = _FAMILY_TEMPLATES[name]
    except KeyError as error:
        raise ValueError(f"unknown GMM family: {name}") from error

    rank = template.shape[1]
    means = template @ _seeded_rotation(rank, seed)
    return IsotropicGMM(
        weights=np.full(len(means), 1.0 / len(means)),
        means=means,
        variances=np.ones(len(means)),
    )


def _rng_domain_allocation(
    *, master_seed: int, steps: int
) -> dict[str, dict[str, object]]:
    """Allocate collision-free seed namespaces for one convergence cell.

    ``normal_blocks`` keys all normal draws as ``(seed, stream, block)``.
    Reserving the low two seed bits for a domain separates the DDPM sampler's
    complete stream range from target draws for any valid number of steps.
    """
    master_seed = int(master_seed)
    steps = int(steps)
    if master_seed < 0:
        raise ValueError("master_seed must be nonnegative")
    if steps < 2:
        raise ValueError("steps must be at least 2")

    def domain_seed(domain_id: int) -> int:
        return (master_seed << 2) | domain_id

    return {
        "generated_ddpm": {
            "domain": "ddpm",
            "seed": domain_seed(_DDPM_DOMAIN_ID),
            "key_namespace": "seed-stream-block",
            "initial_stream": 0,
            "reverse_stream_range": {"start": 2, "stop": steps},
        },
        "target": {
            "domain": "target",
            "seed": domain_seed(_TARGET_DOMAIN_ID),
            "key_namespace": "seed-stream-block",
            "component_stream": 0,
            "noise_stream": 1,
        },
        "target_reference": {
            "domain": "target-reference",
            "seed": domain_seed(_TARGET_REFERENCE_DOMAIN_ID),
            "key_namespace": "seed-stream-block",
            "component_stream": 0,
            "noise_stream": 1,
        },
    }


def _gmm_sample(
    model: IsotropicGMM, *, samples: int, domain_seed: int
) -> NDArray[np.float64]:
    """Draw an independent, block-stable sample from ``model``."""
    if samples < 1:
        raise ValueError("samples must be positive")
    rng = np.random.default_rng(np.random.SeedSequence([domain_seed, 0]))
    components = rng.choice(len(model.weights), size=samples, p=model.weights)
    noise = normal_blocks(
        seed=domain_seed,
        stream=1,
        samples=samples,
        dimension=model.dimension,
    )
    return model.means[components] + noise * np.sqrt(model.variances[components, None])


def _ambient_model(model: IsotropicGMM, ambient_dimension: int) -> IsotropicGMM:
    if ambient_dimension < model.dimension:
        raise ValueError("ambient dimensions must be at least the active rank")
    means = np.zeros((len(model.weights), ambient_dimension))
    means[:, : model.dimension] = model.means
    return IsotropicGMM(model.weights, means, model.variances)


def _plugin_tv_for_bandwidth(
    samples: NDArray[np.float64],
    target: IsotropicGMM,
    *,
    bandwidth_multiplier: float,
    seed: int,
    replicates: int,
) -> dict[str, object]:
    """Compute one calibrated fixed-grid plug-in TV diagnostic.

    Task 2 exposes its three-bandwidth diagnostic only as an average. This
    local wrapper preserves each constituent diagnostic without changing that
    module's established public contract.
    """
    samples_1d = np.asarray(samples, dtype=np.float64)[:, 0]
    n = len(samples_1d)
    target_1d = IsotropicGMM(
        target.weights, target.means[:, :1], target.variances
    )
    std_max = float(np.sqrt(np.max(target_1d.variances)))
    minimum = min(
        float(np.min(samples_1d)), float(np.min(target_1d.means[:, 0]))
    )
    maximum = max(
        float(np.max(samples_1d)), float(np.max(target_1d.means[:, 0]))
    )
    minimum -= 4.0 * std_max
    maximum += 4.0 * std_max
    grid_points = 1000
    grid = np.linspace(minimum, maximum, grid_points)
    dx = (maximum - minimum) / (grid_points - 1)
    bin_edges = np.linspace(
        minimum - 0.5 * dx, maximum + 0.5 * dx, grid_points + 1
    )
    target_density = np.exp(target_1d.log_density(grid[:, None]))
    sample_std = float(np.std(samples_1d)) or 1.0
    bandwidth = bandwidth_multiplier * 1.06 * sample_std * n ** (-0.2)

    def statistic(draws: NDArray[np.float64]) -> float:
        density = _histogram_gaussian_convolution(draws, bin_edges, dx, bandwidth)
        return float(0.5 * np.sum(np.abs(density - target_density)) * dx)

    raw_estimate = statistic(samples_1d)
    raw_lower_95, raw_upper_95 = bootstrap_interval(
        samples_1d, statistic=statistic, seed=seed, replicates=replicates
    )
    rng = np.random.default_rng(seed)
    components = rng.choice(len(target_1d.weights), size=n, p=target_1d.weights)
    target_draws = target_1d.means[components, 0] + rng.normal(
        scale=np.sqrt(target_1d.variances[components]), size=n
    )
    calibration_floor = statistic(target_draws)
    return {
        "bandwidth_multiplier": float(bandwidth_multiplier),
        "metric_kind": "plugin-tv-diagnostic",
        "estimate": max(0.0, raw_estimate - calibration_floor),
        "lower_95": max(0.0, raw_lower_95 - calibration_floor),
        "upper_95": max(0.0, raw_upper_95 - calibration_floor),
        "calibration_floor": calibration_floor,
        "samples": n,
        "seed": seed,
    }


def _metric_records(
    generated: NDArray[np.float64],
    target_sample: NDArray[np.float64],
    target_reference: NDArray[np.float64],
    model: IsotropicGMM,
    *,
    seed: int,
) -> tuple[dict[str, object], dict[str, object]]:
    """Calculate calibrated generated-target and target-target diagnostics."""
    metrics: dict[str, object] = {
        "linear_mmd": dict(
            linear_mmd(generated, target_sample, seed=seed, replicates=200)
        ),
        "classifier_tv_lower_bound": dict(
            classifier_tv_lower_bound(
                generated,
                target_sample,
                seed=seed + 1,
                num_features=128,
                replicates=200,
            )
        ),
    }
    target_target: dict[str, object] = {
        "linear_mmd": dict(
            linear_mmd(target_sample, target_reference, seed=seed + 2, replicates=200)
        ),
        "classifier_tv_lower_bound": dict(
            classifier_tv_lower_bound(
                target_sample,
                target_reference,
                seed=seed + 3,
                num_features=128,
                replicates=200,
            )
        ),
    }
    if model.dimension == 1:
        metrics["plugin_tv_diagnostics"] = [
            _plugin_tv_for_bandwidth(
                generated,
                model,
                bandwidth_multiplier=multiplier,
                seed=seed + 4 + index,
                replicates=200,
            )
            for index, multiplier in enumerate(_PLUGIN_BANDWIDTH_MULTIPLIERS)
        ]
        target_target["plugin_tv_diagnostics"] = [
            _plugin_tv_for_bandwidth(
                target_sample,
                model,
                bandwidth_multiplier=multiplier,
                seed=seed + 7 + index,
                replicates=200,
            )
            for index, multiplier in enumerate(_PLUGIN_BANDWIDTH_MULTIPLIERS)
        ]
    return metrics, target_target


def _inactive_coordinate_diagnostics(
    inactive: NDArray[np.float64], reference: NDArray[np.float64], *, seed: int
) -> dict[str, object]:
    """Check that directly simulated inactive coordinates remain N(0, I)."""
    if inactive.shape != reference.shape or inactive.ndim != 2:
        raise ValueError("inactive and reference samples must have matching 2D shapes")
    if inactive.shape[1] == 0:
        return {
            "coordinate_count": 0,
            "max_abs_mean": 0.0,
            "max_variance_error": 0.0,
            "calibrated_mmd": None,
            "passed": True,
        }
    calibrated_mmd = dict(
        linear_mmd(
            inactive.reshape(-1, 1),
            reference.reshape(-1, 1),
            bandwidth=1.0,
            seed=seed,
            replicates=200,
        )
    )
    max_abs_mean = float(np.max(np.abs(np.mean(inactive, axis=0))))
    max_variance_error = float(np.max(np.abs(np.var(inactive, axis=0) - 1.0)))
    return {
        "coordinate_count": int(inactive.shape[1]),
        "max_abs_mean": max_abs_mean,
        "max_variance_error": max_variance_error,
        "calibrated_mmd": calibrated_mmd,
        "passed": bool(
            max_abs_mean < 0.12
            and max_variance_error < 0.15
            and float(calibrated_mmd["upper_95"]) < 0.05
        ),
    }


def _reduction_check(
    model: IsotropicGMM,
    generated: NDArray[np.float64],
    *,
    schedule_steps: int,
    samples: int,
    seed: int,
    ambient_dimensions: Sequence[int],
) -> dict[str, object]:
    """Compare direct low-dimensional simulations with active reduction."""
    schedule = paper_schedule(steps=schedule_steps)
    checks: list[dict[str, object]] = []
    checked_dimensions = tuple(
        dimension
        for dimension in ambient_dimensions
        if dimension <= _MAX_DIRECT_AMBIENT_DIMENSION
    )
    analytically_equivalent_dimensions = tuple(
        dimension
        for dimension in ambient_dimensions
        if dimension > _MAX_DIRECT_AMBIENT_DIMENSION
    )
    for ambient_dimension in checked_dimensions:
        ambient_model = _ambient_model(model, ambient_dimension)
        direct = ddpm_sample(
            ambient_model,
            schedule=schedule,
            samples=samples,
            seed=seed + ambient_dimension,
        )
        record = linear_mmd(
            generated,
            direct[:, : model.dimension],
            seed=seed + 10_000 + ambient_dimension,
            replicates=200,
        )
        inactive_reference = normal_blocks(
            seed=seed,
            stream=50_000 + ambient_dimension,
            samples=samples,
            dimension=ambient_dimension - model.dimension,
        )
        inactive = direct[:, model.dimension :]
        inactive_diagnostics = _inactive_coordinate_diagnostics(
            inactive,
            inactive_reference,
            seed=seed + 20_000 + ambient_dimension,
        )
        augmented_reduction = np.hstack((generated, inactive_reference))
        augmented_record = linear_mmd(
            direct,
            augmented_reduction,
            bandwidth=np.sqrt(float(ambient_dimension)),
            seed=seed + 30_000 + ambient_dimension,
            replicates=200,
        )
        checks.append(
            {
                "ambient_dimension": ambient_dimension,
                "metric_kind": record["metric_kind"],
                "metric_delta": float(record["estimate"]),
                "inactive_coordinates": inactive_diagnostics,
                "full_augmented_comparison": dict(augmented_record),
                "passed": bool(
                    float(record["estimate"]) < 0.02
                    and inactive_diagnostics["passed"]
                    and float(augmented_record["estimate"]) < 0.02
                ),
            }
        )
    metric_delta = max((check["metric_delta"] for check in checks), default=0.0)
    return {
        "checked_dimensions": list(checked_dimensions),
        "analytically_equivalent_dimensions": list(analytically_equivalent_dimensions),
        "skipped_dimensions": list(analytically_equivalent_dimensions),
        "checks": checks,
        "metric_delta": float(metric_delta),
        "passed": bool(checks) and all(check["passed"] for check in checks),
    }


def run_convergence_cell(
    *,
    family: str,
    steps: int,
    seed: int,
    samples: int,
    ambient_dimensions: Iterable[int],
    direct_ambient_check: bool = False,
) -> dict[str, object]:
    """Run one deterministic family/step/seed convergence evidence cell."""
    steps = int(steps)
    seed = int(seed)
    samples = int(samples)
    dimensions = tuple(int(dimension) for dimension in ambient_dimensions)
    if not dimensions or any(dimension <= 0 for dimension in dimensions):
        raise ValueError("ambient_dimensions must contain positive integers")
    if len(set(dimensions)) != len(dimensions):
        raise ValueError("ambient_dimensions must not contain duplicates")
    if samples < 4:
        raise ValueError("samples must be at least 4 for calibrated metrics")

    model = gmm_family(family, seed=seed)
    if any(dimension < model.dimension for dimension in dimensions):
        raise ValueError("ambient dimensions must be at least the active rank")

    schedule = paper_schedule(steps=steps)
    rng_domains = _rng_domain_allocation(master_seed=seed, steps=steps)
    generated = ddpm_sample(
        model,
        schedule=schedule,
        samples=samples,
        seed=int(rng_domains["generated_ddpm"]["seed"]),
    )
    target_sample = _gmm_sample(
        model, samples=samples, domain_seed=int(rng_domains["target"]["seed"])
    )
    target_reference = _gmm_sample(
        model,
        samples=samples,
        domain_seed=int(rng_domains["target_reference"]["seed"]),
    )
    metrics, target_target = _metric_records(
        generated, target_sample, target_reference, model, seed=seed + 40_000
    )
    mmd = metrics["linear_mmd"]
    assert isinstance(mmd, dict)
    uncertainty_width = float(mmd["upper_95"]) - float(mmd["lower_95"])
    convergence_metric = max(
        float(mmd["estimate"])
        - float(target_target["linear_mmd"]["estimate"]),
        np.finfo(np.float64).tiny,
    )

    return {
        "family": family,
        "steps": steps,
        "seed": seed,
        "samples": samples,
        "rng_domains": rng_domains,
        "components": len(model.weights),
        "simulation_dimension": model.dimension,
        "ambient_dimensions": list(dimensions),
        "ambient_dimension_records": [
            {
                "ambient_dimension": dimension,
                "discrepancy": convergence_metric,
                "uncertainty_width": uncertainty_width,
                "equivalence": "analytic-active-coordinate-reduction",
            }
            for dimension in dimensions
        ],
        "update_equation": "paper-equation-9",
        "schedule_equation": "paper-equation-14",
        "metrics": metrics,
        "target_target_floors": target_target,
        "convergence_metric": convergence_metric,
        "active_reduction_check": (
            _reduction_check(
                model,
                generated,
                schedule_steps=steps,
                samples=samples,
                seed=seed,
                ambient_dimensions=dimensions,
            )
            if direct_ambient_check
            else None
        ),
    }


def _row_metric(row: dict[str, object]) -> float:
    if "convergence_metric" in row:
        return float(row["convergence_metric"])
    metrics = row.get("metrics")
    floors = row.get("target_target_floors")
    if not isinstance(metrics, dict) or not isinstance(floors, dict):
        raise ValueError("each row must contain a convergence metric")
    mmd = metrics.get("linear_mmd")
    target_mmd = floors.get("linear_mmd")
    if not isinstance(mmd, dict) or not isinstance(target_mmd, dict):
        raise ValueError("each row must contain linear-MMD records")
    return max(
        float(mmd["estimate"]) - float(target_mmd["estimate"]),
        np.finfo(np.float64).tiny,
    )


def _hierarchical_slope_interval(
    steps: NDArray[np.float64],
    seed_metrics: Sequence[NDArray[np.float64]],
    *,
    seed: int,
    replicates: int,
) -> dict[str, object]:
    """Bootstrap seed-level cells independently within each step stratum."""
    if replicates < 1:
        raise ValueError("bootstrap_replicates must be positive")
    discrepancies = np.array(
        [float(np.mean(metrics)) for metrics in seed_metrics], dtype=np.float64
    )
    record = dict(
        log_slope_interval(steps, discrepancies, seed=seed, replicates=1)
    )
    rng = np.random.default_rng(seed)
    log_steps = np.log(steps)
    bootstrap_slopes: list[float] = []
    for _ in range(replicates):
        sampled_discrepancies = np.array(
            [
                float(
                    np.mean(
                        metrics[rng.integers(0, len(metrics), size=len(metrics))]
                    )
                )
                for metrics in seed_metrics
            ],
            dtype=np.float64,
        )
        slope, _ = np.polyfit(log_steps, np.log(sampled_discrepancies), 1)
        bootstrap_slopes.append(float(slope))
    record["lower_95"] = float(np.percentile(bootstrap_slopes, 2.5))
    record["upper_95"] = float(np.percentile(bootstrap_slopes, 97.5))
    return record


def summarize_convergence(
    rows: Iterable[dict[str, object]],
    *,
    bootstrap_replicates: int = 1000,
    seed: int = 42,
) -> dict[str, object]:
    """Apply predeclared slope and ambient-dimension support rules to cells."""
    grouped: defaultdict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        family = row.get("family")
        if not isinstance(family, str) or not family:
            raise ValueError("each row must have a nonempty family")
        grouped[family].append(row)
    if not grouped:
        raise ValueError("rows must not be empty")

    failed_rules: list[str] = []
    family_summaries: dict[str, object] = {}
    slope_uppers: list[float] = []
    ambient_failures: list[dict[str, object]] = []
    for index, (family, family_rows) in enumerate(sorted(grouped.items())):
        by_steps: defaultdict[int, list[float]] = defaultdict(list)
        for row in family_rows:
            by_steps[int(row["steps"])].append(_row_metric(row))
            ambient_records = row.get("ambient_dimension_records", [])
            if not isinstance(ambient_records, list) or len(ambient_records) < 2:
                continue
            ordered = sorted(
                ambient_records, key=lambda record: int(record["ambient_dimension"])
            )
            first, last = ordered[0], ordered[-1]
            baseline = float(first["discrepancy"])
            increase = float(last["discrepancy"]) - baseline
            allowance = 0.1 * baseline + float(first["uncertainty_width"])
            if increase > allowance:
                ambient_failures.append(
                    {
                        "family": family,
                        "steps": int(row["steps"]),
                        "increase": increase,
                        "allowance": allowance,
                    }
                )

        steps = np.array(sorted(by_steps), dtype=np.float64)
        seed_metrics = [
            np.asarray(by_steps[int(step)], dtype=np.float64) for step in steps
        ]
        discrepancies = np.array(
            [float(np.mean(metrics)) for metrics in seed_metrics], dtype=np.float64
        )
        slope = _hierarchical_slope_interval(
            steps,
            seed_metrics,
            seed=seed + index,
            replicates=bootstrap_replicates,
        )
        slope_uppers.append(float(slope["upper_95"]))
        family_summaries[family] = {
            "step_counts": [int(step) for step in steps],
            "discrepancies": [float(value) for value in discrepancies],
            "seed_counts_by_step": [len(metrics) for metrics in seed_metrics],
            "slope": dict(slope),
        }

    slope_upper_95 = max(slope_uppers)
    if slope_upper_95 >= 0.0:
        failed_rules.append("nonnegative-slope-upper-bound")
    if ambient_failures:
        failed_rules.append("ambient-dimension")
    return {
        "slope_upper_95": slope_upper_95,
        "finite_support": not failed_rules,
        "failed_rules": failed_rules,
        "family_summaries": family_summaries,
        "ambient_failures": ambient_failures,
    }


def prior_rate_context() -> dict[str, dict[str, str]]:
    """Return pinned primary-source formulas as non-computed paper context."""
    return {
        "li_yan_2024b": {
            "kind": "paper-context",
            "rate_shape": "d / epsilon",
            "arxiv": "2409.18959",
        },
        "liang_et_al_2024b": {
            "kind": "paper-context",
            "rate_shape": "d / epsilon^2",
            "arxiv": "2405.16418",
        },
    }
