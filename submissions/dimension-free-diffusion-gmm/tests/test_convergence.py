import json

import numpy as np
import pytest

from diffusion_gmm_repro.convergence import (
    _inactive_coordinate_diagnostics,
    _rng_domain_allocation,
    gmm_family,
    prior_rate_context,
    run_convergence_cell,
    summarize_convergence,
)


@pytest.mark.parametrize(
    ("name", "components", "rank"),
    [("rank1-k2", 2, 1), ("rank2-k4", 4, 2), ("rank4-k8", 8, 4)],
)
def test_gmm_families_have_requested_rank(
    name: str, components: int, rank: int
) -> None:
    model = gmm_family(name, seed=7)
    assert len(model.weights) == components
    assert np.linalg.matrix_rank(model.means - model.means.mean(axis=0)) == rank
    np.testing.assert_allclose(model.variances, 1.0)
    np.testing.assert_allclose(model.means.mean(axis=0), 0.0, atol=1e-12)
    np.testing.assert_allclose(model.weights, np.full(components, 1.0 / components))
    assert np.linalg.norm(model.means, axis=1).max() <= 3.0


def test_convergence_cell_records_exact_equivalent_dimensions() -> None:
    record = run_convergence_cell(
        family="rank1-k2",
        steps=128,
        seed=3,
        samples=2048,
        ambient_dimensions=(4, 16, 64, 256, 1024),
    )
    assert record["ambient_dimensions"] == [4, 16, 64, 256, 1024]
    assert record["simulation_dimension"] == 1
    assert record["update_equation"] == "paper-equation-9"
    assert record["schedule_equation"] == "paper-equation-14"


def test_direct_ambient_check_matches_active_reduction() -> None:
    record = run_convergence_cell(
        family="rank2-k4",
        steps=128,
        seed=3,
        samples=2048,
        ambient_dimensions=(4, 16),
        direct_ambient_check=True,
    )
    assert record["active_reduction_check"]["passed"] is True
    assert record["active_reduction_check"]["metric_delta"] < 0.02


def test_rank_one_retains_three_distinct_plugin_bandwidth_records() -> None:
    record = run_convergence_cell(
        family="rank1-k2",
        steps=128,
        seed=8,
        samples=256,
        ambient_dimensions=(4,),
    )
    diagnostics = record["metrics"]["plugin_tv_diagnostics"]
    target_diagnostics = record["target_target_floors"]["plugin_tv_diagnostics"]
    assert [item["bandwidth_multiplier"] for item in diagnostics] == [0.75, 1.0, 1.5]
    assert [item["bandwidth_multiplier"] for item in target_diagnostics] == [
        0.75,
        1.0,
        1.5,
    ]
    assert all(item["metric_kind"] == "plugin-tv-diagnostic" for item in diagnostics)


def test_direct_ambient_check_skips_high_dimensional_simulations() -> None:
    record = run_convergence_cell(
        family="rank1-k2",
        steps=128,
        seed=9,
        samples=64,
        ambient_dimensions=(4, 16, 256, 1024),
        direct_ambient_check=True,
    )
    check = record["active_reduction_check"]
    assert check["checked_dimensions"] == [4, 16]
    assert check["analytically_equivalent_dimensions"] == [256, 1024]
    assert check["skipped_dimensions"] == [256, 1024]


def test_inactive_coordinate_diagnostic_rejects_shifted_inactive_samples() -> None:
    rng = np.random.default_rng(11)
    reference = rng.normal(size=(2048, 3))
    baseline = _inactive_coordinate_diagnostics(reference, reference.copy(), seed=12)
    shifted = _inactive_coordinate_diagnostics(reference + 0.75, reference, seed=12)
    assert baseline["passed"] is True
    assert shifted["passed"] is False


def test_convergence_cell_is_json_serializable_with_numpy_integer_inputs() -> None:
    record = run_convergence_cell(
        family="rank1-k2",
        steps=np.int64(128),
        seed=np.int64(4),
        samples=np.int64(256),
        ambient_dimensions=(np.int64(4),),
    )
    json.dumps(record)


def synthetic_convergence_rows(
    *,
    slopes: tuple[float, ...] = (-0.7,),
    dimension_multiplier: float = 1.0,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for index, slope in enumerate(slopes):
        for steps in (32, 64, 128, 256):
            discrepancy = float(steps**slope)
            rows.append(
                {
                    "family": f"family-{index}",
                    "steps": steps,
                    "convergence_metric": discrepancy,
                    "ambient_dimension_records": [
                        {
                            "ambient_dimension": 4,
                            "discrepancy": discrepancy,
                            "uncertainty_width": 0.0,
                        },
                        {
                            "ambient_dimension": 1024,
                            "discrepancy": discrepancy * dimension_multiplier,
                            "uncertainty_width": 0.0,
                        },
                    ],
                }
            )
    return rows


def test_summary_hierarchical_bootstrap_reflects_seed_level_variation() -> None:
    baseline = synthetic_convergence_rows(slopes=(-0.7,))
    variable: list[dict[str, object]] = []
    for row in baseline:
        for seed, multiplier in enumerate((0.5, 1.5)):
            copied = dict(row)
            copied["seed"] = seed
            copied["convergence_metric"] = float(row["convergence_metric"]) * multiplier
            variable.append(copied)

    stable_summary = summarize_convergence(baseline, bootstrap_replicates=200, seed=23)
    variable_summary = summarize_convergence(
        variable, bootstrap_replicates=200, seed=23
    )
    stable_slope = stable_summary["family_summaries"]["family-0"]["slope"]
    variable_slope = variable_summary["family_summaries"]["family-0"]["slope"]
    stable_width = stable_slope["upper_95"] - stable_slope["lower_95"]
    variable_width = variable_slope["upper_95"] - variable_slope["lower_95"]
    assert variable_width > stable_width


@pytest.mark.parametrize("family", ["rank1-k2", "rank2-k4", "rank4-k8"])
def test_all_families_record_classifier_and_mmd_diagnostics(family: str) -> None:
    record = run_convergence_cell(
        family=family,
        steps=128,
        seed=13,
        samples=64,
        ambient_dimensions=(4, 16),
    )
    assert {"linear_mmd", "classifier_tv_lower_bound"} <= set(record["metrics"])
    assert {"linear_mmd", "classifier_tv_lower_bound"} <= set(
        record["target_target_floors"]
    )
    if family == "rank1-k2":
        assert "plugin_tv_diagnostics" in record["metrics"]


def test_cell_is_deterministic_with_disjoint_rng_domains() -> None:
    first = run_convergence_cell(
        family="rank2-k4",
        steps=128,
        seed=17,
        samples=64,
        ambient_dimensions=(4,),
    )
    second = run_convergence_cell(
        family="rank2-k4",
        steps=128,
        seed=17,
        samples=64,
        ambient_dimensions=(4,),
    )
    assert first == second
    assert "sample_streams" not in first
    domains = first["rng_domains"]
    assert set(domains) == {"generated_ddpm", "target", "target_reference"}
    assert len({domain["seed"] for domain in domains.values()}) == 3
    assert domains["generated_ddpm"]["reverse_stream_range"] == {
        "start": 2,
        "stop": 128,
    }


@pytest.mark.parametrize("steps", [128, 1_000_000])
def test_rng_domain_allocator_is_disjoint_from_full_ddpm_stream_range(steps: int) -> None:
    domains = _rng_domain_allocation(master_seed=29, steps=steps)
    generated = domains["generated_ddpm"]
    target = domains["target"]
    target_reference = domains["target_reference"]

    assert generated["reverse_stream_range"] == {"start": 2, "stop": steps}
    assert generated["initial_stream"] == 0
    assert target["component_stream"] == target_reference["component_stream"] == 0
    assert target["noise_stream"] == target_reference["noise_stream"] == 1
    assert generated["seed"] not in {target["seed"], target_reference["seed"]}
    assert target["seed"] != target_reference["seed"]
    assert all(domain["key_namespace"] == "seed-stream-block" for domain in domains.values())


def test_summary_requires_negative_upper_slope_bound() -> None:
    rows = synthetic_convergence_rows(slopes=(-0.8, -0.7, -0.6, -0.5))
    summary = summarize_convergence(rows, bootstrap_replicates=200, seed=5)
    assert summary["slope_upper_95"] < 0.0
    assert summary["finite_support"] is True


def test_summary_rejects_dimension_increase_over_ten_percent() -> None:
    rows = synthetic_convergence_rows(dimension_multiplier=1.25)
    summary = summarize_convergence(rows, bootstrap_replicates=200, seed=5)
    assert summary["finite_support"] is False
    assert "ambient-dimension" in summary["failed_rules"]


def test_prior_rate_context_contains_only_pinned_paper_context() -> None:
    assert prior_rate_context() == {
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
