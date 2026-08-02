from diffusion_gmm_repro.audit import (
    run_dimension_audit,
    run_jacobian_audit,
    run_score_error_audit,
)


def test_dimension_audit_is_seed_deterministic() -> None:
    first = run_dimension_audit(dimensions=(2, 7), seed=41, samples=128)
    second = run_dimension_audit(dimensions=(2, 7), seed=41, samples=128)
    changed_seed = run_dimension_audit(dimensions=(2, 7), seed=42, samples=128)

    assert first == second
    assert [row["sample_checksum"] for row in first] != [
        row["sample_checksum"] for row in changed_seed
    ]


def test_exact_gaussian_discretization_diagnostic_is_dimension_invariant() -> None:
    records = run_dimension_audit(
        dimensions=(1, 4, 32), step_size=0.125, seed=7, samples=64
    )

    assert [row["analytic_per_coordinate_variance_error"] for row in records] == [
        0.125**2,
        0.125**2,
        0.125**2,
    ]
    assert all(row["assumption_satisfied"] for row in records)


def test_score_error_audit_degrades_monotonically() -> None:
    records = run_score_error_audit(
        error_levels=(0.0, 0.1, 0.25, 0.5), dimension=5, step_size=0.2, seed=9
    )

    in_scope = [row for row in records if row["assumption_satisfied"]]
    controls = [row for row in records if not row["assumption_satisfied"]]
    shifts = [row["analytic_mean_shift_l2"] for row in in_scope]
    assert shifts == sorted(shifts)
    assert shifts[0] == 0.0
    assert shifts[-1] > shifts[1]
    assert len(controls) == 1
    assert controls[0]["control"] == "state-dependent-misspecified-score"


def test_jacobian_audit_includes_assumption_breaking_control() -> None:
    records = run_jacobian_audit(dimensions=(2, 8), seed=3, samples=96)

    within_assumptions = [row for row in records if row["assumption_satisfied"]]
    controls = [row for row in records if not row["assumption_satisfied"]]
    assert len(within_assumptions) == len(controls) == 2
    assert {row["control"] for row in controls} == {"unit-covariance-violation"}
    assert {row["components"] for row in within_assumptions} == {2}
    assert max(row["finite_difference_error"] for row in within_assumptions) < 1e-5
    assert (
        max(row["max_trace_i_plus_j"] for row in within_assumptions)
        - min(row["max_trace_i_plus_j"] for row in within_assumptions)
        < 1e-12
    )
    assert all(
        abs(control["mean_trace_i_plus_j"])
        > baseline["max_trace_i_plus_j"]
        for baseline, control in zip(within_assumptions, controls, strict=True)
    )
