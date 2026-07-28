from conditional_dpo_repro.equivalence import run_equivalence_lane


def test_equivalence_lane_distinguishes_two_dpo_objectives():
    result = run_equivalence_lane()
    assert result["case_count"] == 112
    assert result["population_stationary_max_abs_error"] <= 1e-8
    assert result["positive_loss_derivative_max"] < 0.0
    assert result["one_sided_finite_optimum"] is False
    assert result["population_identity_requires_positive_delta"] is False
    assert result["outcome"] == "mixed"


def test_equivalence_outcome_derivation_helper():
    from conditional_dpo_repro.equivalence import derive_equivalence_outcome
    assert derive_equivalence_outcome(
        max_stationary_err=1e-8,
        max_loss_deriv=-0.1,
        one_sided_finite_optimum=False,
    ) == "mixed"
    assert derive_equivalence_outcome(
        max_stationary_err=1.0,
        max_loss_deriv=-0.1,
        one_sided_finite_optimum=False,
    ) == "contradiction"
