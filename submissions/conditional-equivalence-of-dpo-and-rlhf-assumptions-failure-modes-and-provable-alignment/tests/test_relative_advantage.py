from conditional_dpo_repro.failure_modes import run_relative_advantage_lane


def test_relative_lane_separates_relative_and_absolute_preference():
    result = run_relative_advantage_lane()
    assert result["case_count"] == 75
    assert result["relative_improvement_count"] == 75
    assert result["absolute_preference_count"] < 75
    assert result["outcome"] == "consistent"


def test_relative_advantage_outcome_derivation_helper():
    from conditional_dpo_repro.failure_modes import derive_relative_advantage_outcome
    assert derive_relative_advantage_outcome(75, 75, 10) == "consistent"
    assert derive_relative_advantage_outcome(75, 0, 0) == "contradiction"
    assert derive_relative_advantage_outcome(75, 50, 0) == "mixed"
