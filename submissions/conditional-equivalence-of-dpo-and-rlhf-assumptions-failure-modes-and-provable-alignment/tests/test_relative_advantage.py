from conditional_dpo_repro.failure_modes import run_relative_advantage_lane


def test_relative_lane_separates_relative_and_absolute_preference():
    result = run_relative_advantage_lane()
    assert result["case_count"] == 75
    assert result["relative_improvement_count"] == 75
    assert result["absolute_preference_count"] < 75
    assert result["relative_but_not_absolute_count"] > 0
