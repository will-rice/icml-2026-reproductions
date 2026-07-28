from conditional_dpo_repro.failure_modes import run_undesirable_space_lane


def test_undesirable_lane_emits_concrete_witnesses():
    result = run_undesirable_space_lane()
    assert result["witness_count"] > 0
    for witness in result["witnesses"]:
        assert witness["delta_ref"] < witness["delta"] < 0.0
        assert witness["candidate_loss"] < witness["reference_loss"]
        assert witness["preferred_probability"] < 0.5
    assert result["outcome"] == "consistent"


def test_undesirable_space_outcome_derivation_helper():
    from conditional_dpo_repro.failure_modes import derive_undesirable_space_outcome
    assert derive_undesirable_space_outcome(5) == "consistent"
    assert derive_undesirable_space_outcome(0) == "contradiction"
