from conditional_dpo_repro.failure_modes import run_undesirable_space_lane


def test_undesirable_lane_emits_concrete_witnesses():
    result = run_undesirable_space_lane()
    assert result["witness_count"] > 0
    for witness in result["witnesses"]:
        assert witness["delta_ref"] < witness["delta"] < 0.0
        assert witness["candidate_loss"] < witness["reference_loss"]
        assert witness["preferred_probability"] < 0.5
