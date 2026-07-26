import pytest
from recurrent_sampler_repro.evidence import simulate_wavefront_schedule


def test_wavefront_schedule_canonical_invariants():
    res = simulate_wavefront_schedule()
    invs = res["invariants"]
    assert invs["appended_per_step_equals_headway"] is True
    assert invs["prior_active_gained_recurrence"] is True
    assert invs["active_width_bounded_by_max_wavefront"] is True
    assert invs["one_new_position_per_step"] is True
    assert invs["multi_position_wavefront_observed"] is True


def test_wavefront_schedule_negative_controls_recorded():
    res = simulate_wavefront_schedule()
    ctrls = res["negative_controls"]
    assert ctrls["headway_zero"]["one_new_position_per_step"] is False
    assert ctrls["max_wavefront_one"]["multi_position_wavefront_observed"] is False


def test_wavefront_schedule_prefix_truncation():
    # Model lines 1382-1388: states[:, :max_wavefront] retains the prefix, not suffix.
    res = simulate_wavefront_schedule(outer_steps=10, max_wavefront=8, headway=1, initial_active=1)
    trace = res["canonical_trace"]

    # At capacity, candidate 8 is sampled but prefix truncation retains none of it.
    step8 = trace[7]
    assert step8["active_positions_after"] == [0, 1, 2, 3, 4, 5, 6, 7]
    assert step8["candidate_positions"] == [8]
    assert step8["retained_appended_positions"] == []
    assert step8["headway_in_step"] == 0

    # The next candidate is 9; the active prefix remains unchanged.
    step9_after = trace[8]["active_positions_after"]
    assert step9_after == [0, 1, 2, 3, 4, 5, 6, 7]
    assert 0 in step9_after
    assert 8 not in step9_after
    assert trace[8]["candidate_positions"] == [9]
    assert trace[8]["retained_appended_positions"] == []
