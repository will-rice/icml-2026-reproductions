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
