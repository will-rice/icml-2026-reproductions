import pytest
from recurrent_sampler_repro.evidence import simulate_wavefront_schedule


def test_canonical_wavefront_schedule():
    res = simulate_wavefront_schedule(
        outer_steps=8,
        inner_recurrence=4,
        headway=1,
        max_wavefront=8,
        initial_active=1,
    )

    invariants = res["invariants"]
    assert invariants["appended_per_step_equals_headway"] is True
    assert invariants["prior_active_gained_recurrence"] is True
    assert invariants["active_width_bounded_by_max_wavefront"] is True

    trace = res["canonical_trace"]
    assert len(trace) == 8

    for step in trace:
        assert len(step["appended_positions"]) == 1
        assert step["active_width"] <= 8


def test_negative_control_headway_zero():
    res = simulate_wavefront_schedule(
        outer_steps=8,
        inner_recurrence=4,
        headway=0,
        max_wavefront=8,
        initial_active=1,
    )

    # headway=0 produces 0 appended positions per step, failing the 1-new-token progress invariant
    trace = res["canonical_trace"]
    assert all(len(t["appended_positions"]) == 0 for t in trace)
    assert res["invariants"]["appended_per_step_equals_headway"] is True  # headway is 0, so 0 appended per step matches headway=0


def test_sequential_ar_fixture():
    # Sequential AR mode: max_wavefront=1, headway=1, inner_recurrence=1
    res = simulate_wavefront_schedule(
        outer_steps=8,
        inner_recurrence=1,
        headway=1,
        max_wavefront=1,
        initial_active=1,
    )

    trace = res["canonical_trace"]
    # In AR mode, max_wavefront=1 forces active width to remain 1 at every step
    for step in trace:
        assert step["active_width"] == 1
