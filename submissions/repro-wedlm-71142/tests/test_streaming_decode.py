from wedlm_repro.causal_diffusion import simulate_streaming_decode


def test_streaming_decode_commits_confident_left_edge_tokens_and_refills_window():
    """Would fail if finalized tokens were not committed into a growing prefix."""
    trace = simulate_streaming_decode(
        prompt_tokens=["Solve:"],
        planned_tokens=["2", "+", "2#2", "=", "4"],
        confidence_steps=[
            {"2": 0.95, "+": 0.91, "2#2": 0.55},
            {"2#2": 0.93, "=": 0.94},
            {"4": 0.97},
        ],
        window_size=3,
        threshold=0.9,
    )

    assert trace.final_tokens == ["Solve:", "2", "+", "2", "=", "4"]
    assert [step.committed for step in trace.steps] == [["2", "+"], ["2", "="], ["4"]]
    assert [step.active_window for step in trace.steps] == [
        ["2", "+", "2#2"],
        ["2#2", "=", "4"],
        ["4"],
    ]
    assert max(len(step.active_window) for step in trace.steps) <= 3


def test_streaming_decode_stops_when_left_edge_token_is_below_threshold():
    """Would fail if the simulator committed high-confidence non-prefix tokens out of order."""
    trace = simulate_streaming_decode(
        prompt_tokens=[],
        planned_tokens=["A", "B", "C"],
        confidence_steps=[{"A": 0.2, "B": 0.99, "C": 0.99}],
        window_size=3,
        threshold=0.9,
    )

    assert trace.final_tokens == []
    assert trace.steps[0].committed == []
    assert trace.steps[0].active_window == ["A", "B", "C"]
