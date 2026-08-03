import numpy as np

from wedlm_repro.causal_diffusion import causal_reachability, topological_reorder


def test_topological_reorder_moves_observed_tokens_to_prefix_without_losing_logical_positions():
    """Would fail if masked tokens stayed interleaved with observed tokens."""
    result = topological_reorder(tokens=["A", "<mask>", "B", "<mask>", "C"], observed=[0, 2, 4])

    assert result.physical_tokens == ["A", "B", "C", "<mask>", "<mask>"]
    assert result.logical_positions == [0, 2, 4, 1, 3]
    assert result.observed_count == 3
    assert result.physical_index_by_logical == {0: 0, 1: 3, 2: 1, 3: 4, 4: 2}


def test_strict_causal_mask_lets_prediction_positions_attend_to_all_observed_prefix_tokens():
    """Would fail if prediction queries could not condition on known future tokens."""
    result = topological_reorder(tokens=["The", "<mask>", "sky", "<mask>"], observed=[0, 2])
    reachability = causal_reachability(result)

    first_mask_physical = result.physical_index_by_logical[1]
    second_mask_physical = result.physical_index_by_logical[3]

    assert reachability[first_mask_physical].tolist() == [True, True, True, False]
    assert reachability[second_mask_physical].tolist() == [True, True, True, True]
    assert not np.triu(reachability, k=1).any()
