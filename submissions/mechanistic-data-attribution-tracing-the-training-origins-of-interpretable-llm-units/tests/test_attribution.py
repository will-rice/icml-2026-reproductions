import pytest
import torch
from mechanistic_data_attribution_repro.attribution import MechanisticAttribution

def test_mechanistic_attribution_calculation():
    # Setup dummy samples
    samples = [
        torch.tensor([1, 2, 3, 1, 2, 3]), # Repetitive pattern
        torch.tensor([5, 9, 2, 8, 4, 7]), # Random pattern
        torch.tensor([10, 20, 10, 20, 10, 20]), # Repetitive pattern
    ]
    calc = MechanisticAttribution(seed=42)
    scores = calc.compute_attribution_scores(samples)
    assert len(scores) == len(samples)
    assert all(isinstance(s, float) for s in scores)
    # Repetitive patterns should score higher for induction head attribution
    assert scores[0] > scores[1]
    assert scores[2] > scores[1]
