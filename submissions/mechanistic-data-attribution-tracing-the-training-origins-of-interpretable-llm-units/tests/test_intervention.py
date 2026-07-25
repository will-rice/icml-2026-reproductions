import pytest
import torch
from mechanistic_data_attribution_repro.intervention import evaluate_causal_interventions

def test_evaluate_causal_interventions():
    samples = [
        torch.tensor([1, 2, 3, 1, 2, 3]),
        torch.tensor([10, 42, 99, 3, 8, 17]),
        torch.tensor([5, 6, 5, 6, 5, 6]),
        torch.tensor([7, 8, 9, 7, 8, 9]),
    ]
    scores = [0.80, 0.10, 0.90, 0.85]

    results = evaluate_causal_interventions(samples, scores, prune_ratio=0.5, seed=42)
    assert "targeted_prune_probe_drop" in results
    assert "random_prune_probe_drop" in results
    assert results["targeted_prune_probe_drop"] > results["random_prune_probe_drop"]
