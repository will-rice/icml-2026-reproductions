import pytest
import torch
from mechanistic_data_attribution_repro.patterns import classify_sample_pattern, analyze_pattern_attribution

def test_classify_sample_pattern():
    repetitive_seq = torch.tensor([1, 2, 3, 1, 2, 3])
    random_seq = torch.tensor([10, 42, 99, 3, 8, 17])
    
    assert classify_sample_pattern(repetitive_seq) == "repetitive_structural"
    assert classify_sample_pattern(random_seq) == "unstructured_random"

def test_analyze_pattern_attribution():
    samples = [
        torch.tensor([1, 2, 3, 1, 2, 3]),
        torch.tensor([10, 42, 99, 3, 8, 17]),
        torch.tensor([5, 6, 5, 6, 5, 6]),
    ]
    scores = [0.85, 0.12, 0.91]
    res = analyze_pattern_attribution(samples, scores)
    assert "mean_repetitive_score" in res
    assert "mean_unstructured_score" in res
    assert res["mean_repetitive_score"] > res["mean_unstructured_score"]
