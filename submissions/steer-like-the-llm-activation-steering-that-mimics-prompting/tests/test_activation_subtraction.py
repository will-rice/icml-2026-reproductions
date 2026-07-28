import pytest
import torch
from steer_like_llm.activation_subtraction import (
    compute_intervention_vectors,
    analyze_token_dependent_strengths,
)

def test_compute_intervention_vectors():
    prompt_h = torch.tensor([[[2.0, 3.0], [4.0, 5.0]]])
    base_h = torch.tensor([[[1.0, 1.0], [2.0, 1.0]]])
    
    interventions = compute_intervention_vectors(prompt_h, base_h)
    expected = torch.tensor([[[1.0, 2.0], [2.0, 4.0]]])
    assert torch.allclose(interventions, expected)

def test_shape_mismatch():
    prompt_h = torch.randn(2, 5, 10)
    base_h = torch.randn(2, 6, 10)
    with pytest.raises(ValueError, match="Shape mismatch"):
        compute_intervention_vectors(prompt_h, base_h)

def test_analyze_token_dependent_strengths():
    interventions = torch.randn(4, 10, 16)
    analysis = analyze_token_dependent_strengths(interventions)
    assert "mean_norm" in analysis
    assert "std_norm" in analysis
    assert analysis["is_token_dependent"] is True
