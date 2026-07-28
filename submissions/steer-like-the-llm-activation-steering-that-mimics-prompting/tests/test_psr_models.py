import pytest
import torch
from steer_like_llm.psr_models import (
    PSRModel,
    train_psr_mse,
    train_psr_log_likelihood,
)

def test_psr_model_forward():
    model = PSRModel(hidden_dim=32, direction_dim=32)
    h_base = torch.randn(4, 10, 32)
    alpha, pred_v = model(h_base)
    assert alpha.shape == (4, 10, 1)
    assert pred_v.shape == (4, 10, 32)

def test_train_psr_mse():
    torch.manual_seed(42)
    model = PSRModel(hidden_dim=16, direction_dim=16)
    h_base = torch.randn(2, 5, 16)
    target_v = torch.randn(2, 5, 16)
    res = train_psr_mse(model, h_base, target_v, epochs=20, lr=0.01)
    assert res["converged"] is True
    assert res["final_loss"] < res["initial_loss"]

def test_train_psr_log_likelihood():
    torch.manual_seed(42)
    model = PSRModel(hidden_dim=16, direction_dim=16)
    h_base = torch.randn(2, 5, 16)
    target_v = torch.randn(2, 5, 16)
    res = train_psr_log_likelihood(model, h_base, target_v, epochs=20, lr=0.01)
    assert res["converged"] is True
    assert res["final_nll"] < res["initial_nll"]
