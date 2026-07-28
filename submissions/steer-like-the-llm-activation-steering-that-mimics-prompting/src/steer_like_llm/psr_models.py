"""Prompt Steering Regression (PSR) Models & Loss Objectives (Sections 3.4 & 3.5)."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Any

class PSRModel(nn.Module):
    """
    Prompt Steering Regression (PSR) Model.
    Predicts token-specific steering coefficient alpha_t given activation h_t,
    and applies it to candidate direction u to approximate intervention v_t.
    """
    def __init__(self, hidden_dim: int, direction_dim: int, hidden_layer_size: int = 64):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.direction_dim = direction_dim
        
        # Token-specific coefficient estimator network
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_layer_size),
            nn.ReLU(),
            nn.Linear(hidden_layer_size, 1),
        )
        
        # Trainable/fixed steering direction vector u
        self.steering_direction = nn.Parameter(torch.randn(direction_dim))
        
    def forward(self, h_t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        h_t: (batch, seq_len, hidden_dim)
        Returns:
            alpha_t: (batch, seq_len, 1) token-specific coefficients
            predicted_v_t: (batch, seq_len, direction_dim) predicted interventions
        """
        alpha_t = self.net(h_t)  # (batch, seq_len, 1)
        
        # Normalize steering direction vector
        u_norm = F.normalize(self.steering_direction, p=2, dim=-1)
        
        # Predicted intervention v_t = alpha_t * u
        predicted_v_t = alpha_t * u_norm.unsqueeze(0).unsqueeze(0)
        return alpha_t, predicted_v_t

def train_psr_mse(
    model: PSRModel,
    h_base: torch.Tensor,
    target_interventions: torch.Tensor,
    epochs: int = 50,
    lr: float = 0.01,
) -> Dict[str, Any]:
    """Train PSR model using Mean Squared Error (MSE) loss objective."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    losses = []
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        alpha_t, pred_v_t = model(h_base)
        loss = F.mse_loss(pred_v_t, target_interventions)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        
    return {
        "final_loss": losses[-1],
        "initial_loss": losses[0],
        "converged": losses[-1] < losses[0],
        "loss_history": losses,
    }

def train_psr_log_likelihood(
    model: PSRModel,
    h_base: torch.Tensor,
    target_interventions: torch.Tensor,
    noise_scale: float = 0.1,
    epochs: int = 50,
    lr: float = 0.01,
) -> Dict[str, Any]:
    """Train PSR model using Log-Likelihood loss objective (Section 3.5)."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    losses = []
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        alpha_t, pred_v_t = model(h_base)
        # Gaussian log-likelihood loss ~ 0.5 * ||pred - target||^2 / sigma^2
        diff = pred_v_t - target_interventions
        nll = 0.5 * torch.mean(diff ** 2) / (noise_scale ** 2)
        nll.backward()
        optimizer.step()
        losses.append(nll.item())
        
    return {
        "final_nll": losses[-1],
        "initial_nll": losses[0],
        "converged": losses[-1] < losses[0],
        "loss_history": losses,
    }
