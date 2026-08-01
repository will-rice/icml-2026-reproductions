"""
Core LiME Layer implementation (Shared PEFT Adapter + Expert Modulation Vectors).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class LiMELayer(nn.Module):
    """
    LiME Layer: Shares a single LoRA adapter (A, B) across all experts
    and applies lightweight expert-specific modulation vectors (m_e).
    """

    def __init__(self, in_features: int, out_features: int, r: int = 8, num_experts: int = 4, alpha: float = 16.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.r = r
        self.num_experts = num_experts
        self.scaling = alpha / r

        # Single shared PEFT adapter
        self.lora_A = nn.Parameter(torch.randn(r, in_features) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(out_features, r))

        # Lightweight expert-specific modulation vectors (one vector m_e in R^r per expert)
        self.expert_modulations = nn.Parameter(torch.ones(num_experts, r))

    def forward(self, x: torch.Tensor, expert_weights: torch.Tensor) -> torch.Tensor:
        """
        x: (batch_size, seq_len, in_features) or (batch_size, in_features)
        expert_weights: (batch_size, seq_len, num_experts) or (batch_size, num_experts)
        """
        # 1. Project input through shared LoRA A -> (..., r)
        h = F.linear(x, self.lora_A)

        # 2. Modulate per expert & aggregate with routing weights
        # expert_weights shape: (..., num_experts)
        # expert_modulations shape: (num_experts, r)
        # Effective rank modulation: sum_e (w_e * m_e) -> (..., r)
        effective_modulation = torch.matmul(expert_weights, self.expert_modulations)
        h_modulated = h * effective_modulation

        # 3. Project through shared LoRA B -> (..., out_features)
        out = F.linear(h_modulated, self.lora_B) * self.scaling
        return out


class MoELoRABaseline(nn.Module):
    """
    Standard MoE-LoRA baseline replicating a full LoRA adapter per expert.
    """

    def __init__(self, in_features: int, out_features: int, r: int = 8, num_experts: int = 4, alpha: float = 16.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.r = r
        self.num_experts = num_experts
        self.scaling = alpha / r

        self.lora_As = nn.Parameter(torch.randn(num_experts, r, in_features) * 0.01)
        self.lora_Bs = nn.Parameter(torch.zeros(num_experts, out_features, r))

    def forward(self, x: torch.Tensor, expert_weights: torch.Tensor) -> torch.Tensor:
        batch_shape = x.shape[:-1]
        x_flat = x.view(-1, self.in_features)
        w_flat = expert_weights.view(-1, self.num_experts)

        total_out = torch.zeros(x_flat.size(0), self.out_features, device=x.device, dtype=x.dtype)
        for e in range(self.num_experts):
            h_e = F.linear(x_flat, self.lora_As[e])
            out_e = F.linear(h_e, self.lora_Bs[e]) * self.scaling
            total_out += w_flat[:, e:e+1] * out_e

        return total_out.view(*batch_shape, self.out_features)
