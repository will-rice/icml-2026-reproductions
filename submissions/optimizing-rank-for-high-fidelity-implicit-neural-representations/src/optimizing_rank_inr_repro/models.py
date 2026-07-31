"""INR Model architectures and rank metrics."""

import math
import torch
import torch.nn as nn


def compute_stable_rank(weight: torch.Tensor) -> float:
    """Compute stable rank ||W||_F^2 / ||W||_2^2 for a 2D weight matrix."""
    if weight.ndim != 2:
        weight = weight.reshape(weight.size(0), -1)
    if weight.size(0) == 0 or weight.size(1) == 0:
        return 1.0
    fro_sq = torch.sum(weight ** 2).item()
    with torch.no_grad():
        u, s, vh = torch.linalg.svd(weight, full_matrices=False)
        sigma_max = s[0].item()
    if sigma_max <= 1e-12:
        return 1.0
    return float(fro_sq / (sigma_max ** 2))


def compute_psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Compute Peak Signal-to-Noise Ratio (PSNR) in dB."""
    mse = torch.mean((pred - target) ** 2).item()
    if mse <= 1e-12:
        return 100.0
    max_val = 1.0
    return float(10.0 * math.log10((max_val ** 2) / mse))


class SineLayer(nn.Module):
    """Siren sine activation layer."""

    def __init__(self, in_features: int, out_features: int, is_first: bool = False, omega_0: float = 30.0):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        self.linear = nn.Linear(in_features, out_features)
        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1.0 / self.linear.in_features, 1.0 / self.linear.in_features)
            else:
                bound = math.sqrt(6.0 / self.linear.in_features) / self.omega_0
                self.linear.weight.uniform_(-bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(self.omega_0 * self.linear(x))


class SirenINR(nn.Module):
    """Siren Implicit Neural Representation network."""

    def __init__(self, in_dim: int = 2, hidden_dim: int = 64, out_dim: int = 1, num_layers: int = 4, omega_0: float = 30.0):
        super().__init__()
        layers = [SineLayer(in_dim, hidden_dim, is_first=True, omega_0=omega_0)]
        for _ in range(num_layers - 2):
            layers.append(SineLayer(hidden_dim, hidden_dim, is_first=False, omega_0=omega_0))
        self.net = nn.Sequential(*layers)
        self.final_linear = nn.Linear(hidden_dim, out_dim)
        with torch.no_grad():
            bound = math.sqrt(6.0 / hidden_dim) / omega_0
            self.final_linear.weight.uniform_(-bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.net(x)
        return self.final_linear(feat)

    def average_stable_rank(self) -> float:
        ranks = []
        for name, param in self.named_parameters():
            if "weight" in name and param.ndim == 2:
                ranks.append(compute_stable_rank(param.data))
        return float(sum(ranks) / max(len(ranks), 1))


class VanillaMLP(nn.Module):
    """Standard ReLU MLP INR network."""

    def __init__(self, in_dim: int = 2, hidden_dim: int = 64, out_dim: int = 1, num_layers: int = 4):
        super().__init__()
        layers = []
        layers.append(nn.Linear(in_dim, hidden_dim))
        layers.append(nn.ReLU())
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_dim, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def average_stable_rank(self) -> float:
        ranks = []
        for name, param in self.named_parameters():
            if "weight" in name and param.ndim == 2:
                ranks.append(compute_stable_rank(param.data))
        return float(sum(ranks) / max(len(ranks), 1))
