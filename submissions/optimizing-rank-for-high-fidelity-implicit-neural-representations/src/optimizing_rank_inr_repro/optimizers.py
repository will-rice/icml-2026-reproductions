"""Muon and Adam optimizers for INR rank optimization."""

import torch
from torch.optim import Optimizer


def zeropower_via_newtonschulz5(G: torch.Tensor, steps: int = 5, eps: float = 1e-7) -> torch.Tensor:
    """Newton-Schulz iteration to compute the nearest orthogonal matrix / polar factor of G."""
    if G.ndim != 2 or min(G.size(0), G.size(1)) <= 1:
        return G
    X = G.to(torch.float32)
    X = X / (X.norm() + eps)
    if G.size(0) > G.size(1):
        X = X.T

    a, b, c = (3.4445, -4.7750, 2.0315)
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X

    if G.size(0) > G.size(1):
        X = X.T
    return X.to(G.dtype)


class Muon(Optimizer):
    """Muon optimizer: Applies near-orthogonal updates to 2D weight matrices to regulate rank."""

    def __init__(self, params, lr: float = 0.005, momentum: float = 0.95, nesterov: bool = True, ns_steps: int = 5):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']
            nesterov = group['nesterov']
            ns_steps = group['ns_steps']

            for p in group['params']:
                if p.grad is None:
                    continue
                g = p.grad.data
                state = self.state[p]

                if 'momentum_buffer' not in state:
                    state['momentum_buffer'] = torch.zeros_like(p.data)
                buf = state['momentum_buffer']
                buf.mul_(momentum).add_(g)

                if nesterov:
                    update = g.add(buf, alpha=momentum)
                else:
                    update = buf

                if p.ndim == 2 and min(p.size(0), p.size(1)) > 1:
                    ortho_update = zeropower_via_newtonschulz5(update, steps=ns_steps)
                    scale = (max(p.size(0), p.size(1))) ** 0.5
                    p.data.add_(ortho_update, alpha=-lr * scale * 0.1)
                else:
                    p.data.add_(update, alpha=-lr)

        return loss
