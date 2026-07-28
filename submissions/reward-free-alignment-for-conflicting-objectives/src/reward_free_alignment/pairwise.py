from dataclasses import dataclass
import math
from typing import Sequence
import torch
from torch import Tensor


@dataclass(frozen=True)
class PairwiseBatch:
    chosen_logp: Tensor
    rejected_logp: Tensor
    reference_chosen_logp: Tensor
    reference_rejected_logp: Tensor


def validate_positive_finite(name: str, value: float) -> float:
    if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be positive and finite, got {value}")
    return float(value)


def validate_equal_shapes(batch: PairwiseBatch) -> None:
    shapes = (
        batch.chosen_logp.shape,
        batch.rejected_logp.shape,
        batch.reference_chosen_logp.shape,
        batch.reference_rejected_logp.shape,
    )
    if not all(s == shapes[0] for s in shapes):
        raise ValueError(f"All tensors in PairwiseBatch must have equal shapes, got: {shapes}")


def pairwise_logistic_loss(batch: PairwiseBatch, beta: float) -> Tensor:
    beta_val = validate_positive_finite("beta", beta)
    validate_equal_shapes(batch)
    policy_gap = batch.chosen_logp - batch.rejected_logp
    reference_gap = batch.reference_chosen_logp - batch.reference_rejected_logp
    return -torch.nn.functional.logsigmoid(beta_val * (policy_gap - reference_gap)).mean()


def objective_losses(batches: Sequence[PairwiseBatch], beta: float) -> Tensor:
    if not batches:
        raise ValueError("batches sequence cannot be empty")
    losses = [pairwise_logistic_loss(batch, beta) for batch in batches]
    return torch.stack(losses)


def objective_gradients(
    losses: Tensor, parameters: Sequence[Tensor]
) -> tuple[Tensor, ...]:
    if losses.ndim != 1:
        raise ValueError(f"losses must be a 1D tensor, got shape {losses.shape}")
    param_list = list(parameters)
    if not param_list:
        raise ValueError("parameters sequence cannot be empty")

    grads: list[Tensor] = []
    for i in range(losses.shape[0]):
        loss_i = losses[i]
        g_tuple = torch.autograd.grad(loss_i, param_list, retain_graph=True, allow_unused=True)
        g_parts: list[Tensor] = []
        for p, g in zip(param_list, g_tuple):
            if g is None:
                g_parts.append(torch.zeros_like(p).flatten())
            else:
                g_parts.append(g.flatten())
        flat_g = torch.cat(g_parts)
        if not torch.isfinite(flat_g).all():
            raise ValueError(f"Non-finite gradient encountered for objective {i}")
        grads.append(flat_g)
    return tuple(grads)
