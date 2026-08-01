"""Deterministic CPU Pareto-frontier proxy for RACO's empirical claims.

The paper's TL;DR and BeaverTails experiments (claims 3-5) require LLM
fine-tuning that is out of scope for CPU-only reproduction. This module
executes the optimization problem the paper defines - objective-specific
pairwise preference losses combined through CAGrad-Clip - on a synthetic
two-objective preference task with conflicting ground-truth reward
directions, using a small nonconvex tanh scorer so the objective gradients
genuinely conflict along training (the regime Theorem 3.1 addresses), and
compares the resulting Pareto frontier against the linear-scalarization
baseline. All computation is seeded, full-batch, single-threaded float32 on
CPU, so every reported number is byte-stable.
"""

from dataclasses import dataclass
from typing import Sequence

import torch
from torch import Tensor

from reward_free_alignment.cagrad_clip import cagrad_clip
from reward_free_alignment.pairwise import PairwiseBatch, pairwise_logistic_loss

SEED = 20260801
FEATURE_DIM = 6
HIDDEN_DIM = 8
PARAM_DIM = HIDDEN_DIM * FEATURE_DIM + HIDDEN_DIM
TRAIN_PAIRS = 400
VAL_PAIRS = 200
CONFLICT_COS = -0.3
BETA = 1.0
LEARNING_RATE = 0.1
TRAIN_STEPS = 250
INIT_SCALE = 0.3
WEIGHT_GRID = (0.1, 0.3, 0.5, 0.7, 0.9)
ABLATION_RADII = (0.0, 0.25, 0.5, 0.75, 0.9)
DEFAULT_RADIUS = 0.5


@dataclass(frozen=True)
class ObjectiveData:
    """Train/validation preference pairs labelled by one reward direction."""

    train_chosen: Tensor
    train_rejected: Tensor
    val_chosen: Tensor
    val_rejected: Tensor


def make_reward_directions() -> tuple[Tensor, Tensor]:
    """Two unit reward directions with cosine similarity CONFLICT_COS."""
    r1 = torch.zeros(FEATURE_DIM)
    r1[0] = 1.0
    r2 = torch.zeros(FEATURE_DIM)
    r2[0] = CONFLICT_COS
    r2[1] = float(torch.sqrt(torch.tensor(1.0 - CONFLICT_COS**2)))
    return r1, r2


def make_objective_data(reward: Tensor, generator: torch.Generator) -> ObjectiveData:
    """Sample candidate pairs and label the winner by the reward direction."""

    def sample(count: int) -> tuple[Tensor, Tensor]:
        a = torch.randn(count, FEATURE_DIM, generator=generator)
        b = torch.randn(count, FEATURE_DIM, generator=generator)
        prefer_a = (a - b) @ reward > 0
        chosen = torch.where(prefer_a.unsqueeze(1), a, b)
        rejected = torch.where(prefer_a.unsqueeze(1), b, a)
        return chosen, rejected

    train_chosen, train_rejected = sample(TRAIN_PAIRS)
    val_chosen, val_rejected = sample(VAL_PAIRS)
    return ObjectiveData(train_chosen, train_rejected, val_chosen, val_rejected)


def score(features: Tensor, theta: Tensor) -> Tensor:
    """Nonconvex scorer: v . tanh(W x) with (W, v) packed into flat theta."""
    weight = theta[: HIDDEN_DIM * FEATURE_DIM].reshape(HIDDEN_DIM, FEATURE_DIM)
    head = theta[HIDDEN_DIM * FEATURE_DIM :]
    return torch.tanh(features @ weight.T) @ head


def objective_loss(data: ObjectiveData, theta: Tensor) -> Tensor:
    """Objective-specific pairwise logistic loss under the current scorer."""
    zeros = torch.zeros(data.train_chosen.shape[0])
    batch = PairwiseBatch(
        chosen_logp=score(data.train_chosen, theta),
        rejected_logp=score(data.train_rejected, theta),
        reference_chosen_logp=zeros,
        reference_rejected_logp=zeros,
    )
    return pairwise_logistic_loss(batch, beta=BETA)


def validation_accuracy(data: ObjectiveData, theta: Tensor) -> float:
    margins = score(data.val_chosen, theta) - score(data.val_rejected, theta)
    return round((margins > 0).float().mean().item(), 6)


def train_policy(
    objectives: Sequence[ObjectiveData],
    initial_theta: Tensor,
    weights: Tensor,
    clip_radius: float,
    use_cagrad: bool,
) -> tuple[Tensor, list[float]]:
    """Train the scorer with RACO (CAGrad-Clip) or linear scalarization."""
    theta = initial_theta.clone().requires_grad_(True)
    final_losses = [0.0, 0.0]
    for _ in range(TRAIN_STEPS):
        losses = [objective_loss(data, theta) for data in objectives]
        grads = [
            torch.autograd.grad(loss, [theta], retain_graph=True)[0]
            for loss in losses
        ]
        if use_cagrad:
            update = cagrad_clip(grads, weights, clip_radius).gradient
        else:
            update = weights[0] * grads[0] + weights[1] * grads[1]
        with torch.no_grad():
            theta -= LEARNING_RATE * update
        theta.requires_grad_(True)
        final_losses = [round(loss.item(), 6) for loss in losses]
    return theta.detach(), final_losses


def hypervolume(points: Sequence[tuple[float, float]]) -> float:
    """2D hypervolume of the accuracy region dominated by points, origin ref."""
    frontier: list[tuple[float, float]] = []
    for x, y in sorted(points, key=lambda p: (-p[0], -p[1])):
        if not frontier or y > frontier[-1][1]:
            frontier.append((x, y))
    volume = 0.0
    previous_y = 0.0
    for x, y in frontier:
        volume += x * (y - previous_y)
        previous_y = y
    return round(volume, 6)


def run_pareto_proxy() -> dict:
    """Execute the frontier comparison and clip-radius ablation."""
    torch.set_num_threads(1)
    generator = torch.Generator().manual_seed(SEED)
    r1, r2 = make_reward_directions()
    objectives = (
        make_objective_data(r1, generator),
        make_objective_data(r2, generator),
    )
    initial_theta = INIT_SCALE * torch.randn(PARAM_DIM, generator=generator)

    frontier_rows = []
    raco_points: list[tuple[float, float]] = []
    baseline_points: list[tuple[float, float]] = []
    dominated = 0
    for w1 in WEIGHT_GRID:
        weights = torch.tensor([w1, 1.0 - w1])
        raco_theta, raco_losses = train_policy(
            objectives, initial_theta, weights, DEFAULT_RADIUS, use_cagrad=True
        )
        base_theta, base_losses = train_policy(
            objectives, initial_theta, weights, DEFAULT_RADIUS, use_cagrad=False
        )
        raco_acc = (
            validation_accuracy(objectives[0], raco_theta),
            validation_accuracy(objectives[1], raco_theta),
        )
        base_acc = (
            validation_accuracy(objectives[0], base_theta),
            validation_accuracy(objectives[1], base_theta),
        )
        raco_points.append(raco_acc)
        baseline_points.append(base_acc)
        if (
            raco_acc[0] >= base_acc[0]
            and raco_acc[1] >= base_acc[1]
            and raco_acc != base_acc
        ):
            dominated += 1
        frontier_rows.append(
            {
                "weight_objective_1": w1,
                "raco_val_acc": list(raco_acc),
                "baseline_val_acc": list(base_acc),
                "raco_final_losses": raco_losses,
                "baseline_final_losses": base_losses,
            }
        )

    ablation_rows = []
    balanced = torch.tensor([0.5, 0.5])
    for radius in ABLATION_RADII:
        theta, losses = train_policy(
            objectives, initial_theta, balanced, radius, use_cagrad=True
        )
        acc = (
            validation_accuracy(objectives[0], theta),
            validation_accuracy(objectives[1], theta),
        )
        ablation_rows.append(
            {
                "clip_radius": radius,
                "val_acc": list(acc),
                "min_val_acc": round(min(acc), 6),
                "final_losses": losses,
            }
        )

    return {
        "config": {
            "seed": SEED,
            "feature_dim": FEATURE_DIM,
            "hidden_dim": HIDDEN_DIM,
            "param_dim": PARAM_DIM,
            "train_pairs": TRAIN_PAIRS,
            "val_pairs": VAL_PAIRS,
            "conflict_cosine": CONFLICT_COS,
            "beta": BETA,
            "learning_rate": LEARNING_RATE,
            "train_steps": TRAIN_STEPS,
            "init_scale": INIT_SCALE,
            "default_clip_radius": DEFAULT_RADIUS,
            "hypervolume_reference": [0.0, 0.0],
        },
        "frontier": frontier_rows,
        "raco_hypervolume": hypervolume(raco_points),
        "baseline_hypervolume": hypervolume(baseline_points),
        "raco_dominates_baseline_count": dominated,
        "weight_settings": len(WEIGHT_GRID),
        "ablation": ablation_rows,
    }
