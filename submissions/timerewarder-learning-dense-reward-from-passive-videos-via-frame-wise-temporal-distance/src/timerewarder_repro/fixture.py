"""Deterministic, diagnostic-only passive-video learning fixture."""

from dataclasses import dataclass

import numpy as np
import torch
from torch.nn import functional as functional

from timerewarder_repro.method import (
    adjacent_rewards,
    logits_to_scalar,
    scalar_to_two_hot,
)

SEED = 20260725
TRAJECTORY_LENGTH = 9
FIXTURE_SPECIFICATION = {
    "seed": SEED,
    "dtype": "float64",
    "cpu_threads": 1,
    "trajectory_length": TRAJECTORY_LENGTH,
    "train_trajectories": 32,
    "test_trajectories": 8,
    "train_ordered_pairs": 2304,
    "test_ordered_pairs": 576,
    "iterations": 250,
    "learning_rate": 0.2,
    "initial_weights": "zeros",
    "data_order": "lexicographic",
}


@dataclass(frozen=True)
class PassivePairs:
    features: torch.Tensor
    targets: torch.Tensor
    temporal_distances: np.ndarray
    trajectories: np.ndarray


def run_fixture() -> dict[str, object]:
    """Run the fixed CPU diagnostic without using actions or external inputs."""
    torch.set_num_threads(1)
    torch.manual_seed(SEED)
    train = _passive_pairs(32, nuisance_offset=0.0)
    test = _passive_pairs(8, nuisance_offset=2.0)
    positive = _fit_and_measure(train, test, train.targets)
    generator = torch.Generator().manual_seed(SEED)
    permutation = torch.randperm(train.targets.shape[0], generator=generator)
    control = _fit_and_measure(train, test, train.targets[permutation])
    return {
        "specification": FIXTURE_SPECIFICATION,
        "diagnostic_only": True,
        "acceptance_threshold": None,
        "measurements": {"positive": positive, "permuted_label_control": control},
    }


def _passive_pairs(trajectory_count: int, nuisance_offset: float) -> PassivePairs:
    rows: list[np.ndarray] = []
    distances: list[float] = []
    trajectory_ids: list[int] = []
    progress = np.arange(TRAJECTORY_LENGTH, dtype=np.float64) / (TRAJECTORY_LENGTH - 1)
    for trajectory in range(trajectory_count):
        nuisance = np.full(
            TRAJECTORY_LENGTH,
            nuisance_offset + trajectory / max(trajectory_count - 1, 1),
            dtype=np.float64,
        )
        frames = np.stack((progress, nuisance), axis=1)
        for start in range(TRAJECTORY_LENGTH):
            for end in range(TRAJECTORY_LENGTH):
                if start == end:
                    continue
                rows.append(np.concatenate((frames[start], frames[end])))
                distances.append((end - start) / (TRAJECTORY_LENGTH - 1))
                trajectory_ids.append(trajectory)
    temporal_distances = np.asarray(distances, dtype=np.float64)
    return PassivePairs(
        features=torch.from_numpy(np.asarray(rows, dtype=np.float64)),
        targets=torch.from_numpy(scalar_to_two_hot(temporal_distances)),
        temporal_distances=temporal_distances,
        trajectories=np.asarray(trajectory_ids, dtype=np.int64),
    )


def _fit_and_measure(
    train: PassivePairs, test: PassivePairs, train_targets: torch.Tensor
) -> dict[str, float | int]:
    head = torch.nn.Linear(4, 20, bias=True, dtype=torch.float64)
    with torch.no_grad():
        head.weight.zero_()
        head.bias.zero_()
    optimizer = torch.optim.SGD(head.parameters(), lr=0.2)
    for _ in range(250):
        optimizer.zero_grad()
        loss = functional.cross_entropy(head(train.features), train_targets)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        logits = head(test.features).numpy()
    predictions = logits_to_scalar(logits).ravel()
    mae = np.abs(predictions - test.temporal_distances).mean(dtype=np.float64)
    signs = np.sign(predictions) == np.sign(test.temporal_distances)
    sign_accuracy = signs.mean(dtype=np.float64)
    adjacent_density = _adjacent_reward_density(head, test)
    return {
        "decoded_temporal_distance_mae": _rounded(mae),
        "nonzero_adjacent_reward_density": _rounded(adjacent_density),
        "forward_reverse_sign_accuracy": _rounded(sign_accuracy),
        "train_ordered_pairs": train.targets.shape[0],
        "test_ordered_pairs": test.targets.shape[0],
    }


def _adjacent_reward_density(head: torch.nn.Linear, test: PassivePairs) -> float:
    reward_values: list[float] = []
    for trajectory in np.unique(test.trajectories):
        nuisance = 2.0 + trajectory / 7
        progress = np.arange(TRAJECTORY_LENGTH, dtype=np.float64) / (
            TRAJECTORY_LENGTH - 1
        )
        frames = np.stack(
            (progress, np.full(TRAJECTORY_LENGTH, nuisance, dtype=np.float64)), axis=1
        )
        previous = np.concatenate((frames[:1], frames[:-1]), axis=0)
        forward_pairs = np.concatenate((previous, frames), axis=1)
        reverse_pairs = np.concatenate((frames, previous), axis=1)
        with torch.no_grad():
            forward_logits = head(torch.from_numpy(forward_pairs)).numpy()
            reverse_logits = head(torch.from_numpy(reverse_pairs)).numpy()
        rewards = adjacent_rewards(
            logits_to_scalar(forward_logits).ravel(),
            logits_to_scalar(reverse_logits).ravel(),
        )
        reward_values.extend(rewards)
    values = np.asarray(reward_values, dtype=np.float64)
    return float(np.count_nonzero(values) / values.size)


def _rounded(value: float | np.floating) -> float:
    return round(float(value), 12)
