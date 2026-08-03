from __future__ import annotations

from dataclasses import dataclass
from math import log

import numpy as np

from rare_event_llm.process import SequenceRecord


@dataclass(frozen=True)
class BiasedSample:
    beta: float
    records: tuple[SequenceRecord, ...]
    mean_observable: float
    log_partition: float
    acceptance_rate: float | None = None


def biased_probabilities(records: list[SequenceRecord], beta: float) -> tuple[np.ndarray, float]:
    log_weights = np.array(
        [record.log_probability + beta * record.observable for record in records],
        dtype=float,
    )
    log_partition = _logsumexp(log_weights)
    return np.exp(log_weights - log_partition), float(log_partition)


def biased_sample(
    records: list[SequenceRecord], beta: float, sample_count: int, seed: int
) -> BiasedSample:
    probabilities, log_partition = biased_probabilities(records, beta)
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(records), size=sample_count, p=probabilities)
    sampled = tuple(records[int(index)] for index in indices)
    return BiasedSample(
        beta=beta,
        records=sampled,
        mean_observable=float(np.mean([record.observable for record in sampled])),
        log_partition=log_partition,
    )


def transition_path_sample(
    records: list[SequenceRecord], beta_schedule: list[float], steps: int, seed: int
) -> list[BiasedSample]:
    rng = np.random.default_rng(seed)
    current = int(rng.integers(0, len(records)))
    output = []
    for beta in beta_schedule:
        accepted = 0
        trajectory = []
        for _ in range(steps):
            proposal = int(rng.integers(0, len(records)))
            log_ratio = (
                records[proposal].log_probability
                + beta * records[proposal].observable
                - records[current].log_probability
                - beta * records[current].observable
            )
            if log_ratio >= 0 or log(rng.random()) < log_ratio:
                current = proposal
                accepted += 1
            trajectory.append(records[current])
        _, log_partition = biased_probabilities(records, beta)
        output.append(
            BiasedSample(
                beta=beta,
                records=tuple(trajectory),
                mean_observable=float(np.mean([record.observable for record in trajectory])),
                log_partition=log_partition,
                acceptance_rate=accepted / steps,
            )
        )
    return output


def _logsumexp(values: np.ndarray) -> float:
    offset = float(np.max(values))
    return offset + float(np.log(np.sum(np.exp(values - offset))))
