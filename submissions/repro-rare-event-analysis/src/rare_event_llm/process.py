from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from math import log

import numpy as np


@dataclass(frozen=True)
class SequenceRecord:
    text: str
    tokens: tuple[str, ...]
    log_probability: float
    observable: float


@dataclass(frozen=True)
class TextProcess:
    tokens: tuple[str, ...]
    initial: tuple[float, ...]
    transition: tuple[tuple[float, ...], ...]
    length: int

    @classmethod
    def default(cls, length: int = 6) -> "TextProcess":
        return cls(
            tokens=("cat", "runs", "quietly", "!"),
            initial=(0.34, 0.26, 0.30, 0.10),
            transition=(
                (0.18, 0.40, 0.30, 0.12),
                (0.32, 0.12, 0.42, 0.14),
                (0.38, 0.30, 0.12, 0.20),
                (0.45, 0.25, 0.20, 0.10),
            ),
            length=length,
        )


def enumerate_sequences(process: TextProcess) -> list[SequenceRecord]:
    records = []
    token_count = len(process.tokens)
    for indices in product(range(token_count), repeat=process.length):
        probability = process.initial[indices[0]]
        for left, right in zip(indices, indices[1:]):
            probability *= process.transition[left][right]
        tokens = tuple(process.tokens[index] for index in indices)
        records.append(
            SequenceRecord(
                text=" ".join(tokens),
                tokens=tokens,
                log_probability=log(probability),
                observable=_readability_observable(tokens),
            )
        )
    return records


def direct_sample(
    process: TextProcess, sample_count: int, seed: int
) -> list[SequenceRecord]:
    records = enumerate_sequences(process)
    probabilities = np.array([np.exp(record.log_probability) for record in records])
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(records), size=sample_count, p=probabilities)
    return [records[int(index)] for index in indices]


def _readability_observable(tokens: tuple[str, ...]) -> float:
    vowel_tokens = sum(token in {"cat", "quietly"} for token in tokens)
    punctuation = tokens.count("!")
    repeats = sum(left == right for left, right in zip(tokens, tokens[1:]))
    alternations = sum(left != right for left, right in zip(tokens, tokens[1:]))
    return (
        0.9 * vowel_tokens / len(tokens)
        + 0.35 * punctuation
        + 0.08 * alternations
        - 0.18 * repeats
    )
