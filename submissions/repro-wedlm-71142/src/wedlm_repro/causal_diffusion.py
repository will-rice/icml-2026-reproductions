"""Small executable checks for WeDLM causal diffusion scheduling."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class ReorderResult:
    physical_tokens: list[str]
    logical_positions: list[int]
    observed_count: int
    physical_index_by_logical: dict[int, int]


@dataclass(frozen=True)
class StreamingStep:
    active_window: list[str]
    committed: list[str]
    prefix_after: list[str]


@dataclass(frozen=True)
class StreamingTrace:
    final_tokens: list[str]
    steps: list[StreamingStep]


def topological_reorder(tokens: Sequence[str], observed: Sequence[int]) -> ReorderResult:
    """Move observed logical positions to the physical prefix."""
    observed_set = set(observed)
    if any(pos < 0 or pos >= len(tokens) for pos in observed_set):
        raise ValueError("observed position out of range")

    observed_positions = [pos for pos in range(len(tokens)) if pos in observed_set]
    masked_positions = [pos for pos in range(len(tokens)) if pos not in observed_set]
    logical_positions = observed_positions + masked_positions
    physical_tokens = [tokens[pos] for pos in logical_positions]
    physical_index_by_logical = {
        logical_pos: physical_pos for physical_pos, logical_pos in enumerate(logical_positions)
    }
    return ReorderResult(
        physical_tokens=physical_tokens,
        logical_positions=logical_positions,
        observed_count=len(observed_positions),
        physical_index_by_logical=physical_index_by_logical,
    )


def causal_reachability(reorder: ReorderResult) -> np.ndarray:
    """Return a strict causal-attention reachability matrix over physical order."""
    size = len(reorder.physical_tokens)
    return np.tril(np.ones((size, size), dtype=bool))


def _display_token(token_id: str) -> str:
    return token_id.split("#", 1)[0]


def simulate_streaming_decode(
    *,
    prompt_tokens: Sequence[str],
    planned_tokens: Sequence[str],
    confidence_steps: Sequence[Mapping[str, float]],
    window_size: int,
    threshold: float,
) -> StreamingTrace:
    """Simulate left-to-right streaming commits with a fixed active window."""
    if window_size <= 0:
        raise ValueError("window_size must be positive")

    prefix = list(prompt_tokens)
    pending = list(planned_tokens)
    steps: list[StreamingStep] = []

    for confidences in confidence_steps:
        if not pending:
            break
        active = pending[:window_size]
        committed_ids: list[str] = []
        while pending and len(committed_ids) < window_size:
            token_id = pending[0]
            if confidences.get(token_id, 0.0) < threshold:
                break
            committed_ids.append(token_id)
            prefix.append(_display_token(token_id))
            pending.pop(0)
        steps.append(
            StreamingStep(
                active_window=active,
                committed=[_display_token(token_id) for token_id in committed_ids],
                prefix_after=list(prefix),
            )
        )

    return StreamingTrace(final_tokens=prefix, steps=steps)
