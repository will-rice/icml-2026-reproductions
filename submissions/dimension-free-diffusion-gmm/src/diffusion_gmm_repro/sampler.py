"""Deterministic active-coordinate DDPM sampling for unit-covariance GMMs."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
from numpy.typing import NDArray

from .model import IsotropicGMM, score_at_time
from .schedule import DDPMSchedule

_BLOCK_SIZE = 1_024


def orthonormal_embedding(
    *, active_dimension: int, ambient_dimension: int
) -> NDArray[np.float64]:
    """Embed active coordinates in the leading ambient-coordinate subspace."""
    if active_dimension < 0:
        raise ValueError("active_dimension must be nonnegative")
    if ambient_dimension < active_dimension:
        raise ValueError("ambient_dimension must be at least active_dimension")
    embedding = np.zeros((ambient_dimension, active_dimension), dtype=np.float64)
    embedding[:active_dimension, :] = np.eye(active_dimension)
    return embedding


def constant_rms_profile(
    *, steps: int, rms: float, shape: str
) -> NDArray[np.float64]:
    """Return a nonnegative time profile with the requested RMS over 1..T."""
    if steps < 1:
        raise ValueError("steps must be positive")
    if not np.isfinite(rms) or rms < 0.0:
        raise ValueError("rms must be finite and nonnegative")

    times = np.arange(1, steps + 1, dtype=np.float64)
    if shape == "uniform":
        weights = np.ones(steps, dtype=np.float64)
    elif shape == "front-loaded":
        weights = times[::-1]
    elif shape == "back-loaded":
        weights = times
    else:
        raise ValueError("shape must be uniform, front-loaded, or back-loaded")

    profile = np.zeros(steps + 1, dtype=np.float64)
    profile[1:] = rms * weights / np.sqrt(np.mean(weights**2))
    return profile


def _block_indices(*, samples: int, blocks_per_batch: int) -> Iterator[range]:
    block_count = (samples + _BLOCK_SIZE - 1) // _BLOCK_SIZE
    for first_block in range(0, block_count, blocks_per_batch):
        yield range(first_block, min(first_block + blocks_per_batch, block_count))


def normal_blocks(
    *,
    seed: int,
    stream: int,
    samples: int,
    dimension: int,
    blocks_per_batch: int = 1,
) -> NDArray[np.float64]:
    """Draw normal samples from immutable 1,024-row blocks.

    Every block's generator is keyed only by ``(seed, stream, block_index)``,
    so changing the batching policy cannot change any random draw.
    """
    if samples < 1:
        raise ValueError("samples must be positive")
    if dimension < 0:
        raise ValueError("dimension must be nonnegative")
    if blocks_per_batch < 1:
        raise ValueError("blocks_per_batch must be positive")

    draws = np.empty((samples, dimension), dtype=np.float64)
    for indices in _block_indices(
        samples=samples, blocks_per_batch=blocks_per_batch
    ):
        blocks = [
            np.random.default_rng(np.random.SeedSequence([seed, stream, index]))
            .standard_normal((_BLOCK_SIZE, dimension))
            for index in indices
        ]
        for index, block in zip(indices, blocks, strict=True):
            start = index * _BLOCK_SIZE
            stop = min(start + _BLOCK_SIZE, samples)
            draws[start:stop] = block[: stop - start]
    return draws


def ddpm_sample(
    target: IsotropicGMM,
    *,
    schedule: DDPMSchedule,
    samples: int,
    seed: int,
    score_profile: NDArray[np.float64] | None = None,
    error_direction: NDArray[np.float64] | None = None,
    blocks_per_batch: int = 1,
    inactive_dimension: int = 0,
) -> NDArray[np.float64]:
    """Run Equation 9 in active coordinates and return those coordinates only.

    ``inactive_dimension`` records omitted standard-Gaussian coordinates for
    the caller's evidence metadata; Equation 9 leaves that independent law
    unchanged, so no inactive samples are materialized here.
    """
    if not np.all(target.variances == 1.0):
        raise ValueError("the theorem-family sampler requires unit variances")
    if inactive_dimension < 0:
        raise ValueError("inactive_dimension must be nonnegative")

    rank = target.dimension
    if score_profile is not None:
        score_profile = np.asarray(score_profile, dtype=np.float64)
        if score_profile.shape != (schedule.steps + 1,):
            raise ValueError("score_profile must have shape (steps + 1,)")
        if not np.all(np.isfinite(score_profile)):
            raise ValueError("score_profile must be finite")
        if error_direction is None:
            raise ValueError("error_direction is required with score_profile")
        error_direction = np.asarray(error_direction, dtype=np.float64)
        if error_direction.shape != (rank,):
            raise ValueError("error_direction must have shape (target.dimension,)")
        if not np.all(np.isfinite(error_direction)):
            raise ValueError("error_direction must be finite")
    elif error_direction is not None:
        raise ValueError("error_direction requires score_profile")

    y = normal_blocks(
        seed=seed,
        stream=0,
        samples=samples,
        dimension=rank,
        blocks_per_batch=blocks_per_batch,
    )
    for t in range(schedule.steps, 1, -1):
        alpha_t = schedule.alpha[t]
        score = score_at_time(target, y, schedule=schedule, time=t)
        if score_profile is not None:
            score = score + score_profile[t] * error_direction
        noise = normal_blocks(
            seed=seed,
            stream=t,
            samples=samples,
            dimension=rank,
            blocks_per_batch=blocks_per_batch,
        )
        y = (
            (y + (1.0 - alpha_t) * score) / np.sqrt(alpha_t)
            + np.sqrt(1.0 - alpha_t) * noise
        )
    return y
