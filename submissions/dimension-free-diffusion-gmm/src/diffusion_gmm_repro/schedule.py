"""Paper-exact DDPM schedule construction."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class DDPMSchedule:
    """Equation 14 coefficients indexed from zero through ``steps``."""

    steps: int
    c0: float
    c1: float
    alpha_bar: NDArray[np.float64]
    alpha: NDArray[np.float64]


def paper_schedule(
    *, steps: int, c0: float = 2.0, c1: float = 10.0
) -> DDPMSchedule:
    """Construct the reverse-time schedule defined by Equation 14."""
    if steps < 2:
        raise ValueError("steps must be at least 2")
    if not np.isfinite(c0) or not np.isfinite(c1) or c0 <= 0.0 or c1 <= 0.0:
        raise ValueError("c0 and c1 must be finite and positive")
    if c1 / c0 <= 4.0:
        raise ValueError("c1 / c0 must be greater than 4")

    alpha_bar = np.empty(steps + 1, dtype=np.float64)
    alpha_bar[0] = 1.0
    alpha_bar[steps] = steps ** (-c0)
    rate = c1 * np.log(float(steps)) / steps
    for time in range(steps, 1, -1):
        current = alpha_bar[time]
        alpha_bar[time - 1] = current + rate * current * (1.0 - current)

    alpha = np.ones(steps + 1, dtype=np.float64)
    alpha[1:] = alpha_bar[1:] / alpha_bar[:-1]
    if np.any(alpha[1:] <= 0.0) or np.any(alpha[1:] > 1.0):
        raise ValueError("Equation 14 produced alpha outside (0, 1]")
    return DDPMSchedule(steps, c0, c1, alpha_bar, alpha)
