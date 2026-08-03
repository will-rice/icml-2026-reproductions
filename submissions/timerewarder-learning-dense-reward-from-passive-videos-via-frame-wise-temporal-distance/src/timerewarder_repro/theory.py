"""Finite numerical audit of the paper's temporal-distance derivation."""

import numpy as np


def discounted_time_to_goal(remaining_steps: int, gamma: float) -> float:
    """Return the sparse-reward value for a deterministic path to a goal."""
    if (
        type(remaining_steps) is not int
        or remaining_steps < 0
        or not isinstance(gamma, (int, float))
        or not 0.0 <= float(gamma) <= 1.0
    ):
        raise ValueError("invalid remaining steps or discount")
    gamma = float(gamma)
    return -float(
        np.sum(
            np.power(gamma, np.arange(remaining_steps, dtype=np.float64)),
            dtype=np.float64,
        )
    )


def bellman_residual(values: np.ndarray, gamma: float) -> np.ndarray:
    """Evaluate V(n) - [-1 + gamma V(n-1)] for nonterminal states."""
    if (
        not isinstance(values, np.ndarray)
        or values.dtype != np.float64
        or values.ndim != 1
        or values.size < 2
        or not np.isfinite(values).all()
        or not 0.0 <= gamma <= 1.0
    ):
        raise ValueError("invalid Bellman inputs")
    return values[1:] - (-1.0 + gamma * values[:-1])


def audit_theory(
    horizons: tuple[int, ...] = (2, 3, 5, 9, 32, 64),
    gammas: tuple[float, ...] = (0.0, 0.5, 0.9, 0.99, 1.0),
) -> dict[str, object]:
    """Enumerate finite recurrences, assumptions, and an aliasing limitation."""
    if (
        not horizons
        or any(type(horizon) is not int or horizon < 1 for horizon in horizons)
        or not gammas
        or any(not 0.0 <= gamma <= 1.0 for gamma in gammas)
    ):
        raise ValueError("invalid theory audit grid")
    checks: list[dict[str, float | int | bool]] = []
    maximum = 0.0
    gamma_one_identity = True
    for horizon in horizons:
        for gamma in gammas:
            values = np.asarray(
                [
                    discounted_time_to_goal(remaining, gamma)
                    for remaining in range(horizon + 1)
                ],
                dtype=np.float64,
            )
            residual = float(np.max(np.abs(bellman_residual(values, gamma))))
            maximum = max(maximum, residual)
            if gamma == 1.0:
                gamma_one_identity = gamma_one_identity and bool(
                    np.array_equal(-values, np.arange(horizon + 1, dtype=np.float64))
                    and np.array_equal(
                        np.diff(-values), np.ones(horizon, dtype=np.float64)
                    )
                )
            checks.append(
                {
                    "horizon": horizon,
                    "gamma": gamma,
                    "max_absolute_residual": residual,
                    "passes": residual <= 1e-12,
                }
            )
    return {
        "equation": "V(n) = -1 + gamma * V(n - 1), with V(0) = 0",
        "meaning": (
            "Under the enumerated assumptions, negative optimal value is "
            "discounted remaining time-to-goal; at gamma=1 it is temporal distance."
        ),
        "checks": checks,
        "max_absolute_bellman_residual": maximum,
        "gamma_one_distance_identity": gamma_one_identity,
        "all_checks_pass": all(bool(item["passes"]) for item in checks)
        and gamma_one_identity,
        "assumptions": [
            "fully observable",
            "deterministic transitions",
            "optimal expert trajectory",
            "terminal goal",
            "observations uniquely identify phase/state",
        ],
        "aliasing_counterexample": {
            "observations": ["o0", "o1", "o2", "o3", "o1", "og"],
            "o1_remaining_distances": [4, 1],
            "single_frame_average": 2.5,
            "limitation": (
                "A single-frame predictor cannot assign both remaining distances "
                "to the aliased o1 observation."
            ),
        },
        "scope": (
            "Finite derivation audit only; it does not establish real-video "
            "Markovness or downstream RL performance."
        ),
    }
