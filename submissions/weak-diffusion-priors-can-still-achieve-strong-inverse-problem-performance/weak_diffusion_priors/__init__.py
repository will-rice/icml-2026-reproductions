"""Reproduction core module for Weak Diffusion Priors Can Still Achieve Strong Inverse-Problem Performance."""

from .theory import simulate_theorem_3_1_posterior_concentration
from .inverse_problem import evaluate_table_1_inverse_problem_baselines

__all__ = [
    "simulate_theorem_3_1_posterior_concentration",
    "evaluate_table_1_inverse_problem_baselines",
]
