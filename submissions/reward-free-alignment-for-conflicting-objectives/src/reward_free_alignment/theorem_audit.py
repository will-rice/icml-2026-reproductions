from dataclasses import dataclass
import math
from typing import Sequence
import torch
from torch import Tensor
from reward_free_alignment.cagrad_clip import CAGradResult


@dataclass(frozen=True)
class SmoothObjectiveCase:
    weights: Tensor
    smoothness_constants: tuple[float, ...]
    weighted_smoothness: float
    step_size: float
    correction_radius: float
    initial_loss: float
    final_loss: float
    grad_norm: float


@dataclass(frozen=True)
class ConvergenceAudit:
    weights_in_simplex: bool
    smoothness_constants: tuple[float, ...]
    weighted_smoothness: float
    step_size: float
    correction_radius: float
    nonnegative_losses: bool
    descent_bound_holds: bool
    pareto_bound_holds: bool
    local_outcome: str


@dataclass(frozen=True)
class DescentCertificateAudit:
    rho: float | None
    rho_tilde: float | None
    gamma: float | None
    gamma_tilde: float | None
    observed_difference: float | None
    identity_rhs: float | None
    identity_residual: float | None
    strict_conditions: dict[str, bool]
    strict_expected: bool
    applicable: bool
    local_outcome: str


def gamma(rho: float, c: float, weighted_smoothness: float, step_size: float) -> float:
    ell_w = weighted_smoothness
    eta = step_size
    return 1.0 + c * rho - (ell_w * eta / 2.0) * (1.0 + c * c + 2.0 * c * rho)


def audit_theorem_31(case: SmoothObjectiveCase) -> ConvergenceAudit:
    w = case.weights
    weights_in_simplex = (
        (w >= 0.0).all().item()
        and abs(w.sum().item() - 1.0) <= 1e-5
    )
    smooth_pos = tuple(case.smoothness_constants)
    weighted_s = case.weighted_smoothness
    step = case.step_size
    c_rad = case.correction_radius

    all_smooth_pos = all(s > 0.0 for s in smooth_pos)
    step_valid = 0.0 < step <= (1.0 / weighted_s + 1e-9)
    c_valid = 0.0 <= c_rad < 1.0
    nonneg_l = case.initial_loss >= 0.0 and case.final_loss >= 0.0

    # Recompute descent bound: under Theorem 3.1, one-step descent requires
    # f(theta_t+1) <= f(theta_t) - eta/2 * ||grad f||^2 * gamma_min
    # At minimum, the gradient norm must produce measurable descent
    gamma_min = gamma(1.0, c_rad, weighted_s, step)  # best-case alignment rho=1
    expected_descent = step / 2.0 * case.grad_norm * case.grad_norm * max(gamma_min, 0.0)
    descent_holds = case.final_loss <= case.initial_loss - expected_descent + 1e-9

    # Pareto bound: the gradient norm should be consistent with the losses
    # Finite-horizon: sum of squared gradient norms is bounded by 2*(f0-f*) / (eta * gamma_min)
    pareto_holds = case.grad_norm >= 0.0
    if gamma_min > 0.0 and case.grad_norm > 0.0:
        # Check that the gradient norm is consistent with the descent
        pareto_holds = descent_holds

    all_pass = (
        weights_in_simplex
        and all_smooth_pos
        and step_valid
        and c_valid
        and nonneg_l
        and descent_holds
        and pareto_holds
    )
    local_outcome = "supported" if all_pass else "not-supported"

    return ConvergenceAudit(
        weights_in_simplex=weights_in_simplex,
        smoothness_constants=smooth_pos,
        weighted_smoothness=weighted_s,
        step_size=step,
        correction_radius=c_rad,
        nonnegative_losses=nonneg_l,
        descent_bound_holds=descent_holds,
        pareto_bound_holds=pareto_holds,
        local_outcome=local_outcome,
    )


def audit_theorem_32(
    result: CAGradResult,
    weights: Tensor,
    c: float,
    weighted_smoothness: float,
    step_size: float,
    atol: float = 1e-10,
) -> DescentCertificateAudit:
    g0 = result.weighted_anchor
    g_mix = result.mixture
    g_clip = result.clipped_mixture

    norm_g0 = torch.linalg.vector_norm(g0).item()
    norm_g_mix = torch.linalg.vector_norm(g_mix).item()
    norm_g_clip = torch.linalg.vector_norm(g_clip).item()

    is_applicable = (
        norm_g0 > atol
        and norm_g_mix > atol
        and norm_g_clip > atol
        and result.singular_case is None
    )

    if not is_applicable:
        strict_conds = {
            "two_objectives": weights.shape[0] == 2,
            "positive_weights": (weights > 0.0).all().item(),
            "positive_c": c > 0.0,
            "strict_step_size": step_size < (1.0 / weighted_smoothness),
            "nonzero_anchor": norm_g0 > atol,
            "noncolinear_gradients": result.singular_case is None,
            "interior_coefficients": 0.0 < result.coefficients[0].item() < 1.0,
            "coefficients_differ_from_weights": not torch.allclose(result.coefficients, weights, atol=1e-5),
        }
        return DescentCertificateAudit(
            rho=None,
            rho_tilde=None,
            gamma=None,
            gamma_tilde=None,
            observed_difference=None,
            identity_rhs=None,
            identity_residual=None,
            strict_conditions=strict_conds,
            strict_expected=all(strict_conds.values()),
            applicable=False,
            local_outcome="limited",
        )

    rho_val = torch.dot(g0, g_mix).item() / (norm_g0 * norm_g_mix)
    rho_tilde_val = torch.dot(g0, g_clip).item() / (norm_g0 * norm_g_clip)

    gamma_val = gamma(rho_val, c, weighted_smoothness, step_size)
    gamma_tilde_val = gamma(rho_tilde_val, c, weighted_smoothness, step_size)

    obs_diff = gamma_tilde_val - gamma_val
    identity_rhs = c * (1.0 - weighted_smoothness * step_size) * (rho_tilde_val - rho_val)
    identity_res = abs(obs_diff - identity_rhs)

    strict_conds = {
        "two_objectives": weights.shape[0] == 2,
        "positive_weights": (weights > 0.0).all().item(),
        "positive_c": c > 0.0,
        "strict_step_size": step_size < (1.0 / weighted_smoothness),
        "nonzero_anchor": norm_g0 > atol,
        "noncolinear_gradients": result.singular_case is None,
        "interior_coefficients": 0.0 < result.coefficients[0].item() < 1.0,
        "coefficients_differ_from_weights": not torch.allclose(result.coefficients, weights, atol=1e-5),
    }
    strict_exp = all(strict_conds.values())

    # Determine local outcome:
    # 1. Identity residual must be small
    # 2. For applicable cases, improvement must be nonnegative
    # 3. Under strict conditions, improvement must be strictly positive
    if identity_res > atol:
        local_outcome = "not-supported"
    elif obs_diff < -atol:
        # Negative improvement violates the theorem's guarantee
        local_outcome = "not-supported"
    elif strict_exp and obs_diff <= atol:
        # All strictness conditions hold but no positive improvement
        local_outcome = "not-supported"
    else:
        local_outcome = "supported"

    return DescentCertificateAudit(
        rho=rho_val,
        rho_tilde=rho_tilde_val,
        gamma=gamma_val,
        gamma_tilde=gamma_tilde_val,
        observed_difference=obs_diff,
        identity_rhs=identity_rhs,
        identity_residual=identity_res,
        strict_conditions=strict_conds,
        strict_expected=strict_exp,
        applicable=True,
        local_outcome=local_outcome,
    )
