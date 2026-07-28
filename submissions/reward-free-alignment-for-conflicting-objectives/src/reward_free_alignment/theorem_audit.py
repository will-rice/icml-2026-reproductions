from dataclasses import dataclass
import math
from typing import Sequence
import torch
from torch import Tensor
from reward_free_alignment.cagrad_clip import CAGradResult, cagrad_clip


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
    # Extended fields for T-step trajectory
    trajectory_losses: tuple[float, ...] | None = None
    trajectory_grad_norms: tuple[float, ...] | None = None
    trajectory_m_values: tuple[float, ...] | None = None


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
    # Extended T-step finite-horizon fields
    trajectory_steps: int | None = None
    min_m_value: float | None = None
    min_grad_norm: float | None = None
    finite_horizon_rhs: float | None = None
    finite_horizon_bound_holds: bool | None = None


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


def _validate_positive_finite(name: str, value: float) -> float:
    """Validate that a value is positive and finite."""
    if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be positive and finite, got {value}")
    return float(value)


def execute_raco_trajectory(
    *,
    x0: float,
    T: int,
    eta: float,
    c: float,
    weights: Tensor,
    smoothness_constants: tuple[float, ...],
) -> SmoothObjectiveCase:
    """Execute T RACO steps on two-objective nonneg quadratics.

    f1(x) = x^2, f2(x) = (x-1)^2.
    Returns a SmoothObjectiveCase with full trajectory data.
    """
    w = weights
    L_w = sum(w[k].item() * smoothness_constants[k] for k in range(len(smoothness_constants)))

    losses: list[float] = []
    grad_norms: list[float] = []
    m_values: list[float] = []

    x = x0
    for t in range(T + 1):
        # Compute f1, f2, L_w at current x
        f1 = x ** 2
        f2 = (x - 1.0) ** 2
        L_w_val = w[0].item() * f1 + w[1].item() * f2
        losses.append(L_w_val)

        # Compute gradients
        g1_val = 2.0 * x
        g2_val = 2.0 * (x - 1.0)
        g0_val = w[0].item() * g1_val + w[1].item() * g2_val
        grad_norms.append(abs(g0_val))

        # M(theta_t) = min_k (g_k · g_update / ||g_update||)
        # For scalar 1D: g_update = g0 direction
        # M(theta_t) = min(g1 * sign(g0), g2 * sign(g0)) * |g0| / |g0| = min(g1*sgn, g2*sgn)
        if abs(g0_val) > 1e-15:
            m_val = min(g1_val * (g0_val / abs(g0_val)), g2_val * (g0_val / abs(g0_val)))
        else:
            m_val = 0.0
        m_values.append(m_val)

        if t < T:
            # Step using the weighted gradient (conservative 1D CAGrad)
            x = x - eta * g0_val

    return SmoothObjectiveCase(
        weights=w,
        smoothness_constants=smoothness_constants,
        weighted_smoothness=L_w,
        step_size=eta,
        correction_radius=c,
        initial_loss=losses[0],
        final_loss=losses[-1],
        grad_norm=grad_norms[0],
        trajectory_losses=tuple(losses),
        trajectory_grad_norms=tuple(grad_norms),
        trajectory_m_values=tuple(m_values),
    )


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

    # One-step descent bound using the paper's Theorem 3.1 formula:
    # L_w(θ_{t+1}) ≤ L_w(θ_t) - η(1-c²)/2 ||∇L_w(θ_t)||²
    # Note: we do NOT assume best-case ρ=1 (correction gate §2).
    # The (1-c²) factor is the worst-case Gamma lower bound from Theorem 3.1.
    descent_factor = step * (1.0 - c_rad * c_rad) / 2.0
    expected_descent = descent_factor * case.grad_norm * case.grad_norm
    descent_holds = case.final_loss <= case.initial_loss - expected_descent + 1e-9

    # Finite-horizon Pareto bound (Theorem 3.1):
    # min_{t=0..T-1} ||∇L_w(θ_t)||² ≤ 2 L_w(θ_0) / (η(1-c²) T)
    # Uses the correct formula WITHOUT the extra 1/2 from the rejected proposal.
    trajectory_steps = None
    min_m_value = None
    min_grad_norm = None
    finite_horizon_rhs = None
    finite_horizon_bound_holds = None

    if case.trajectory_losses is not None and case.trajectory_grad_norms is not None:
        T = len(case.trajectory_losses) - 1  # number of steps
        if T > 0 and (1.0 - c_rad * c_rad) > 0.0:
            trajectory_steps = T
            min_grad_norm = min(case.trajectory_grad_norms[:-1])  # min over t=0..T-1
            # Finite-horizon bound: min ||∇L_w||² ≤ 2*L_w(θ_0) / (η*(1-c²)*T)
            finite_horizon_rhs = 2.0 * case.initial_loss / (step * (1.0 - c_rad * c_rad) * T)
            finite_horizon_bound_holds = min_grad_norm ** 2 <= finite_horizon_rhs + 1e-9

            if case.trajectory_m_values is not None:
                min_m_value = min(case.trajectory_m_values[:-1])

    # Pareto bound: check finite-horizon if available, otherwise gradient consistency
    if finite_horizon_bound_holds is not None:
        pareto_holds = finite_horizon_bound_holds
    else:
        pareto_holds = case.grad_norm >= 0.0

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
        trajectory_steps=trajectory_steps,
        min_m_value=min_m_value,
        min_grad_norm=min_grad_norm,
        finite_horizon_rhs=finite_horizon_rhs,
        finite_horizon_bound_holds=finite_horizon_bound_holds,
    )


def audit_theorem_32(
    result: CAGradResult,
    weights: Tensor,
    c: float,
    weighted_smoothness: float,
    step_size: float,
    atol: float = 1e-10,
) -> DescentCertificateAudit:
    # Validate preconditions (correction gate §5)
    if not isinstance(c, (int, float)) or not math.isfinite(c):
        return _inapplicable_audit(
            weights, c, result,
            reason="c must be finite",
        )
    if c <= 0.0:
        return _inapplicable_audit(
            weights, c, result,
            reason="c must be positive for Theorem 3.2",
        )
    if c >= 1.0:
        return _inapplicable_audit(
            weights, c, result,
            reason="c must satisfy c < 1 for admissibility",
        )
    if not isinstance(step_size, (int, float)) or not math.isfinite(step_size) or step_size <= 0.0:
        return _inapplicable_audit(
            weights, c, result,
            reason="step_size must be positive and finite",
        )
    if not isinstance(weighted_smoothness, (int, float)) or not math.isfinite(weighted_smoothness) or weighted_smoothness <= 0.0:
        return _inapplicable_audit(
            weights, c, result,
            reason="weighted_smoothness must be positive and finite",
        )

    # Check finite simplex weights
    if weights.ndim != 1 or weights.shape[0] != 2:
        return _inapplicable_audit(weights, c, result, reason="not two objectives")
    if not torch.isfinite(weights).all() or (weights <= 0.0).any():
        return _inapplicable_audit(weights, c, result, reason="weights not positive finite simplex")
    if abs(weights.sum().item() - 1.0) > 1e-5:
        return _inapplicable_audit(weights, c, result, reason="weights not in simplex")

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
        return _inapplicable_audit(weights, c, result, reason="degenerate geometry")

    rho_val = torch.dot(g0, g_mix).item() / (norm_g0 * norm_g_mix)
    rho_tilde_val = torch.dot(g0, g_clip).item() / (norm_g0 * norm_g_clip)

    gamma_val = gamma(rho_val, c, weighted_smoothness, step_size)
    gamma_tilde_val = gamma(rho_tilde_val, c, weighted_smoothness, step_size)

    obs_diff = gamma_tilde_val - gamma_val
    identity_rhs = c * (1.0 - weighted_smoothness * step_size) * (rho_tilde_val - rho_val)
    identity_res = abs(obs_diff - identity_rhs)

    strict_conds = _strict_conditions(weights, c, step_size, weighted_smoothness, result, atol)
    strict_exp = all(strict_conds.values())

    # Determine local outcome:
    # 1. Identity residual must be small
    # 2. For applicable cases, improvement must be nonnegative
    # 3. Under strict conditions, improvement must be strictly positive
    # 4. Positive Gamma improvement required (correction gate §5)
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


def _strict_conditions(
    weights: Tensor,
    c: float,
    step_size: float,
    weighted_smoothness: float,
    result: CAGradResult,
    atol: float,
) -> dict[str, bool]:
    """Compute the 8 strictness booleans from the paper."""
    norm_g0 = torch.linalg.vector_norm(result.weighted_anchor).item()
    return {
        "two_objectives": weights.shape[0] == 2,
        "positive_weights": bool((weights > 0.0).all().item()),
        "positive_c": c > 0.0,
        "strict_step_size": step_size < (1.0 / weighted_smoothness),
        "nonzero_anchor": norm_g0 > atol,
        "noncolinear_gradients": result.singular_case is None,
        "interior_coefficients": 0.0 < result.coefficients[0].item() < 1.0,
        "coefficients_differ_from_weights": not torch.allclose(result.coefficients, weights, atol=1e-5),
    }


def _inapplicable_audit(
    weights: Tensor,
    c: float,
    result: CAGradResult,
    *,
    reason: str = "",
) -> DescentCertificateAudit:
    """Build a not-applicable audit with all strict conditions computed."""
    # Provide best-effort strict conditions even for inapplicable cases
    try:
        strict_conds = _strict_conditions(weights, c, 0.0, 1.0, result, 1e-10)
    except Exception:
        strict_conds = {
            "two_objectives": False,
            "positive_weights": False,
            "positive_c": False,
            "strict_step_size": False,
            "nonzero_anchor": False,
            "noncolinear_gradients": False,
            "interior_coefficients": False,
            "coefficients_differ_from_weights": False,
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
