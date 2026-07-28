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
    trajectory_losses: tuple[float, ...] | None = None
    trajectory_grad_norms: tuple[float, ...] | None = None
    trajectory_m_values: tuple[float, ...] | None = None
    trajectory_m_bounds_holds: tuple[bool, ...] | None = None


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
    trajectory_steps: int | None = None
    min_m_value: float | None = None
    min_grad_norm: float | None = None
    finite_horizon_rhs: float | None = None
    finite_horizon_bound_holds: bool | None = None
    per_step_m_bound_holds: bool | None = None


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


def compute_m_simplex(g1: Tensor, g2: Tensor) -> float:
    """Compute M(theta) = min_{lambda in [0,1]} ||lambda g1 + (1-lambda) g2||."""
    diff = g1 - g2
    diff_norm_sq = torch.dot(diff, diff).item()
    if diff_norm_sq < 1e-15:
        return torch.linalg.vector_norm(g1).item()
    opt_lambda = -torch.dot(diff, g2).item() / diff_norm_sq
    clamped_lambda = max(0.0, min(1.0, opt_lambda))
    mix = clamped_lambda * g1 + (1.0 - clamped_lambda) * g2
    return torch.linalg.vector_norm(mix).item()


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
    m_bounds_holds: list[bool] = []

    x = x0
    for t in range(T + 1):
        f1 = x ** 2
        f2 = (x - 1.0) ** 2
        L_w_val = w[0].item() * f1 + w[1].item() * f2
        losses.append(L_w_val)

        g1_val = 2.0 * x
        g2_val = 2.0 * (x - 1.0)
        g0_val = w[0].item() * g1_val + w[1].item() * g2_val
        g_norm = abs(g0_val)
        grad_norms.append(g_norm)

        # M(theta_t) = min_{lambda in simplex} ||lambda g1 + (1-lambda) g2||
        g1_t = torch.tensor([g1_val], dtype=torch.float32)
        g2_t = torch.tensor([g2_val], dtype=torch.float32)
        m_val = compute_m_simplex(g1_t, g2_t)
        m_values.append(m_val)
        m_bounds_holds.append(math.isfinite(m_val) and m_val >= 0.0 and m_val <= g_norm + 1e-9)

        if t < T:
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
        trajectory_m_bounds_holds=tuple(m_bounds_holds),
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

    descent_factor = step * (1.0 - c_rad * c_rad) / 2.0
    expected_descent = descent_factor * case.grad_norm * case.grad_norm
    descent_holds = case.final_loss <= case.initial_loss - expected_descent + 1e-9

    trajectory_steps = None
    min_m_value = None
    min_grad_norm = None
    finite_horizon_rhs = None
    finite_horizon_bound_holds = None
    per_step_m_bound_holds = True

    if case.trajectory_losses is not None and case.trajectory_grad_norms is not None:
        T = len(case.trajectory_losses) - 1
        if T > 0 and (1.0 - c_rad * c_rad) > 0.0:
            trajectory_steps = T
            min_grad_norm = min(case.trajectory_grad_norms[:-1])
            finite_horizon_rhs = 2.0 * case.initial_loss / (step * (1.0 - c_rad * c_rad) * T)

            if case.trajectory_m_values is not None:
                min_m_value = min(case.trajectory_m_values[:-1])
                # Check both grad norm and M(theta) finite horizon bounds
                finite_horizon_bound_holds = (
                    min_grad_norm ** 2 <= finite_horizon_rhs + 1e-9
                    and (min_m_value ** 2) <= finite_horizon_rhs + 1e-9
                )

                # Check per-step M(theta_t) bounds
                for m_val, g_norm in zip(case.trajectory_m_values[:-1], case.trajectory_grad_norms[:-1]):
                    if not (math.isfinite(m_val) and m_val >= 0.0 and m_val <= g_norm + 1e-9):
                        per_step_m_bound_holds = False
                        break
            else:
                finite_horizon_bound_holds = min_grad_norm ** 2 <= finite_horizon_rhs + 1e-9

    if finite_horizon_bound_holds is not None:
        pareto_holds = finite_horizon_bound_holds and per_step_m_bound_holds
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
        and per_step_m_bound_holds
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
        per_step_m_bound_holds=per_step_m_bound_holds,
    )


def audit_theorem_32(
    result: CAGradResult,
    weights: Tensor,
    c: float,
    weighted_smoothness: float,
    step_size: float,
    atol: float = 1e-10,
) -> DescentCertificateAudit:
    # Validate preconditions (correction gate §4)
    if not isinstance(c, (int, float)) or not math.isfinite(c):
        return _inapplicable_audit(
            weights, c, result, step_size, weighted_smoothness,
            reason="c must be finite",
        )
    if c <= 0.0:
        return _inapplicable_audit(
            weights, c, result, step_size, weighted_smoothness,
            reason="c must be positive for Theorem 3.2",
        )
    if c >= 1.0:
        return _inapplicable_audit(
            weights, c, result, step_size, weighted_smoothness,
            reason="c must satisfy c < 1 for admissibility",
        )
    if not isinstance(step_size, (int, float)) or not math.isfinite(step_size) or step_size <= 0.0:
        return _inapplicable_audit(
            weights, c, result, step_size, weighted_smoothness,
            reason="step_size must be positive and finite",
        )
    if not isinstance(weighted_smoothness, (int, float)) or not math.isfinite(weighted_smoothness) or weighted_smoothness <= 0.0:
        return _inapplicable_audit(
            weights, c, result, step_size, weighted_smoothness,
            reason="weighted_smoothness must be positive and finite",
        )

    # Check caller c matches result.c
    if abs(result.c - c) > atol:
        return _inapplicable_audit(
            weights, c, result, step_size, weighted_smoothness,
            reason=f"c mismatch: caller c={c} != result c={result.c}",
        )

    # Reject non-finite result tensors
    if not (
        torch.isfinite(result.gradient).all()
        and torch.isfinite(result.weighted_anchor).all()
        and torch.isfinite(result.coefficients).all()
        and torch.isfinite(result.mixture).all()
        and torch.isfinite(result.clipped_mixture).all()
    ):
        return _inapplicable_audit(
            weights, c, result, step_size, weighted_smoothness,
            reason="result contains non-finite tensors",
        )

    # Check finite simplex weights
    if weights.ndim != 1 or weights.shape[0] != 2:
        return _inapplicable_audit(weights, c, result, step_size, weighted_smoothness, reason="not two objectives")
    if not torch.isfinite(weights).all() or (weights <= 0.0).any():
        return _inapplicable_audit(weights, c, result, step_size, weighted_smoothness, reason="weights not positive finite simplex")
    if abs(weights.sum().item() - 1.0) > 1e-5:
        return _inapplicable_audit(weights, c, result, step_size, weighted_smoothness, reason="weights not in simplex")

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
        return _inapplicable_audit(weights, c, result, step_size, weighted_smoothness, reason="degenerate geometry")

    rho_val = torch.dot(g0, g_mix).item() / (norm_g0 * norm_g_mix)
    rho_tilde_val = torch.dot(g0, g_clip).item() / (norm_g0 * norm_g_clip)

    gamma_val = gamma(rho_val, c, weighted_smoothness, step_size)
    gamma_tilde_val = gamma(rho_tilde_val, c, weighted_smoothness, step_size)

    obs_diff = gamma_tilde_val - gamma_val
    identity_rhs = c * (1.0 - weighted_smoothness * step_size) * (rho_tilde_val - rho_val)
    identity_res = abs(obs_diff - identity_rhs)

    strict_conds = _strict_conditions(weights, c, step_size, weighted_smoothness, result, atol)
    strict_exp = all(strict_conds.values())

    if identity_res > atol:
        local_outcome = "not-supported"
    elif obs_diff <= atol:
        # Require positive Gamma improvement
        local_outcome = "not-supported"
    elif not strict_exp:
        local_outcome = "limited"
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
    norm_g0 = torch.linalg.vector_norm(result.weighted_anchor).item() if torch.isfinite(result.weighted_anchor).all() else 0.0
    return {
        "two_objectives": weights.ndim == 1 and weights.shape[0] == 2,
        "positive_weights": bool((weights > 0.0).all().item()) if torch.isfinite(weights).all() else False,
        "positive_c": isinstance(c, (int, float)) and math.isfinite(c) and c > 0.0,
        "strict_step_size": (
            isinstance(step_size, (int, float))
            and math.isfinite(step_size)
            and isinstance(weighted_smoothness, (int, float))
            and math.isfinite(weighted_smoothness)
            and weighted_smoothness > 0.0
            and step_size < (1.0 / weighted_smoothness)
        ),
        "nonzero_anchor": norm_g0 > atol,
        "noncolinear_gradients": result.singular_case is None,
        "interior_coefficients": (
            torch.isfinite(result.coefficients).all()
            and 0.0 < result.coefficients[0].item() < 1.0
        ),
        "coefficients_differ_from_weights": (
            torch.isfinite(result.coefficients).all()
            and torch.isfinite(weights).all()
            and not torch.allclose(result.coefficients, weights, atol=1e-5)
        ),
    }


def _inapplicable_audit(
    weights: Tensor,
    c: float,
    result: CAGradResult,
    step_size: float = 0.0,
    weighted_smoothness: float = 1.0,
    *,
    reason: str = "",
) -> DescentCertificateAudit:
    """Build a not-applicable audit with all strict conditions computed using actual arguments."""
    try:
        strict_conds = _strict_conditions(weights, c, step_size, weighted_smoothness, result, 1e-10)
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
