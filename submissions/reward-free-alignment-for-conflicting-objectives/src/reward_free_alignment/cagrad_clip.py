from dataclasses import dataclass
import math
from typing import Sequence
import torch
from torch import Tensor


@dataclass(frozen=True)
class AlphaSolution:
    alpha: float
    coefficients: Tensor
    weighted_anchor: Tensor
    objective_value: float
    candidate_count: int
    singular_case: str | None


@dataclass(frozen=True)
class CAGradResult:
    gradient: Tensor
    weighted_anchor: Tensor
    coefficients: Tensor
    clipped_coefficients: Tensor
    mixture: Tensor
    clipped_mixture: Tensor
    clipped_coordinates: tuple[int, ...]
    singular_case: str | None
    c: float


def validate_weights(weights: Tensor, atol: float = 1e-5) -> Tensor:
    if weights.ndim != 1 or weights.shape[0] != 2:
        raise ValueError(f"weights must be a 1D tensor of length 2, got shape {weights.shape}")
    if not torch.isfinite(weights).all():
        raise ValueError(f"weights must be finite, got {weights}")
    if (weights < 0.0).any():
        raise ValueError(f"weights must be non-negative, got {weights}")
    w_sum = weights.sum().item()
    if abs(w_sum - 1.0) > atol:
        raise ValueError(f"weights must sum to 1.0, got sum {w_sum}")
    return weights


def _validate_finite_gradient(name: str, g: Tensor) -> None:
    """Reject non-finite gradient vectors."""
    if not torch.isfinite(g).all():
        raise ValueError(f"{name} must be finite, got non-finite values")


def solve_two_objective_alpha(
    g1: Tensor,
    g2: Tensor,
    weights: Tensor,
    c: float,
    atol: float = 1e-12,
) -> AlphaSolution:
    validate_weights(weights)
    _validate_finite_gradient("g1", g1)
    _validate_finite_gradient("g2", g2)
    if not isinstance(c, (int, float)) or not math.isfinite(c) or c < 0.0:
        raise ValueError(f"c must be non-negative and finite, got {c}")
    if c >= 1.0:
        raise ValueError(f"c must satisfy 0 <= c < 1 (Theorem 3.1 requirement), got {c}")
    if g1.shape != g2.shape or g1.ndim != 1:
        raise ValueError("g1 and g2 must be 1D tensors of the same shape")

    w1 = weights[0].item()
    g0 = weights[0] * g1 + weights[1] * g2
    norm_g0 = torch.linalg.vector_norm(g0).item()
    norm_g1 = torch.linalg.vector_norm(g1).item()
    norm_g2 = torch.linalg.vector_norm(g2).item()
    diff = g1 - g2
    norm_diff = torch.linalg.vector_norm(diff).item()

    ref_scale = max(norm_g1, norm_g2, 1e-300)
    rel_tol = atol * ref_scale

    if norm_g0 <= rel_tol:
        return AlphaSolution(
            alpha=w1,
            coefficients=weights.clone(),
            weighted_anchor=g0,
            objective_value=0.0,
            candidate_count=1,
            singular_case="zero_anchor",
        )

    if norm_diff <= rel_tol:
        return AlphaSolution(
            alpha=w1,
            coefficients=weights.clone(),
            weighted_anchor=g0,
            objective_value=torch.dot(g1, g0).item(),
            candidate_count=1,
            singular_case="identical_gradients",
        )

    if c <= atol:
        b1 = torch.dot(g1, g0).item()
        b2 = torch.dot(g2, g0).item()
        delta_b = b1 - b2
        if delta_b < -rel_tol:
            best_alpha = 1.0
        elif delta_b > rel_tol:
            best_alpha = 0.0
        else:
            best_alpha = w1
        best_obj = best_alpha * b1 + (1.0 - best_alpha) * b2
        coeffs = torch.tensor([best_alpha, 1.0 - best_alpha], dtype=g1.dtype, device=g1.device)
        return AlphaSolution(
            alpha=best_alpha,
            coefficients=coeffs,
            weighted_anchor=g0,
            objective_value=best_obj,
            candidate_count=1,
            singular_case="zero_radius",
        )

    s = c * norm_g0

    def h(a: float) -> float:
        mix = a * g1 + (1.0 - a) * g2
        return (torch.dot(mix, g0) + s * torch.linalg.vector_norm(mix)).item()

    if norm_g1 <= rel_tol or norm_g2 <= rel_tol:
        singular_case = "colinear_gradients"
    else:
        cos_sim = abs(torch.dot(g1, g2).item()) / (norm_g1 * norm_g2)
        singular_case = "colinear_gradients" if 1.0 - cos_sim <= 1e-5 else None

    if singular_case == "colinear_gradients":
        h0 = h(0.0)
        h1 = h(1.0)
        if h0 < h1 - rel_tol:
            best_alpha = 0.0
        elif h1 < h0 - rel_tol:
            best_alpha = 1.0
        elif abs(0.0 - w1) <= abs(1.0 - w1):
            best_alpha = 0.0
        else:
            best_alpha = 1.0
        best_obj = h(best_alpha)
        coeffs = torch.tensor([best_alpha, 1.0 - best_alpha], dtype=g1.dtype, device=g1.device)
        return AlphaSolution(
            alpha=best_alpha,
            coefficients=coeffs,
            weighted_anchor=g0,
            objective_value=best_obj,
            candidate_count=2,
            singular_case=singular_case,
        )

    # General case: enumerate candidate alpha values
    b1 = torch.dot(g1, g0).item()
    b2 = torch.dot(g2, g0).item()
    delta_b = b1 - b2

    q2 = norm_diff ** 2
    q1 = 2.0 * (torch.dot(g1, g2).item() - torch.dot(g2, g2).item())
    q0 = torch.dot(g2, g2).item()

    def Q(a: float) -> float:
        return q2 * a * a + q1 * a + q0

    raw_candidates = [0.0, 1.0]

    q_coeff_ref = max(abs(q2), abs(q1), abs(q0), 1e-300)
    q_tol = atol * q_coeff_ref

    if q2 > q_tol:
        disc_q = q1 * q1 - 4.0 * q2 * q0
        if disc_q >= 0:
            sqrt_dq = math.sqrt(max(0.0, disc_q))
            raw_candidates.append((-q1 - sqrt_dq) / (2.0 * q2))
            raw_candidates.append((-q1 + sqrt_dq) / (2.0 * q2))

    A = delta_b * delta_b * q2 - s * s * q2 * q2
    B = delta_b * delta_b * q1 - s * s * q2 * q1
    C = delta_b * delta_b * q0 - s * s * q1 * q1 / 4.0

    poly_coeff_ref = max(abs(A), abs(B), abs(C), 1e-300)
    poly_tol = atol * poly_coeff_ref

    def _stationarity_check(r: float) -> bool:
        q_val = Q(r)
        if q_val < -q_tol:
            return False
        lhs = delta_b * math.sqrt(max(0.0, q_val))
        rhs = -s * (q2 * r + q1 / 2.0)
        denom = max(abs(lhs), abs(rhs), 1e-300)
        return abs(lhs - rhs) <= 1e-5 * denom

    if abs(A) > poly_tol:
        disc_stat = B * B - 4.0 * A * C
        if disc_stat >= 0:
            sqrt_ds = math.sqrt(max(0.0, disc_stat))
            for r in [(-B - sqrt_ds) / (2.0 * A), (-B + sqrt_ds) / (2.0 * A)]:
                if -atol <= r <= 1.0 + atol:
                    if _stationarity_check(r):
                        raw_candidates.append(r)
    elif abs(B) > poly_tol:
        r = -C / B
        if -atol <= r <= 1.0 + atol:
            if _stationarity_check(r):
                raw_candidates.append(r)

    valid_candidates: list[float] = []
    for cand in raw_candidates:
        if not math.isfinite(cand):
            continue
        if -atol <= cand <= 1.0 + atol:
            clamped = max(0.0, min(1.0, float(cand)))
            if not any(abs(clamped - v) <= 1e-9 for v in valid_candidates):
                valid_candidates.append(clamped)

    evaluated = [(h(a), abs(a - w1), a) for a in valid_candidates]
    evaluated.sort(key=lambda x: (x[0], x[1], x[2]))
    best_alpha = evaluated[0][2]
    best_obj = evaluated[0][0]

    coeffs = torch.tensor([best_alpha, 1.0 - best_alpha], dtype=g1.dtype, device=g1.device)
    return AlphaSolution(
        alpha=best_alpha,
        coefficients=coeffs,
        weighted_anchor=g0,
        objective_value=best_obj,
        candidate_count=len(valid_candidates),
        singular_case=None,
    )


def cagrad_clip(
    gradients: Sequence[Tensor],
    weights: Tensor,
    c: float,
    atol: float = 1e-12,
) -> CAGradResult:
    grad_list = list(gradients)
    if len(grad_list) != 2:
        raise ValueError(f"cagrad_clip currently requires exactly 2 objective gradients, got {len(grad_list)}")

    g1, g2 = grad_list[0], grad_list[1]
    _validate_finite_gradient("g1", g1)
    _validate_finite_gradient("g2", g2)
    solution = solve_two_objective_alpha(g1, g2, weights, c, atol=atol)

    coeffs = solution.coefficients
    clipped_coeffs = torch.minimum(coeffs, weights)

    clipped_coords: list[int] = []
    for i in range(2):
        if (coeffs[i] - clipped_coeffs[i]).item() > 1e-7:
            clipped_coords.append(i)

    mixture = coeffs[0] * g1 + coeffs[1] * g2
    clipped_mixture = clipped_coeffs[0] * g1 + clipped_coeffs[1] * g2

    norm_clipped_mix = torch.linalg.vector_norm(clipped_mixture).item()

    ref_scale = max(
        torch.linalg.vector_norm(g1).item(),
        torch.linalg.vector_norm(g2).item(),
        1e-300,
    )
    norm_g0 = torch.linalg.vector_norm(solution.weighted_anchor).item()

    if solution.singular_case == "zero_anchor":
        gradient = torch.zeros_like(solution.weighted_anchor)
    elif norm_clipped_mix <= atol * ref_scale:
        gradient = solution.weighted_anchor
    else:
        gradient = solution.weighted_anchor + c * norm_g0 * (clipped_mixture / norm_clipped_mix)

    return CAGradResult(
        gradient=gradient,
        weighted_anchor=solution.weighted_anchor,
        coefficients=coeffs,
        clipped_coefficients=clipped_coeffs,
        mixture=mixture,
        clipped_mixture=clipped_mixture,
        clipped_coordinates=tuple(clipped_coords),
        singular_case=solution.singular_case,
        c=c,
    )
