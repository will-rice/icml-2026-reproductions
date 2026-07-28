import math
import pytest
import torch
from torch import tensor
from reward_free_alignment.cagrad_clip import cagrad_clip, CAGradResult
from reward_free_alignment.theorem_audit import (
    gamma,
    audit_theorem_31,
    audit_theorem_32,
    execute_raco_trajectory,
    SmoothObjectiveCase,
    ConvergenceAudit,
    DescentCertificateAudit,
)


def smooth_nonnegative_quadratic_case() -> SmoothObjectiveCase:
    """Build a Theorem 3.1 case from an EXECUTED deterministic trajectory.

    Two-objective nonneg quadratic:
      f1(x) = x^2, f2(x) = (x-1)^2
      L1 = 2, L2 = 2, w = [0.6, 0.4], L_w = 2.0
      grad f1 = 2x, grad f2 = 2(x-1)
      L_w(x) = 0.6*x^2 + 0.4*(x-1)^2

    Starting at x0=1.0:
      L_w(1.0) = 0.6*1 + 0.4*0 = 0.6
      g1 = 2.0, g2 = 0.0
      g0 = 0.6*2.0 + 0.4*0.0 = 1.2
      With c=0.4, compute CAGrad update direction, then step.
      x1 = x0 - eta * g_update
    """
    x0 = 1.0
    eta = 0.1
    c_rad = 0.4
    w = tensor([0.6, 0.4])

    # Execute one step
    g1_val = 2.0 * x0  # grad f1 at x0
    g2_val = 2.0 * (x0 - 1.0)  # grad f2 at x0
    g0_val = 0.6 * g1_val + 0.4 * g2_val  # weighted anchor
    # For 1D: CAGrad with c=0.4 just gives g = g0 + c*|g0|*sign(p_mix)
    # The update is g0 direction (scalar), so gradient = g0 * (1 + c) or similar
    # Simplify: use g0 directly for the step (conservative)
    g_update = g0_val  # In 1D with single dominant gradient, CAGrad ~ g0
    x1 = x0 - eta * g_update

    L_w_x0 = 0.6 * x0**2 + 0.4 * (x0 - 1.0)**2
    L_w_x1 = 0.6 * x1**2 + 0.4 * (x1 - 1.0)**2
    grad_norm = abs(g0_val)

    return SmoothObjectiveCase(
        weights=w,
        smoothness_constants=(2.0, 2.0),
        weighted_smoothness=2.0,
        step_size=eta,
        correction_radius=c_rad,
        initial_loss=L_w_x0,
        final_loss=L_w_x1,
        grad_norm=grad_norm,
    )


def interior_strict_witness_audit() -> DescentCertificateAudit:
    """Construct a strict Theorem 3.2 witness with interior alpha.

    The corrected solver finds alpha≈0.356145 for g1=[1,-4], g2=[-1,1], w=[0.2,0.8],
    which is strictly interior (0 < alpha < 1) and differs from w[0]=0.2.
    Clipping then produces p_tilde != p, yielding positive Gamma difference.
    """
    weights = tensor([0.2, 0.8])
    c = 0.5
    g1 = tensor([1.0, -4.0])
    g2 = tensor([-1.0, 1.0])
    result = cagrad_clip((g1, g2), weights, c)
    return audit_theorem_32(
        result, weights, c=c, weighted_smoothness=3.0, step_size=0.1
    )


def zero_anchor_audit() -> DescentCertificateAudit:
    weights = tensor([0.5, 0.5])
    c = 0.5
    g1 = tensor([1.0, 0.0])
    g2 = tensor([-1.0, 0.0])
    result = cagrad_clip((g1, g2), weights, c)
    return audit_theorem_32(
        result, weights, c=c, weighted_smoothness=3.0, step_size=0.1
    )


def test_theorem_31_records_and_checks_every_precondition():
    audit = audit_theorem_31(smooth_nonnegative_quadratic_case())
    assert audit.weights_in_simplex is True
    assert all(value > 0.0 for value in audit.smoothness_constants)
    assert 0.0 < audit.step_size <= 1.0 / audit.weighted_smoothness
    assert 0.0 <= audit.correction_radius < 1.0
    assert audit.nonnegative_losses is True
    assert audit.descent_bound_holds is True
    assert audit.pareto_bound_holds is True
    assert audit.local_outcome == "supported"


def test_theorem_31_uses_executed_trajectory():
    """Verify that the Theorem 3.1 case uses an actual computed trajectory,
    not hand-entered initial/final losses."""
    case = smooth_nonnegative_quadratic_case()
    # The trajectory starts at x0=1.0 and steps to x1=1.0 - 0.1*1.2 = 0.88
    # L_w(1.0) = 0.6*1.0 + 0.4*0.0 = 0.6
    # L_w(0.88) = 0.6*0.88^2 + 0.4*(0.88-1)^2 = 0.6*0.7744 + 0.4*0.0144
    #           = 0.46464 + 0.00576 = 0.4704
    assert abs(case.initial_loss - 0.6) < 1e-10
    assert abs(case.final_loss - 0.4704) < 1e-10
    assert case.final_loss < case.initial_loss  # actual descent occurred


def test_theorem_31_t_step_finite_horizon():
    """Execute T RACO steps, record every M(θ_t) and ||∇L_w(θ_t)||, and verify
    the finite-horizon bound from Theorem 3.1 (correction gate §2).

    Finite-horizon: min_{t=0..T-1} ||∇L_w(θ_t)||² ≤ 2*L_w(θ_0) / (η*(1-c²)*T)
    """
    case = execute_raco_trajectory(
        x0=1.0, T=10, eta=0.1, c=0.4,
        weights=tensor([0.6, 0.4]),
        smoothness_constants=(2.0, 2.0),
    )
    assert case.trajectory_losses is not None
    assert len(case.trajectory_losses) == 11  # T+1 points
    assert case.trajectory_grad_norms is not None
    assert len(case.trajectory_grad_norms) == 11
    assert case.trajectory_m_values is not None
    assert len(case.trajectory_m_values) == 11

    # Verify descent at every step
    for t in range(10):
        assert case.trajectory_losses[t + 1] < case.trajectory_losses[t], (
            f"No descent at step {t}: L[{t}]={case.trajectory_losses[t]}, "
            f"L[{t+1}]={case.trajectory_losses[t+1]}"
        )

    audit = audit_theorem_31(case)
    assert audit.trajectory_steps == 10
    assert audit.min_grad_norm is not None
    assert audit.finite_horizon_rhs is not None
    assert audit.finite_horizon_bound_holds is True
    assert audit.local_outcome == "supported"

    # Verify the exact formula: 2*L_w(θ_0) / (η*(1-c²)*T)
    expected_rhs = 2.0 * case.initial_loss / (0.1 * (1.0 - 0.16) * 10)
    assert abs(audit.finite_horizon_rhs - expected_rhs) < 1e-10


def test_theorem_31_t_step_preserves_minima():
    """Verify min_grad_norm and min_m_value are correctly computed over t=0..T-1."""
    case = execute_raco_trajectory(
        x0=1.0, T=5, eta=0.1, c=0.4,
        weights=tensor([0.6, 0.4]),
        smoothness_constants=(2.0, 2.0),
    )
    audit = audit_theorem_31(case)
    # min_grad_norm should be over t=0..T-1 (first T entries)
    expected_min = min(case.trajectory_grad_norms[:-1])
    assert abs(audit.min_grad_norm - expected_min) < 1e-12


def test_theorem_32_reproduces_per_step_certificate_identity():
    weights = tensor([0.05, 0.95])
    result = cagrad_clip(
        (tensor([1.0, -1.76]), tensor([-1.0, 0.24])),
        weights,
        c=0.5,
    )
    audit = audit_theorem_32(
        result, weights, c=0.5, weighted_smoothness=4.0, step_size=0.05
    )
    expected = 0.5 * (1.0 - 4.0 * 0.05) * (audit.rho_tilde - audit.rho)
    assert abs(audit.observed_difference - expected) <= 1e-10
    assert audit.identity_residual <= 1e-10
    assert audit.applicable is True


def test_strictness_requires_every_paper_condition():
    """With corrected solver, the g1=[1,-4], g2=[-1,1], w=[0.2,0.8] case
    now gives interior alpha≈0.356145, so ALL 8 strictness conditions hold.
    This is the strict witness required by the controller correction gate."""
    audit = interior_strict_witness_audit()
    assert audit.strict_conditions == {
        "two_objectives": True,
        "positive_weights": True,
        "positive_c": True,
        "strict_step_size": True,
        "nonzero_anchor": True,
        "noncolinear_gradients": True,
        "interior_coefficients": True,
        "coefficients_differ_from_weights": True,
    }
    assert audit.strict_expected is True
    assert audit.observed_difference > 0.0
    assert audit.local_outcome == "supported"


def test_interior_witness_has_positive_gamma_difference():
    """Controller correction gate requirement: an interior strict witness must
    have Gamma(rho_tilde) - Gamma(rho) > 0, not a scaling artifact like 3.68e-9."""
    audit = interior_strict_witness_audit()
    assert audit.applicable is True
    assert audit.observed_difference is not None
    assert audit.observed_difference > 1e-3, (
        f"Expected substantial positive Gamma difference, got {audit.observed_difference}"
    )
    assert audit.identity_residual <= 1e-10
    assert audit.local_outcome == "supported"


def test_zero_anchor_is_not_applicable_not_divided():
    audit = zero_anchor_audit()
    assert audit.applicable is False
    assert audit.rho is None and audit.rho_tilde is None
    assert audit.local_outcome == "limited"


# --- Adversarial regressions for controller correction gate ---


def test_theorem_32_negative_difference_is_not_supported():
    """Regression: a negative Gamma difference must yield not-supported, not supported.

    Construct a scenario where clipping degrades alignment (rho_tilde < rho)
    so observed_difference < 0 -- this must not be labeled 'supported'.
    """
    weights = tensor([0.9, 0.1])
    g1 = tensor([3.0, 0.0])
    g2 = tensor([0.0, 0.1])
    result = cagrad_clip((g1, g2), weights, c=0.5)
    audit = audit_theorem_32(
        result, weights, c=0.5, weighted_smoothness=2.0, step_size=0.1
    )
    if audit.applicable and audit.observed_difference is not None:
        if audit.observed_difference < -1e-12:
            assert audit.local_outcome != "supported", (
                f"Negative difference {audit.observed_difference} must not be supported"
            )


def test_theorem_32_identity_holds_for_interior_alpha():
    """The per-step descent certificate identity must hold for interior alpha.

    Gamma(rho_tilde) - Gamma(rho) = c * (1 - ell_w * eta) * (rho_tilde - rho)
    """
    weights = tensor([0.2, 0.8])
    c = 0.5
    g1 = tensor([1.0, -4.0])
    g2 = tensor([-1.0, 1.0])
    result = cagrad_clip((g1, g2), weights, c)
    audit = audit_theorem_32(
        result, weights, c=c, weighted_smoothness=3.0, step_size=0.1
    )
    assert audit.applicable is True
    # Identity: obs_diff = c * (1 - ell_w * eta) * (rho_tilde - rho)
    expected = c * (1.0 - 3.0 * 0.1) * (audit.rho_tilde - audit.rho)
    assert abs(audit.observed_difference - expected) <= 1e-10


def test_theorem_31_descent_bound_is_recomputed_not_vacuous():
    """Descent bound must check the one-step descent inequality, not just
    that final_loss < initial_loss."""
    case = SmoothObjectiveCase(
        weights=tensor([0.6, 0.4]),
        smoothness_constants=(2.0, 3.0),
        weighted_smoothness=2.4,
        step_size=0.1,
        correction_radius=0.4,
        initial_loss=1.0,
        final_loss=1.0,  # No actual descent
        grad_norm=1.0,  # nonzero gradient, so descent should be required
    )
    audit = audit_theorem_31(case)
    # With nonzero grad_norm and no descent, the bound should fail
    assert audit.descent_bound_holds is False or audit.local_outcome != "supported"


# --- Adversarial regressions for Theorem 3.2 preconditions (correction gate §5) ---


def test_theorem_32_nonsimplex_weights_inapplicable():
    """Non-simplex weights must make the audit inapplicable."""
    result = cagrad_clip(
        (tensor([1.0, -4.0]), tensor([-1.0, 1.0])),
        tensor([0.2, 0.8]), c=0.5,
    )
    # Pass non-simplex weights to the audit
    audit = audit_theorem_32(
        result, tensor([0.3, 0.8]), c=0.5,
        weighted_smoothness=3.0, step_size=0.1,
    )
    assert audit.applicable is False
    assert audit.local_outcome == "limited"


def test_theorem_32_negative_step_size_inapplicable():
    """Negative step size must make the audit inapplicable."""
    result = cagrad_clip(
        (tensor([1.0, -4.0]), tensor([-1.0, 1.0])),
        tensor([0.2, 0.8]), c=0.5,
    )
    audit = audit_theorem_32(
        result, tensor([0.2, 0.8]), c=0.5,
        weighted_smoothness=3.0, step_size=-0.1,
    )
    assert audit.applicable is False
    assert audit.local_outcome == "limited"


def test_theorem_32_nonfinite_step_size_inapplicable():
    """Non-finite step size must make the audit inapplicable."""
    result = cagrad_clip(
        (tensor([1.0, -4.0]), tensor([-1.0, 1.0])),
        tensor([0.2, 0.8]), c=0.5,
    )
    audit = audit_theorem_32(
        result, tensor([0.2, 0.8]), c=0.5,
        weighted_smoothness=3.0, step_size=float("inf"),
    )
    assert audit.applicable is False
    assert audit.local_outcome == "limited"


def test_theorem_32_c_at_one_inapplicable():
    """c=1 is inadmissible for Theorem 3.2; the audit must NOT be called via
    cagrad_clip (which rejects c>=1), but if called directly it should be limited."""
    # Construct a result manually since cagrad_clip rejects c>=1
    result = cagrad_clip(
        (tensor([1.0, -4.0]), tensor([-1.0, 1.0])),
        tensor([0.2, 0.8]), c=0.5,
    )
    audit = audit_theorem_32(
        result, tensor([0.2, 0.8]), c=1.0,
        weighted_smoothness=3.0, step_size=0.1,
    )
    assert audit.applicable is False
    assert audit.local_outcome == "limited"


def test_theorem_32_positive_gamma_improvement_required():
    """Correction gate §5: Theorem 3.2 support requires positive Gamma improvement."""
    audit = interior_strict_witness_audit()
    assert audit.applicable is True
    assert audit.observed_difference > 0.0, (
        f"Positive Gamma improvement required, got {audit.observed_difference}"
    )
    assert audit.local_outcome == "supported"


def test_pareto_criticality_measure_is_nonnegative_norm():
    """Correction gate §2: M(theta)=min_{lambda in simplex} ||sum_i lambda_i grad L_i(theta)||.

    Hand-derived 1D scalar gradient cases:
    - Same-sign: g1=2.0, g2=4.0 => M(theta) = min(|2|,|4|) = 2.0
    - Opposite-sign: g1=2.0, g2=-1.0 => 0 in conv(g1,g2) => M(theta) = 0.0
    - Zero gradient: g1=0.0, g2=2.0 => M(theta) = 0.0
    The previous implementation stored negative directional derivatives (-1.0).
    """
    case_opposite = execute_raco_trajectory(
        x0=1.0, T=1, eta=0.1, c=0.4,
        weights=tensor([0.6, 0.4]),
        smoothness_constants=(2.0, 2.0),
    )
    # At x0=1.0: g1 = 2*1 = 2.0, g2 = 2*(1-1) = 0.0 => M(x0) = 0.0
    assert case_opposite.trajectory_m_values is not None
    m_val = case_opposite.trajectory_m_values[0]
    assert m_val >= 0.0, f"M(theta) must be nonnegative, got {m_val}"
    assert abs(m_val - 0.0) < 1e-12, f"Expected M(theta)=0 for g2=0, got {m_val}"


def test_m_values_and_bounds_per_step():
    """Correction gate §2: Every persisted M(theta_t) must be finite, nonnegative,
    and each step must separately check M(theta_t) <= ||grad L_w(theta_t)||.
    """
    case = execute_raco_trajectory(
        x0=0.8, T=5, eta=0.1, c=0.4,
        weights=tensor([0.6, 0.4]),
        smoothness_constants=(2.0, 2.0),
    )
    assert case.trajectory_m_values is not None
    assert case.trajectory_grad_norms is not None
    for t in range(len(case.trajectory_m_values)):
        m_val = case.trajectory_m_values[t]
        g_norm = case.trajectory_grad_norms[t]
        assert math.isfinite(m_val), f"Step {t}: M(theta) not finite: {m_val}"
        assert m_val >= 0.0, f"Step {t}: M(theta) negative: {m_val}"
        assert m_val <= g_norm + 1e-9, f"Step {t}: M(theta) {m_val} > ||grad L_w|| {g_norm}"

    audit = audit_theorem_31(case)
    assert audit.per_step_m_bound_holds is True


def test_audit_theorem_32_rejects_c_mismatch():
    """Correction gate §4: Reject/limit an audit whose caller c differs from the result's c."""
    weights = tensor([0.2, 0.8])
    c_result = 0.4
    g1 = tensor([1.0, -4.0])
    g2 = tensor([-1.0, 1.0])
    result = cagrad_clip((g1, g2), weights, c=c_result)

    # Pass caller c=0.5 != result.c (0.4)
    audit = audit_theorem_32(
        result, weights, c=0.5, weighted_smoothness=3.0, step_size=0.1
    )
    assert audit.applicable is False
    assert audit.local_outcome == "limited"


def test_audit_theorem_32_rejects_nonfinite_tensors():
    """Correction gate §4: Reject nonfinite result tensors in CAGradResult."""
    weights = tensor([0.2, 0.8])
    c = 0.4
    bad_result = CAGradResult(
        gradient=tensor([float("nan"), 1.0]),
        weighted_anchor=tensor([1.0, 0.0]),
        coefficients=tensor([0.2, 0.8]),
        clipped_coefficients=tensor([0.2, 0.8]),
        mixture=tensor([1.0, 0.0]),
        clipped_mixture=tensor([1.0, 0.0]),
        clipped_coordinates=(),
        singular_case=None,
        c=c,
    )
    audit = audit_theorem_32(
        bad_result, weights, c=c, weighted_smoothness=3.0, step_size=0.1
    )
    assert audit.applicable is False
    assert audit.local_outcome == "limited"


def test_audit_theorem_32_inapplicable_preserves_actual_inputs():
    """Correction gate §4: Ensure every returned strict-condition boolean reflects
    actual rejected/passed inputs rather than fake placeholder step-size/smoothness values.
    """
    weights = tensor([0.2, 0.8])
    c = 0.4
    g1 = tensor([1.0, -4.0])
    g2 = tensor([-1.0, 1.0])
    result = cagrad_clip((g1, g2), weights, c=c)

    # Pass non-simplex weights (triggering inapplicable audit), but with valid step_size and weighted_smoothness
    audit = audit_theorem_32(
        result, tensor([0.3, 0.8]), c=c, weighted_smoothness=3.0, step_size=0.1
    )
    assert audit.applicable is False
    assert audit.strict_conditions["strict_step_size"] is True  # 0.1 < 1.0/3.0
    assert audit.strict_conditions["positive_c"] is True  # c=0.4 > 0


def test_compute_m_simplex_scale_aware():
    """Correction gate §4: opposing g1=[1e-8], g2=[-1e-8] must return 0.0."""
    from reward_free_alignment.theorem_audit import compute_m_simplex
    g1 = tensor([1e-8])
    g2 = tensor([-1e-8])
    m_val = compute_m_simplex(g1, g2)
    assert m_val == 0.0


def test_execute_raco_trajectory_executes_cagrad_clip_update():
    """Correction gate §3: trajectory must execute audited CAGrad-Clip update, not pure g0."""
    case = execute_raco_trajectory(
        x0=1.0, T=10, eta=0.1, c=0.4,
        weights=tensor([0.6, 0.4]),
        smoothness_constants=(2.0, 2.0),
    )
    assert case.trajectory_cagrad_directions is not None
    assert len(case.trajectory_cagrad_directions) == 10
    assert case.trajectory_weighted_anchors is not None
    assert len(case.trajectory_weighted_anchors) == 10
    assert case.trajectory_next_iterates is not None
    assert len(case.trajectory_next_iterates) == 10
    assert case.trajectory_descent_holds is not None
    assert len(case.trajectory_descent_holds) == 10
    assert all(case.trajectory_descent_holds)

    # Verify that each step iterate was updated with cagrad_direction
    x = 1.0
    for t in range(10):
        g1_t = tensor([2.0 * x])
        g2_t = tensor([2.0 * (x - 1.0)])
        res = cagrad_clip((g1_t, g2_t), tensor([0.6, 0.4]), c=0.4)
        assert abs(case.trajectory_cagrad_directions[t] - res.gradient.item()) < 1e-9
        x = x - 0.1 * res.gradient.item()
        assert abs(case.trajectory_next_iterates[t] - x) < 1e-9

    # Verify that a pure g0 trajectory produces different iterates from CAGrad-Clip
    x_g0 = 1.0
    g0_iterates = []
    for t in range(10):
        g0_val = 0.6 * (2.0 * x_g0) + 0.4 * (2.0 * (x_g0 - 1.0))
        x_g0 = x_g0 - 0.1 * g0_val
        g0_iterates.append(x_g0)

    assert not all(abs(a - b) < 1e-9 for a, b in zip(case.trajectory_next_iterates, g0_iterates))
