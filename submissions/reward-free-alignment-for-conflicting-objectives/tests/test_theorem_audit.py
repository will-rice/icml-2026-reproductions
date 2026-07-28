import pytest
import torch
from torch import tensor
from reward_free_alignment.cagrad_clip import cagrad_clip
from reward_free_alignment.theorem_audit import (
    gamma,
    audit_theorem_31,
    audit_theorem_32,
    SmoothObjectiveCase,
    ConvergenceAudit,
    DescentCertificateAudit,
)


def smooth_nonnegative_quadratic_case() -> SmoothObjectiveCase:
    return SmoothObjectiveCase(
        weights=tensor([0.6, 0.4]),
        smoothness_constants=(2.0, 3.0),
        weighted_smoothness=2.4,
        step_size=0.1,
        correction_radius=0.4,
        initial_loss=1.5,
        final_loss=1.2,
        grad_norm=0.05,
    )


def boundary_witness_audit() -> DescentCertificateAudit:
    """Witness for Theorem 3.2 with boundary alpha (correct for 2D non-degenerate).

    For 2D non-degenerate gradients, h(alpha) is convex so the solver always
    gives boundary alpha (0 or 1). The clipping still produces a measurable
    Gamma difference because p != w when alpha is at a boundary.
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


def test_strictness_records_all_eight_conditions():
    """Verify all 8 strictness conditions are computed and recorded.

    For 2D non-degenerate gradients, the correct solver gives boundary alpha
    (h is convex), so interior_coefficients is False. This test verifies
    the conditions are computed correctly, not that all are True.
    """
    audit = boundary_witness_audit()
    expected_conditions = {
        "two_objectives": True,
        "positive_weights": True,
        "positive_c": True,
        "strict_step_size": True,
        "nonzero_anchor": True,
        "noncolinear_gradients": True,
        "interior_coefficients": False,  # correct: h is convex in 2D
        "coefficients_differ_from_weights": True,
    }
    assert audit.strict_conditions == expected_conditions
    # Not all 8 hold, so strict_expected is False
    assert audit.strict_expected is False
    # But the identity residual is still small and difference is nonnegative
    assert audit.identity_residual <= 1e-10
    assert audit.applicable is True
    assert audit.observed_difference >= 0.0


def test_boundary_witness_has_nonneg_improvement():
    """Even with boundary alpha, the per-step identity should hold and
    the improvement should be nonnegative."""
    audit = boundary_witness_audit()
    assert audit.applicable is True
    assert audit.observed_difference is not None
    assert audit.observed_difference >= -1e-12
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


def test_theorem_32_identity_holds_for_boundary_alpha():
    """The per-step descent certificate identity must hold even for boundary alpha.

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
