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


def strict_witness_audit() -> DescentCertificateAudit:
    weights = tensor([0.2, 0.8])
    c = 0.5
    # g1 and g2 chosen such that alpha is in (0, 1) and coefficients differ from weights
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


def test_strictness_requires_every_paper_condition():
    audit = strict_witness_audit()
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


def test_zero_anchor_is_not_applicable_not_divided():
    audit = zero_anchor_audit()
    assert audit.applicable is False
    assert audit.rho is None and audit.rho_tilde is None
    assert audit.local_outcome == "limited"
