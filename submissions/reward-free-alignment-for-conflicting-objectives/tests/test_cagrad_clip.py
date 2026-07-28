import pytest
import torch
from torch import tensor
from reward_free_alignment.cagrad_clip import (
    solve_two_objective_alpha,
    cagrad_clip,
    AlphaSolution,
    CAGradResult,
)


def test_solver_uses_asymmetric_user_weight_anchor():
    g1, g2 = tensor([1.0, 0.0]), tensor([0.0, 2.0])
    weights = tensor([0.8, 0.2])
    solution = solve_two_objective_alpha(g1, g2, weights, c=0.4)
    assert torch.allclose(solution.weighted_anchor, tensor([0.8, 0.4]))
    assert 0.0 <= solution.alpha <= 1.0
    assert torch.allclose(
        solution.coefficients, tensor([solution.alpha, 1.0 - solution.alpha])
    )


def test_clipping_is_coordinatewise_and_not_renormalized():
    result = cagrad_clip(
        (tensor([1.0, -1.0]), tensor([-0.2, 1.0])),
        weights=tensor([0.1, 0.9]),
        c=0.5,
    )
    expected = torch.minimum(result.coefficients, tensor([0.1, 0.9]))
    assert torch.allclose(result.clipped_coefficients, expected)
    assert result.clipped_coefficients.sum() <= 1.0
    assert torch.allclose(
        result.clipped_mixture,
        expected[0] * tensor([1.0, -1.0]) + expected[1] * tensor([-0.2, 1.0]),
    )


@pytest.mark.parametrize(
    ("g1", "g2", "weights", "c", "case"),
    [
        ([1.0, 2.0], [1.0, 2.0], [0.7, 0.3], 0.4, "identical_gradients"),
        ([1.0, 0.0], [-1.0, 0.0], [0.5, 0.5], 0.4, "zero_anchor"),
        ([0.0, 0.0], [0.0, 0.0], [0.2, 0.8], 0.4, "zero_anchor"),
        ([1.0, 2.0], [2.0, 4.0], [0.5, 0.5], 0.4, "colinear_gradients"),
        ([1.0, 0.0], [0.0, 1.0], [0.6, 0.4], 0.0, "zero_radius"),
    ],
)
def test_singular_cases_are_finite_and_deterministic(g1, g2, weights, c, case):
    first = cagrad_clip((tensor(g1), tensor(g2)), tensor(weights), c)
    second = cagrad_clip((tensor(g1), tensor(g2)), tensor(weights), c)
    assert torch.isfinite(first.gradient).all()
    assert torch.equal(first.gradient, second.gradient)
    assert first.singular_case == case


def test_zero_weight_coordinate_cannot_be_reintroduced():
    result = cagrad_clip(
        (tensor([1.0, 0.0]), tensor([0.0, 1.0])),
        weights=tensor([1.0, 0.0]),
        c=0.4,
    )
    assert result.clipped_coefficients[1].item() == 0.0


# --- Adversarial regressions for controller correction gate ---


def test_zero_radius_minimizes_h_not_shortcut_to_w():
    """c=0 regression: h(alpha) = alpha*dot(g1,g0) + (1-alpha)*dot(g2,g0).

    With g1=[1,0], g2=[0,1], w=[0.6,0.4]: g0=[0.6,0.4].
    h(alpha) = 0.4 + 0.2*alpha, minimized at alpha=0.
    Previous bug returned alpha=w1=0.6.
    """
    g1, g2 = tensor([1.0, 0.0]), tensor([0.0, 1.0])
    weights = tensor([0.6, 0.4])
    solution = solve_two_objective_alpha(g1, g2, weights, c=0.0)
    assert abs(solution.alpha - 0.0) < 1e-9, (
        f"c=0: expected alpha=0 but got {solution.alpha}"
    )
    assert solution.singular_case == "zero_radius"


def test_colinear_minimizes_h_not_shortcut_to_w():
    """Colinear regression: g2=2*g1 with w=[0.5,0.5].

    g0 = 0.5*g1 + 0.5*2*g1 = 1.5*g1; mix = (2-alpha)*g1.
    h(alpha) = (2-alpha)*(dot(g1,g0) + c*||g0||*||g1||), minimized at alpha=1.
    Previous bug returned alpha=w1=0.5.
    """
    g1, g2 = tensor([1.0, 2.0]), tensor([2.0, 4.0])
    weights = tensor([0.5, 0.5])
    solution = solve_two_objective_alpha(g1, g2, weights, c=0.4)
    assert abs(solution.alpha - 1.0) < 1e-9, (
        f"colinear: expected alpha=1 but got {solution.alpha}"
    )
    assert solution.singular_case == "colinear_gradients"


def test_c_equals_one_is_rejected():
    """c=1 is forbidden because Theorem 3.1 requires 0 <= c < 1."""
    with pytest.raises(ValueError, match="c"):
        solve_two_objective_alpha(
            tensor([1.0, 0.0]), tensor([0.0, 1.0]),
            tensor([0.5, 0.5]), c=1.0,
        )


def test_c_greater_than_one_is_rejected():
    """c>1 is also forbidden."""
    with pytest.raises(ValueError, match="c"):
        solve_two_objective_alpha(
            tensor([1.0, 0.0]), tensor([0.0, 1.0]),
            tensor([0.5, 0.5]), c=1.5,
        )
