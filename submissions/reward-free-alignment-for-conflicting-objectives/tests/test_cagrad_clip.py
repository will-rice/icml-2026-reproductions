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
