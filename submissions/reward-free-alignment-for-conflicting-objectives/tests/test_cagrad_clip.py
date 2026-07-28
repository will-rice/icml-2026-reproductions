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


def test_singular_case_zero_anchor_returns_zero_gradient_and_user_weights():
    """Correction gate §5: g0=0 chooses w, uses effective c=0, returns zero update."""
    g1, g2 = tensor([1.0, 0.0]), tensor([-1.0, 0.0])
    weights = tensor([0.5, 0.5])
    result = cagrad_clip((g1, g2), weights, c=0.4)
    assert result.singular_case == "zero_anchor"
    assert torch.allclose(result.coefficients, weights)
    assert torch.allclose(result.gradient, tensor([0.0, 0.0]))
    assert result.c == 0.0


def test_singular_case_identical_gradients_returns_user_weights():
    """Correction gate §3: Identical gradients choose user weight w."""
    g1, g2 = tensor([1.0, 2.0]), tensor([1.0, 2.0])
    weights = tensor([0.7, 0.3])
    result = cagrad_clip((g1, g2), weights, c=0.4)
    assert result.singular_case == "identical_gradients"
    assert torch.allclose(result.coefficients, weights)


def test_identical_gradients_subproblem_objective_includes_radius_term():
    """Correction gate §5: identical gradients must include c*||g0||*||g|| in objective_value."""
    g1, g2 = tensor([2.0, 3.0]), tensor([2.0, 3.0])
    weights = tensor([0.7, 0.3])
    c_val = 0.4
    solution = solve_two_objective_alpha(g1, g2, weights, c=c_val)
    g0 = weights[0] * g1 + weights[1] * g2
    norm_g0 = torch.linalg.vector_norm(g0).item()
    norm_g1 = torch.linalg.vector_norm(g1).item()
    expected_obj = torch.dot(g1, g0).item() + c_val * norm_g0 * norm_g1
    assert abs(solution.objective_value - expected_obj) < 1e-9



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


# --- Adversarial regressions for corrected stationary quadratic ---


def _grid_search_minimizer(g1, g2, g0, c, n=10_000):
    """Independent brute-force h minimizer for comparison."""
    s = c * torch.linalg.vector_norm(g0).item()
    best_alpha, best_h = 0.0, float("inf")
    for i in range(n + 1):
        a = i / n
        mix = a * g1 + (1.0 - a) * g2
        hv = (torch.dot(mix, g0) + s * torch.linalg.vector_norm(mix)).item()
        if hv < best_h:
            best_h = hv
            best_alpha = a
    return best_alpha, best_h


def test_solver_matches_independent_minimizer_plan_witness():
    """Regress the plan's exact witness: alpha≈0.356145, h≈0.422241.

    Previous buggy formula (-2*s²*q2*q1) gave alpha=1.0, h=0.636932.
    Corrected formula (-s²*q1*q2) gives alpha≈0.356145, h≈0.422241.
    """
    g1 = tensor([1.0, -4.0])
    g2 = tensor([-1.0, 1.0])
    weights = tensor([0.2, 0.8])
    c = 0.5
    g0 = weights[0] * g1 + weights[1] * g2

    solution = solve_two_objective_alpha(g1, g2, weights, c)
    grid_alpha, grid_h = _grid_search_minimizer(g1, g2, g0, c)

    assert abs(solution.objective_value - grid_h) < 1e-3, (
        f"Solver h={solution.objective_value:.6f} != grid h={grid_h:.6f}"
    )
    assert abs(solution.alpha - 0.356145) < 1e-3, (
        f"Expected interior alpha≈0.356145, got {solution.alpha}"
    )
    assert solution.singular_case is None


def test_solver_matches_grid_search_seeded_property():
    """Property test: solver h must match grid-search h for seeded random cases."""
    rng = torch.Generator().manual_seed(42)
    for trial in range(20):
        g1 = torch.randn(4, generator=rng)
        g2 = torch.randn(4, generator=rng)
        w1 = torch.rand(1, generator=rng).item()
        weights = tensor([w1, 1.0 - w1])
        c = 0.3 + 0.4 * torch.rand(1, generator=rng).item()
        g0 = weights[0] * g1 + weights[1] * g2
        solution = solve_two_objective_alpha(g1, g2, weights, c)
        _, grid_h = _grid_search_minimizer(g1, g2, g0, c, n=5_000)
        assert abs(solution.objective_value - grid_h) < 1e-2, (
            f"Trial {trial}: solver h={solution.objective_value:.6f} != grid h={grid_h:.6f}"
        )


def test_delta_b_zero_finds_interior_minimum():
    """When delta_b=0, h(alpha) = const + s*||mix||, minimized at
    alpha = -q1/(2*q2), not at a boundary."""
    g1 = tensor([1.0, 0.0])
    g2 = tensor([0.0, 2.0])
    weights = tensor([0.8, 0.2])
    c = 0.4
    solution = solve_two_objective_alpha(g1, g2, weights, c)
    assert abs(solution.alpha - 0.8) < 1e-5, (
        f"Expected interior alpha=0.8, got {solution.alpha}"
    )


# --- Scale-invariance regressions (controller correction gate §1) ---


@pytest.mark.parametrize("scale", [1e-8, 1e-6, 1e-4, 1e-2, 1.0, 1e2, 1e4, 1e6, 1e8])
def test_scale_invariance_plan_witness(scale):
    """The plan witness must find alpha≈0.356145 at every scale from 1e-8 to 1e8.

    Previous bug: absolute thresholds (1e-14, 1e-5) caused the solver to miss
    interior solutions at small or large scales.
    """
    g1 = scale * tensor([1.0, -4.0])
    g2 = scale * tensor([-1.0, 1.0])
    weights = tensor([0.2, 0.8])
    c = 0.5
    g0 = weights[0] * g1 + weights[1] * g2
    solution = solve_two_objective_alpha(g1, g2, weights, c)
    _, grid_h = _grid_search_minimizer(g1, g2, g0, c, n=50_000)
    assert abs(solution.objective_value - grid_h) < 1e-2 * abs(grid_h) + 1e-15, (
        f"scale={scale}: solver h={solution.objective_value:.6e} != grid h={grid_h:.6e}"
    )
    assert abs(solution.alpha - 0.356145) < 1e-3, (
        f"scale={scale}: expected alpha≈0.356145, got {solution.alpha}"
    )
    assert solution.singular_case is None


def test_wide_log_scale_random_property():
    """Property test: solver h must match grid-search h across wide-log-scale
    random gradients as required by correction gate §1."""
    rng = torch.Generator().manual_seed(2026)
    for trial in range(30):
        log_scale = -8.0 + 16.0 * torch.rand(1, generator=rng).item()
        scale = 10.0 ** log_scale
        g1 = scale * torch.randn(4, generator=rng)
        g2 = scale * torch.randn(4, generator=rng)
        w1 = torch.rand(1, generator=rng).item()
        w1 = max(0.01, min(0.99, w1))
        weights = tensor([w1, 1.0 - w1])
        c = 0.3 + 0.4 * torch.rand(1, generator=rng).item()
        g0 = weights[0] * g1 + weights[1] * g2
        solution = solve_two_objective_alpha(g1, g2, weights, c)
        _, grid_h = _grid_search_minimizer(g1, g2, g0, c, n=5_000)
        assert abs(solution.objective_value - grid_h) < 1e-2 * abs(grid_h) + 1e-15, (
            f"Trial {trial} (scale={scale:.1e}): "
            f"solver h={solution.objective_value:.6e} != grid h={grid_h:.6e}"
        )


def test_nonfinite_gradients_rejected():
    """Non-finite gradients must be rejected, not silently produce wrong answers."""
    import math
    for bad_val in [float("nan"), float("inf"), float("-inf")]:
        g1 = tensor([bad_val, 1.0])
        g2 = tensor([1.0, 0.0])
        with pytest.raises(ValueError, match="finite"):
            solve_two_objective_alpha(g1, g2, tensor([0.5, 0.5]), c=0.4)
        with pytest.raises(ValueError, match="finite"):
            cagrad_clip((g1, g2), tensor([0.5, 0.5]), c=0.4)


def test_nonfinite_weights_rejected():
    """Non-finite weights must be rejected."""
    g1 = tensor([1.0, 0.0])
    g2 = tensor([0.0, 1.0])
    with pytest.raises(ValueError, match="finite"):
        solve_two_objective_alpha(g1, g2, tensor([float("nan"), 0.5]), c=0.4)
