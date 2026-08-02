import numpy as np
import pytest

from diffusion_gmm_repro.model import IsotropicGMM, forward_gmm, score_at_time
from diffusion_gmm_repro.schedule import paper_schedule


def test_single_gaussian_matches_closed_form_identities() -> None:
    model = IsotropicGMM(
        weights=np.array([3.0]),
        means=np.array([[1.0, -2.0]]),
        variances=np.array([4.0]),
    )
    x = np.array([[1.0, -2.0], [3.0, 0.0]])

    expected_log_density = -np.log(8.0 * np.pi) - np.array([0.0, 1.0])
    np.testing.assert_allclose(model.log_density(x), expected_log_density)
    np.testing.assert_allclose(model.score(x), -(x - model.means[0]) / 4.0)
    np.testing.assert_allclose(model.score_jacobian_trace(x), [-0.5, -0.5])
    np.testing.assert_allclose(model.weights, [1.0])


@pytest.mark.parametrize(
    ("weights", "means", "variances", "message"),
    [
        ([1.0, 1.0], [[0.0, 0.0]], [1.0], "component counts"),
        ([1.0], [0.0, 1.0], [1.0], "two-dimensional"),
        ([1.0], [[0.0, 1.0]], [0.0], "positive"),
        ([1.0, -1.0], [[0.0], [1.0]], [1.0, 1.0], "positive"),
    ],
)
def test_rejects_invalid_mixture_shapes_and_parameters(
    weights: list[float],
    means: list[float] | list[list[float]],
    variances: list[float],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        IsotropicGMM(np.asarray(weights), np.asarray(means), np.asarray(variances))


def test_score_jacobian_trace_matches_finite_difference() -> None:
    model = IsotropicGMM(
        weights=np.array([0.25, 0.75]),
        means=np.array([[-1.0, 0.5], [2.0, -0.25]]),
        variances=np.array([0.7, 1.8]),
    )
    points = np.array([[-0.4, 0.2], [1.1, -0.7]])
    epsilon = 1e-5

    finite_difference_trace = np.zeros(len(points))
    for axis in range(points.shape[1]):
        offset = np.zeros(points.shape[1])
        offset[axis] = epsilon
        score_plus = model.score(points + offset)
        score_minus = model.score(points - offset)
        finite_difference_trace += (score_plus[:, axis] - score_minus[:, axis]) / (
            2.0 * epsilon
        )

    np.testing.assert_allclose(
        model.score_jacobian_trace(points),
        finite_difference_trace,
        rtol=1e-6,
        atol=1e-7,
    )


def test_forward_gmm_scales_means_and_preserves_unit_covariance() -> None:
    base = IsotropicGMM([0.4, 0.6], [[-2.0], [1.0]], [1.0, 1.0])
    schedule = paper_schedule(steps=128)
    noisy = forward_gmm(base, schedule=schedule, time=64)
    np.testing.assert_allclose(
        noisy.means,
        np.sqrt(schedule.alpha_bar[64]) * base.means,
    )
    np.testing.assert_allclose(noisy.variances, np.ones(2))


def test_score_at_time_uses_the_forward_mixture() -> None:
    base = IsotropicGMM([1.0], [[2.0]], [1.0])
    schedule = paper_schedule(steps=128)
    points = np.array([[0.5]])
    noisy = forward_gmm(base, schedule=schedule, time=64)
    np.testing.assert_allclose(
        score_at_time(base, points, schedule=schedule, time=64),
        noisy.score(points),
    )
