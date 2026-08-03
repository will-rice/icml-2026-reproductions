import numpy as np
import pytest

from timerewarder_repro.method import (
    adjacent_rewards,
    inverse_distance_probabilities,
    inverse_support_transform,
    logits_to_scalar,
    scalar_to_two_hot,
    support_transform,
    temporal_distance,
)


def test_temporal_distance_enumerates_106_ordered_pairs() -> None:
    cases = [
        (length, i, j)
        for length in (5, 9)
        for i in range(length)
        for j in range(length)
    ]
    assert len(cases) == 106
    for length, i, j in cases:
        observed = temporal_distance(i, j, length)
        assert isinstance(observed, float)
        np.testing.assert_allclose(
            observed, (j - i) / (length - 1), atol=1e-12, rtol=0.0
        )
        np.testing.assert_allclose(
            observed, -temporal_distance(j, i, length), atol=1e-12, rtol=0.0
        )
        assert -1.0 <= observed <= 1.0


def test_inverse_distance_distribution_is_exactly_normalized() -> None:
    probabilities = inverse_distance_probabilities(8)
    np.testing.assert_allclose(probabilities.sum(), 1.0, atol=1e-12, rtol=0.0)
    np.testing.assert_allclose(
        probabilities / probabilities[0],
        1 / np.arange(1, 9),
        atol=1e-12,
        rtol=0.0,
    )
    assert probabilities.dtype == np.float64


def test_two_hot_mass_and_scalar_round_trip() -> None:
    values = np.linspace(-1.0, 0.98, 100, dtype=np.float64)
    targets = scalar_to_two_hot(values)
    np.testing.assert_allclose(targets.sum(axis=1), 1.0, atol=1e-12, rtol=0.0)
    np.testing.assert_allclose(
        logits_to_scalar(np.log(np.clip(targets, 1e-300, None))).ravel(),
        values,
        atol=1e-12,
        rtol=0.0,
    )
    assert targets.dtype == np.float64


def test_two_hot_upper_endpoint_matches_pinned_clamp_formula() -> None:
    bins = 20
    target = scalar_to_two_hot(np.array([1.0], dtype=np.float64))[0]
    x_min = -(np.sqrt(2.0) - 1.0) - 0.001
    x_max = np.sqrt(2.0) - 1.0 + 0.001
    scale = (x_max - x_min) / (bins - 1)
    position = x_max / scale - 1e-5 - x_min / scale
    low = np.floor(position).astype(np.int64)
    expected = np.zeros(bins, dtype=np.float64)
    expected[low] = np.ceil(position) - position
    expected[low + 1] = position - low
    np.testing.assert_allclose(target, expected, atol=1e-12, rtol=0.0)


def test_method_array_outputs_are_float64() -> None:
    values = np.array([-0.5, 0.5], dtype=np.float64)
    logits = np.zeros((2, 20), dtype=np.float64)
    assert support_transform(values).dtype == np.float64
    assert inverse_support_transform(values).dtype == np.float64
    assert logits_to_scalar(logits).dtype == np.float64
    assert adjacent_rewards(values, values).dtype == np.float64


@pytest.mark.parametrize(
    ("function", "arguments"),
    [
        (support_transform, ([0.0],)),
        (inverse_support_transform, (np.array([0.0], dtype=np.float32),)),
        (scalar_to_two_hot, ([0.0],)),
        (logits_to_scalar, (np.zeros((1, 20), dtype=np.float32),)),
        (
            adjacent_rewards,
            (np.array([0.0], dtype=np.float64), np.array([0.0], dtype=np.float32)),
        ),
    ],
)
def test_method_arrays_require_float64_ndarrays(function, arguments) -> None:
    with pytest.raises(TypeError, match="float64 NumPy array"):
        function(*arguments)


@pytest.mark.parametrize("bins", [1, 0, -1])
def test_two_hot_interfaces_require_at_least_two_bins(bins: int) -> None:
    with pytest.raises(ValueError, match="bins must be at least 2"):
        scalar_to_two_hot(np.array([0.0], dtype=np.float64), bins=bins)
    with pytest.raises(ValueError, match="bins must be at least 2"):
        logits_to_scalar(np.zeros((1, 2), dtype=np.float64), bins=bins)


@pytest.mark.parametrize(
    ("start", "end", "length"),
    [(0, 0, 1), (-1, 0, 5), (0, -1, 5), (5, 0, 5), (0, 5, 5)],
)
def test_temporal_distance_rejects_invalid_bounds(
    start: int, end: int, length: int
) -> None:
    with pytest.raises(ValueError, match="invalid trajectory indices"):
        temporal_distance(start, end, length)


@pytest.mark.parametrize("max_distance", [0, -1])
def test_inverse_distance_rejects_invalid_bounds(max_distance: int) -> None:
    with pytest.raises(ValueError, match="max_distance must be positive"):
        inverse_distance_probabilities(max_distance)


def test_adjacent_rewards_cover_forward_stationary_and_reverse() -> None:
    forward = np.array([0.0, 0.4, 0.2, -0.4], dtype=np.float64)
    reverse = np.array([0.0, -0.4, 0.2, 0.4], dtype=np.float64)
    np.testing.assert_allclose(
        adjacent_rewards(forward, reverse),
        [0.0, 0.8, 0.0, -0.8],
        atol=1e-12,
        rtol=0.0,
    )


@pytest.mark.parametrize(
    ("forward", "reverse"),
    [
        (
            np.array([], dtype=np.float64),
            np.array([], dtype=np.float64),
        ),
        (
            np.array([0.0], dtype=np.float64),
            np.array([0.0, 1.0], dtype=np.float64),
        ),
        (
            np.array([[0.0]], dtype=np.float64),
            np.array([[0.0]], dtype=np.float64),
        ),
    ],
)
def test_adjacent_rewards_require_nonempty_equal_vectors(
    forward: np.ndarray, reverse: np.ndarray
) -> None:
    with pytest.raises(
        ValueError,
        match="forward and reverse scores must be nonempty equal-length vectors",
    ):
        adjacent_rewards(forward, reverse)
