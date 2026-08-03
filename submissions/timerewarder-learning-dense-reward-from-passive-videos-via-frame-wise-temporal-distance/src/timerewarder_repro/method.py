import numpy as np


def temporal_distance(start: int, end: int, trajectory_length: int) -> float:
    if (
        trajectory_length < 2
        or not 0 <= start < trajectory_length
        or not 0 <= end < trajectory_length
    ):
        raise ValueError("invalid trajectory indices")
    return (end - start) / (trajectory_length - 1)


def inverse_distance_probabilities(max_distance: int) -> np.ndarray:
    if max_distance < 1:
        raise ValueError("max_distance must be positive")
    weights = 1.0 / np.arange(1, max_distance + 1, dtype=np.float64)
    return weights / weights.sum(dtype=np.float64)


def support_transform(values: np.ndarray) -> np.ndarray:
    values = _require_float64_array(values)
    return np.sign(values) * (np.sqrt(np.abs(values) + 1.0) - 1.0) + 0.001 * values


def inverse_support_transform(values: np.ndarray) -> np.ndarray:
    values = _require_float64_array(values)
    absolute = ((np.sqrt(1 + 0.004 * (np.abs(values) + 1.001)) - 1) / 0.002) ** 2 - 1
    return np.sign(values) * absolute


def scalar_to_two_hot(values: np.ndarray, bins: int = 20) -> np.ndarray:
    if bins < 2:
        raise ValueError("bins must be at least 2")
    values = _require_float64_array(values)
    transformed = support_transform(values)
    support = np.linspace(
        support_transform(np.array(-1.0, dtype=np.float64)),
        support_transform(np.array(1.0, dtype=np.float64)),
        bins,
    )
    scale = support[1] - support[0]
    position = np.clip(
        transformed / scale, support[0] / scale, support[-1] / scale - 1e-5
    )
    position -= support[0] / scale
    low = np.floor(position).astype(np.int64)
    high = np.ceil(position).astype(np.int64)
    fraction = position - low
    result = np.zeros((position.size, bins), dtype=np.float64)
    rows = np.arange(position.size)
    result[rows, low.ravel()] += (1.0 - fraction).ravel()
    result[rows, high.ravel()] += fraction.ravel()
    return result.reshape(*position.shape, bins)


def logits_to_scalar(logits: np.ndarray, bins: int = 20) -> np.ndarray:
    if bins < 2:
        raise ValueError("bins must be at least 2")
    logits = _require_float64_array(logits)
    probabilities = np.exp(logits - logits.max(axis=-1, keepdims=True))
    probabilities /= probabilities.sum(axis=-1, keepdims=True)
    support = np.linspace(
        support_transform(np.array(-1.0, dtype=np.float64)),
        support_transform(np.array(1.0, dtype=np.float64)),
        bins,
    )
    return inverse_support_transform(
        (probabilities * support).sum(axis=-1, keepdims=True)
    )


def adjacent_rewards(forward: np.ndarray, reverse: np.ndarray) -> np.ndarray:
    forward = _require_float64_array(forward)
    reverse = _require_float64_array(reverse)
    if forward.shape != reverse.shape or forward.ndim != 1 or forward.size == 0:
        raise ValueError(
            "forward and reverse scores must be nonempty equal-length vectors"
        )
    reward = forward - reverse
    reward[0] = 0.0
    return reward


def _require_float64_array(values: np.ndarray) -> np.ndarray:
    if not isinstance(values, np.ndarray) or values.dtype != np.float64:
        raise TypeError("values must be a float64 NumPy array")
    return values
