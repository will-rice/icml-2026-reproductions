from __future__ import annotations

import numpy as np


def graph_laplacian(node_count: int, edges: list[tuple[int, int]]) -> np.ndarray:
    if node_count <= 0:
        raise ValueError("node_count")
    adjacency = np.zeros((node_count, node_count), dtype=float)
    for left, right in edges:
        if left == right or not (0 <= left < node_count) or not (0 <= right < node_count):
            raise ValueError("edge")
        adjacency[left, right] = 1.0
        adjacency[right, left] = 1.0
    degree = np.diag(adjacency.sum(axis=1))
    return degree - adjacency


def spectral_coordinates(laplacian: np.ndarray, dimensions: int) -> np.ndarray:
    laplacian = _square_array(laplacian)
    if dimensions <= 0 or dimensions >= laplacian.shape[0]:
        raise ValueError("dimensions")
    values, vectors = np.linalg.eigh(laplacian)
    order = np.argsort(values)
    values = values[order]
    vectors = vectors[:, order]
    nonzero = np.flatnonzero(values > 1e-10)
    if len(nonzero) < dimensions:
        raise ValueError("dimensions")
    coords = vectors[:, nonzero[:dimensions]].copy()
    return _canonicalize_columns(coords)


def wire_logits(
    features: np.ndarray,
    coordinates: np.ndarray,
    frequencies: np.ndarray,
) -> np.ndarray:
    features = np.asarray(features, dtype=float)
    coordinates = np.asarray(coordinates, dtype=float)
    frequencies = np.asarray(frequencies, dtype=float)
    if features.ndim != 2 or features.shape[1] % 2 != 0:
        raise ValueError("features")
    if coordinates.ndim != 2 or coordinates.shape[0] != features.shape[0]:
        raise ValueError("coordinates")
    if frequencies.ndim != 2 or frequencies.shape[1] != coordinates.shape[1]:
        raise ValueError("frequencies")
    block_count = features.shape[1] // 2
    if frequencies.shape[0] != block_count:
        raise ValueError("frequencies")

    pairwise_distance = np.linalg.norm(
        coordinates[:, None, :] - coordinates[None, :, :], axis=-1
    )
    logits = np.zeros((features.shape[0], features.shape[0]), dtype=float)
    for block in range(block_count):
        left = features[:, 2 * block : 2 * block + 2]
        base_inner = left @ left.T
        freq_norm = float(np.linalg.norm(frequencies[block]))
        logits += base_inner * np.cos(freq_norm * pairwise_distance)
    return logits


def standard_rope_logits(
    features: np.ndarray,
    positions: np.ndarray,
    frequencies: np.ndarray,
) -> np.ndarray:
    features = np.asarray(features, dtype=float)
    positions = np.asarray(positions, dtype=float)
    frequencies = np.asarray(frequencies, dtype=float)
    if features.ndim != 2 or features.shape[1] != 2 * len(frequencies):
        raise ValueError("features")
    if positions.ndim != 1 or len(positions) != features.shape[0]:
        raise ValueError("positions")
    logits = np.zeros((features.shape[0], features.shape[0]), dtype=float)
    relative = positions[:, None] - positions[None, :]
    for block, frequency in enumerate(frequencies):
        left = features[:, 2 * block : 2 * block + 2]
        logits += (left @ left.T) * np.cos(float(frequency) * relative)
    return logits


def effective_resistance(laplacian: np.ndarray) -> np.ndarray:
    laplacian = _square_array(laplacian)
    pseudoinverse = np.linalg.pinv(laplacian, hermitian=True)
    diagonal = np.diag(pseudoinverse)
    resistance = diagonal[:, None] + diagonal[None, :] - 2.0 * pseudoinverse
    resistance[np.abs(resistance) < 1e-12] = 0.0
    return resistance


def resistance_spectral_distances(laplacian: np.ndarray) -> np.ndarray:
    laplacian = _square_array(laplacian)
    values, vectors = np.linalg.eigh(laplacian)
    nonzero = values > 1e-10
    scaled = vectors[:, nonzero] / np.sqrt(values[nonzero])
    diff = scaled[:, None, :] - scaled[None, :, :]
    distances = np.sum(diff * diff, axis=-1)
    distances[np.abs(distances) < 1e-12] = 0.0
    return distances


def _square_array(value: np.ndarray) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.ndim != 2 or array.shape[0] != array.shape[1]:
        raise ValueError("square")
    if not np.allclose(array, array.T, atol=1e-12):
        raise ValueError("symmetric")
    return array


def _canonicalize_columns(coords: np.ndarray) -> np.ndarray:
    canonical = coords.copy()
    for column in range(canonical.shape[1]):
        values = canonical[:, column]
        pivot = int(np.argmax(np.abs(values)))
        if values[pivot] < 0:
            canonical[:, column] *= -1.0
    return canonical
