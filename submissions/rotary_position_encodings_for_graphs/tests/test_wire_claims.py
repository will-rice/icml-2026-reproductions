import json
from pathlib import Path

import numpy as np


def test_wire_spectral_rotary_logits_are_deterministic_and_graph_sensitive():
    from wire_graph_rope.claims import build_evidence_bundle
    from wire_graph_rope.wire import graph_laplacian, spectral_coordinates, wire_logits

    features = np.tile(np.array([[1.0, 0.0]]), (5, 1))
    path = graph_laplacian(5, [(0, 1), (1, 2), (2, 3), (3, 4)])
    star = graph_laplacian(5, [(0, 1), (0, 2), (0, 3), (0, 4)])
    frequencies = np.array([[1.7, -0.4]])

    path_logits = wire_logits(features, spectral_coordinates(path, dimensions=2), frequencies)
    repeated_path_logits = wire_logits(
        features, spectral_coordinates(path, dimensions=2), frequencies
    )
    star_logits = wire_logits(features, spectral_coordinates(star, dimensions=2), frequencies)

    assert np.allclose(path_logits, repeated_path_logits, atol=1e-12)
    assert not np.allclose(path_logits, star_logits, atol=1e-3)

    bundle = build_evidence_bundle()
    assert bundle["paper_id"] == "trn64znfNx"
    assert bundle["claims"][0]["challenge_claim_sha256"] == (
        "93cba11d965ebf4c88b3d92822ea553f5cc9999b0e55623186312d4f1b735a3a"
    )


def test_wire_logits_are_equivariant_to_node_permutation_on_cycle_graphs():
    from wire_graph_rope.wire import graph_laplacian, spectral_coordinates, wire_logits

    edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0)]
    permutation = np.array([2, 5, 1, 4, 0, 3])
    inverse = np.argsort(permutation)
    permuted_edges = [(inverse[u], inverse[v]) for u, v in edges]
    features = np.tile(np.array([[1.0, 0.0]]), (6, 1))
    frequencies = np.array([[0.8, -1.1]])

    base = wire_logits(
        features,
        spectral_coordinates(graph_laplacian(6, edges), dimensions=2),
        frequencies,
    )
    permuted = wire_logits(
        features[inverse],
        spectral_coordinates(graph_laplacian(6, permuted_edges), dimensions=2),
        frequencies,
    )

    assert np.allclose(base, permuted[permutation][:, permutation], atol=1e-9)


def test_wire_recovers_standard_rope_when_grid_coordinates_are_used():
    from wire_graph_rope.wire import standard_rope_logits, wire_logits

    positions = np.arange(7, dtype=float)[:, None]
    features = np.tile(np.array([[1.0, 0.0, 1.0, 0.0]]), (7, 1))
    frequencies = np.array([[0.25], [0.75]])

    wire = wire_logits(features, positions, frequencies)
    rope = standard_rope_logits(features, positions[:, 0], frequencies[:, 0])

    assert np.allclose(wire, rope, atol=1e-12)


def test_effective_resistance_matches_scaled_spectral_feature_distances():
    from wire_graph_rope.wire import (
        effective_resistance,
        graph_laplacian,
        resistance_spectral_distances,
    )

    laplacian = graph_laplacian(4, [(0, 1), (1, 2), (2, 3)])

    resistance = effective_resistance(laplacian)
    spectral_distances = resistance_spectral_distances(laplacian)

    assert np.allclose(resistance, spectral_distances, atol=1e-10)
    assert np.isclose(resistance[0, 3], 3.0, atol=1e-10)


def test_evidence_bundle_is_written_with_bound_claim_results(tmp_path):
    from wire_graph_rope.claims import build_evidence_bundle, write_evidence_bundle

    output = tmp_path / "bundle.json"
    bundle = write_evidence_bundle(output)

    assert bundle == build_evidence_bundle()
    assert json.loads(Path(output).read_text()) == bundle
    assert {claim["status"] for claim in bundle["claims"]} <= {
        "verified",
        "toy",
        "falsified",
        "inconclusive",
    }
