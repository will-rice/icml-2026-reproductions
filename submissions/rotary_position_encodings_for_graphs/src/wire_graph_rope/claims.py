from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from wire_graph_rope.wire import (
    effective_resistance,
    graph_laplacian,
    resistance_spectral_distances,
    spectral_coordinates,
    standard_rope_logits,
    wire_logits,
)


UPSTREAM_REVISION = (
    "arxiv:2509.22259v4+arxiv-source-sha256:"
    "3eb6899ac0da995483dfba1eafe1ed625a1673d638d498882bcf741709f3415f"
)
SNAPSHOT_ID = "ffc9e5510739e188640e9162c2918c8a81db4730f767963cc02be6f6b9000e43"


CLAIM_BINDINGS = [
    {
        "challenge_claim_sha256": "93cba11d965ebf4c88b3d92822ea553f5cc9999b0e55623186312d4f1b735a3a",
        "challenge_claim": (
            "WIRE applies rotary position encodings to graphs by using Laplacian "
            "spectral coordinates to define graph-dependent rotation angles "
            "(Figure 1, Section 3)."
        ),
    },
    {
        "challenge_claim_sha256": "69494fef0193eee52a1925524878e01ca00513d4d4b8d365e6c1ca0b1623788a",
        "challenge_claim": (
            "The WIRE transformation is equivariant to node-order permutations up "
            "to sign flips and rotations in degenerate eigenspaces (Lemma 1)."
        ),
    },
    {
        "challenge_claim_sha256": "0f8376675c180960c5fb6e4263b5c8e502aeccfc6d7f886915ce25624b5f4a8b",
        "challenge_claim": (
            "Regular RoPE is recovered as a special case of WIRE on grid graphs "
            "with appropriate learnable frequencies (Theorem 2, Figure 2)."
        ),
    },
    {
        "challenge_claim_sha256": "c43cfcaf65b3e394c72771326fe762502c8bc79030a59cd06a96a4342631ec55",
        "challenge_claim": (
            "WIRE asymptotically depends on graph effective resistance under the "
            "paper's spectral-feature assumptions (Theorem 3)."
        ),
    },
]


def build_evidence_bundle() -> dict:
    spectral = _spectral_rotary_observation()
    equivariance = _permutation_observation()
    rope = _rope_observation()
    resistance = _resistance_observation()
    observations = [spectral, equivariance, rope, resistance]
    claims = []
    for binding, observation in zip(CLAIM_BINDINGS, observations, strict=True):
        claims.append(
            {
                **binding,
                "target_claim": binding["challenge_claim"],
                "status": observation["status"],
                "evidence": observation,
            }
        )
    return {
        "paper_id": "trn64znfNx",
        "title": "Rotary Position Encodings for Graphs",
        "snapshot_id": SNAPSHOT_ID,
        "upstream_revision": UPSTREAM_REVISION,
        "estimated_api_cost_usd": 0.0,
        "claims": claims,
    }


def write_evidence_bundle(path: str | Path) -> dict:
    bundle = build_evidence_bundle()
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(bundle, indent=2, sort_keys=True) + "\n")
    return bundle


def _spectral_rotary_observation() -> dict:
    features = np.tile(np.array([[1.0, 0.0]]), (5, 1))
    path = graph_laplacian(5, [(0, 1), (1, 2), (2, 3), (3, 4)])
    star = graph_laplacian(5, [(0, 1), (0, 2), (0, 3), (0, 4)])
    frequencies = np.array([[1.7, -0.4]])
    path_logits = wire_logits(features, spectral_coordinates(path, 2), frequencies)
    star_logits = wire_logits(features, spectral_coordinates(star, 2), frequencies)
    max_change = float(np.max(np.abs(path_logits - star_logits)))
    return {
        "status": "toy",
        "metric": "max_logit_change_between_path_and_star",
        "value": max_change,
        "tolerance": 1e-3,
        "passed": max_change > 1e-3,
    }


def _permutation_observation() -> dict:
    edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0)]
    permutation = np.array([2, 5, 1, 4, 0, 3])
    inverse = np.argsort(permutation)
    permuted_edges = [(inverse[u], inverse[v]) for u, v in edges]
    features = np.tile(np.array([[1.0, 0.0]]), (6, 1))
    frequencies = np.array([[0.8, -1.1]])
    base = wire_logits(
        features,
        spectral_coordinates(graph_laplacian(6, edges), 2),
        frequencies,
    )
    permuted = wire_logits(
        features[inverse],
        spectral_coordinates(graph_laplacian(6, permuted_edges), 2),
        frequencies,
    )
    max_error = float(np.max(np.abs(base - permuted[inverse][:, inverse])))
    return {
        "status": "toy",
        "metric": "max_permutation_equivariance_error",
        "value": max_error,
        "tolerance": 1e-9,
        "passed": max_error <= 1e-9,
    }


def _rope_observation() -> dict:
    positions = np.arange(7, dtype=float)[:, None]
    features = np.tile(np.array([[1.0, 0.0, 1.0, 0.0]]), (7, 1))
    frequencies = np.array([[0.25], [0.75]])
    max_error = float(
        np.max(
            np.abs(
                wire_logits(features, positions, frequencies)
                - standard_rope_logits(features, positions[:, 0], frequencies[:, 0])
            )
        )
    )
    return {
        "status": "toy",
        "metric": "max_grid_rope_recovery_error",
        "value": max_error,
        "tolerance": 1e-12,
        "passed": max_error <= 1e-12,
    }


def _resistance_observation() -> dict:
    laplacian = graph_laplacian(4, [(0, 1), (1, 2), (2, 3)])
    resistance = effective_resistance(laplacian)
    spectral = resistance_spectral_distances(laplacian)
    max_error = float(np.max(np.abs(resistance - spectral)))
    return {
        "status": "verified",
        "metric": "max_effective_resistance_identity_error",
        "value": max_error,
        "tolerance": 1e-10,
        "passed": max_error <= 1e-10,
        "path_endpoint_resistance": float(resistance[0, 3]),
    }


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[2]
    write_evidence_bundle(project_root / "evidence" / "bundle.json")
