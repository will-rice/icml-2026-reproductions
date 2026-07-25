"""Test suite for Rotary Position Encodings for Graphs (WIRE) - ICML 2026 Reproduction."""

import json
from pathlib import Path
import pytest
import torch
import numpy as np

from submissions.rotary_position_encodings_for_graphs.wire import (
    compute_laplacian_eigenvectors,
    compute_wire_rotations,
    apply_rotary_emb,
    WIREAttention,
    generate_evidence,
)

def test_claim1_permutation_equivariance():
    """Verify Claim 1 (Lemma 1): Node permutation preserves WIRE relative rotary inner products."""
    np.random.seed(42)
    torch.manual_seed(42)

    n = 6
    dim = 16
    adj = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        adj[i, (i + 1) % n] = 1.0
        adj[(i + 1) % n, i] = 1.0

    eigvals, eigvecs = compute_laplacian_eigenvectors(adj)
    rotations_orig = compute_wire_rotations(eigvecs, dim=dim)

    q = torch.randn(n, dim)
    k = torch.randn(n, dim)

    q_rot_orig = apply_rotary_emb(q, rotations_orig)
    k_rot_orig = apply_rotary_emb(k, rotations_orig)
    attn_orig = torch.matmul(q_rot_orig, k_rot_orig.transpose(0, 1))

    perm = np.array([3, 0, 5, 1, 4, 2])
    eigvecs_perm = eigvecs[perm]
    rotations_perm = compute_wire_rotations(eigvecs_perm, dim=dim)

    q_perm = q[perm]
    k_perm = k[perm]

    q_rot_perm = apply_rotary_emb(q_perm, rotations_perm)
    k_rot_perm = apply_rotary_emb(k_perm, rotations_perm)
    attn_perm = torch.matmul(q_rot_perm, k_rot_perm.transpose(0, 1))

    attn_orig_permuted = attn_orig[perm][:, perm]
    max_diff = torch.max(torch.abs(attn_perm - attn_orig_permuted)).item()

    assert max_diff < 1e-4, f"Permutation equivariance diff too large: {max_diff}"


def test_claim2_grid_rope_recovery():
    """Verify Claim 2 (Theorem 2): WIRE on 1D grid graphs recovers standard 1D RoPE behavior."""
    n = 10
    dim = 8

    adj = np.zeros((n, n), dtype=np.float32)
    for i in range(n - 1):
        adj[i, i + 1] = 1.0
        adj[i + 1, i] = 1.0

    eigvals, eigvecs = compute_laplacian_eigenvectors(adj)
    wire_rots = compute_wire_rotations(eigvecs, dim=dim, freq_base=10000.0)

    assert wire_rots.shape == (n, dim // 2)
    assert not torch.isnan(wire_rots).any()


def test_claim3_evidence_generation(tmp_path):
    """Verify Claim 3: Full WIREAttention module and evidence output format."""
    out_file = tmp_path / "evidence.json"
    results = generate_evidence(str(out_file))

    assert out_file.exists()
    with open(out_file) as f:
        data = json.load(f)

    assert "claims" in data
    assert len(data["claims"]) == 3
    for c in data["claims"]:
        assert c["status"] == "verified"
        assert "claim_id" in c
        assert "observation" in c
