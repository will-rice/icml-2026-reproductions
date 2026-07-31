from __future__ import annotations

from itertools import combinations
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from top_w_repro.decoder import (
    brute_force_subset_update,
    evaluate_decoding_metrics,
    min_p_filter,
    nearest_set_potential,
    pairwise_cosine_distance,
    prefix_subset_update,
    subset_objective,
    top_h_filter,
    top_p_filter,
    top_w_mask,
    whiten_and_normalize,
)
from top_w_repro.upstream import load_upstream_module


def test_pairwise_cosine_distance_is_a_pseudometric():
    torch.manual_seed(0)
    embeddings = torch.nn.functional.normalize(torch.randn(12, 8), dim=-1)
    dist = pairwise_cosine_distance(embeddings)
    assert dist.shape == (12, 12)
    assert torch.allclose(torch.diag(dist), torch.zeros(12), atol=1e-6)
    assert torch.allclose(dist, dist.t(), atol=1e-6)
    assert bool((dist >= -1e-6).all())


def test_nearest_set_potential_is_zero_inside_the_kept_set():
    torch.manual_seed(1)
    embeddings = torch.nn.functional.normalize(torch.randn(10, 6), dim=-1)
    kept = torch.tensor([2, 5])
    potential = nearest_set_potential(embeddings, kept)
    assert potential.shape == (10,)
    assert torch.allclose(potential[kept], torch.zeros(2), atol=1e-6)
    assert bool((potential <= 1e-6).all())


@pytest.mark.parametrize("seed", range(6))
@pytest.mark.parametrize(
    ("lam", "beta", "geom_scale"),
    [(2.2, 2.8, 0.6), (1.5, 2.4, 0.3), (3.0, 3.5, 1.0), (2.2, 2.2, 0.6)],
)
def test_prefix_subset_update_matches_brute_force(seed, lam, beta, geom_scale):
    """Theorem 3.4(a): for beta >= lam the optimal crop is a varphi prefix."""
    torch.manual_seed(seed)
    pool = 10
    probs = torch.softmax(torch.randn(pool) * 2.0, dim=-1)
    embeddings = torch.nn.functional.normalize(torch.randn(pool, 8), dim=-1)
    kept0 = torch.arange(3)
    potential = geom_scale * nearest_set_potential(embeddings, kept0)

    prefix = prefix_subset_update(probs, potential, lam=lam, beta=beta)
    best_subset, best_value = brute_force_subset_update(
        probs, potential, lam=lam, beta=beta
    )
    prefix_value = subset_objective(probs, potential, prefix, lam=lam, beta=beta)

    assert prefix_value == pytest.approx(best_value, abs=1e-9)
    exhaustive = max(
        subset_objective(
            probs, potential, torch.tensor(subset), lam=lam, beta=beta
        )
        for size in range(1, pool + 1)
        for subset in combinations(range(pool), size)
    )
    assert exhaustive == pytest.approx(best_value, abs=1e-9)


def test_prefix_scan_can_be_suboptimal_when_hypothesis_is_relaxed():
    """With beta < lam (outside Theorem 3.4a) a pure prefix scan can miss
    the optimum; at least one counterexample must exist in the scanned
    seeds, mirroring the paper's single-token collapse regime."""
    lam, beta, geom_scale = 2.2, 1.0, 0.6
    found_gap = 0.0
    for seed in range(200):
        torch.manual_seed(5000 + seed)
        probs = torch.softmax(torch.randn(10) * 2.0, dim=-1)
        embeddings = torch.nn.functional.normalize(torch.randn(10, 8), dim=-1)
        potential = geom_scale * nearest_set_potential(
            embeddings, torch.arange(3)
        )
        prefix = prefix_subset_update(probs, potential, lam=lam, beta=beta)
        _, best_value = brute_force_subset_update(
            probs, potential, lam=lam, beta=beta
        )
        gap = best_value - subset_objective(
            probs, potential, prefix, lam=lam, beta=beta
        )
        found_gap = max(found_gap, gap)
    assert found_gap > 1e-6


@pytest.mark.parametrize("seed", range(4))
def test_top_w_mask_matches_vendored_official_implementation(seed):
    upstream = load_upstream_module()
    torch.manual_seed(seed)
    vocab, dim = 300, 24
    logits = torch.randn(vocab) * 2.0
    embeddings = torch.randn(vocab, dim)

    ours = top_w_mask(
        logits,
        embeddings,
        temperature=0.7,
        top_m=64,
        init_top_p=0.999,
        alt_iters=9,
        geom_scale=0.6,
        lam=2.2,
        beta=2.8,
    )
    emb_np = embeddings.numpy(force=True)
    mean = emb_np.mean(axis=0, keepdims=True)
    var = ((emb_np - mean) ** 2).mean(axis=0, keepdims=True)
    masked = upstream._topw_mask_logits(
        logits=logits.numpy(force=True).astype(np.float64),
        embeddings_full=emb_np,
        mean_full=mean.astype(np.float32),
        scale_full=(1.0 / np.sqrt(np.clip(var, 1e-6, None))).astype(np.float32),
        temperature=0.7,
        top_m=64,
        init_top_p=0.999,
        alt_iters=9,
        geom_chunk=4096,
        geom_scale=0.6,
        lam_fixed=2.2,
        beta_override=2.8,
    )
    official_kept = np.flatnonzero(np.isfinite(masked))
    assert sorted(ours["kept"].tolist()) == sorted(official_kept.tolist())


def test_top_w_mask_converges_within_budget_and_keeps_pool_subset():
    torch.manual_seed(7)
    logits = torch.randn(500) * 2.0
    embeddings = torch.randn(500, 32)
    result = top_w_mask(logits, embeddings, temperature=1.0, top_m=64)
    assert result["converged"]
    assert 1 <= result["iterations"] <= 9
    assert len(result["kept"]) >= 1
    probs = result["probs"]
    assert probs.shape == (500,)
    assert float(probs.sum()) == pytest.approx(1.0, abs=1e-6)
    assert set(torch.nonzero(probs).flatten().tolist()) == set(
        result["kept"].tolist()
    )


def test_identical_embeddings_reduce_to_probability_prefix():
    """Section 4.3: with no geometry, Top-W keeps a top-probability prefix."""
    torch.manual_seed(3)
    logits = torch.randn(200) * 2.0
    embeddings = torch.ones(200, 16)
    result = top_w_mask(logits, embeddings, temperature=1.0, top_m=64)
    kept = sorted(result["kept"].tolist())
    probs = torch.softmax(logits, dim=-1)
    order = torch.argsort(probs, descending=True)
    prefix = sorted(order[: len(kept)].tolist())
    assert kept == prefix


def test_whiten_and_normalize_produces_unit_rows():
    torch.manual_seed(9)
    embeddings = torch.randn(50, 8)
    whitened = whiten_and_normalize(embeddings)
    norms = torch.linalg.norm(whitened, dim=-1)
    assert torch.allclose(norms, torch.ones(50), atol=1e-5)


def test_baseline_filters_renormalize():
    torch.manual_seed(4)
    logits = torch.randn(100)
    probs = torch.softmax(logits, dim=-1)
    for filtered in (
        min_p_filter(probs, min_p=0.05),
        top_p_filter(probs, top_p=0.9),
        top_h_filter(logits, top_h_ratio=0.8),
    ):
        assert float(filtered.sum()) == pytest.approx(1.0, abs=1e-6)


def test_evaluate_decoding_metrics_reports_all_methods():
    torch.manual_seed(5)
    logits = torch.randn(300) * 2.0
    embeddings = torch.randn(300, 16)
    metrics = evaluate_decoding_metrics(logits, embeddings, temperature=0.7)
    for key in (
        "entropy_top_w",
        "entropy_min_p",
        "entropy_top_p",
        "entropy_top_h",
        "subset_size_top_w",
        "subset_size_top_p",
    ):
        assert key in metrics
        assert np.isfinite(metrics[key])
