from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from dmpo_repro.evidence import (
    build_bundle,
    centered_advantages,
    reward_tilted_weights,
    weighted_denoising_ce,
)


UPSTREAM = Path("/tmp/icml-dmpo-upstream-1785350300")


def test_reward_tilted_weights_match_softmax_formula() -> None:
    log_rnds = np.array([-0.7, -0.1, -1.4], dtype=np.float64)
    rewards = np.array([0.0, 1.0, 0.25], dtype=np.float64)
    observed = reward_tilted_weights(log_rnds, rewards, alpha=0.5, coeff=0.8)
    logits = 0.8 * (log_rnds + rewards / 0.5)
    expected = np.exp(logits - logits.max())
    expected = expected / expected.sum()
    assert np.allclose(observed, expected)
    assert math.isclose(float(observed.sum()), 1.0)
    assert observed.argmax() == 1


def test_weighted_denoising_ce_matches_upstream_loss_shape() -> None:
    token_losses = np.array(
        [
            [0.0, 0.2, 0.8, 0.0],
            [0.1, 0.0, 0.4, 0.7],
            [0.0, 0.3, 0.5, 0.9],
            [0.2, 0.1, 0.0, 0.6],
        ],
        dtype=np.float64,
    )
    mask_counts = np.array([2, 3, 3, 3], dtype=np.float64)
    advantages = np.array([0.55, 0.45, 0.55, 0.45], dtype=np.float64)
    observed = weighted_denoising_ce(token_losses, mask_counts, advantages, num_replicates=2)
    expected = float(((token_losses.sum(axis=-1) / mask_counts) * advantages).sum() / 2)
    assert math.isclose(observed, expected)


def test_centered_advantages_subtract_negative_baseline() -> None:
    advantages = np.array([0.75, 0.25], dtype=np.float64)
    negative = np.array([0.2, 0.8], dtype=np.float64)
    centered = centered_advantages(advantages, negative, strength=0.5)
    assert np.allclose(centered, np.array([0.65, -0.15]))
    assert centered.sum() < advantages.sum()


def test_bundle_verifies_target_claims_from_pinned_upstream() -> None:
    upstream = UPSTREAM if UPSTREAM.exists() else None
    bundle = build_bundle(upstream)
    assert bundle["attempt_id"] == "2ad6b75a-74a9-4603-bf31-e15707b3e683"
    assert bundle["upstream"]["github"]["commit"] == "1661fa7d75f0ccec3bbc1b6cae94e9e3fb88571a"
    statuses = {claim["sha256"]: claim["status"] for claim in bundle["claims"]}
    assert statuses["de36b989902f6868972692307865db7ac7943f97c4bf4ffc50880d83c14bff6c"] == "verified"
    assert statuses["f09381d3dd0cb2d8f52436eeff59f2739930f041584ba590c2cbf4275e03e368"] == "verified"
    assert statuses["9b048a80e727b9407f8c05a6b7873db6a4e334e65b039ffa90e635c31914bfaa"] in {"verified", "toy"}
    assert "dmpo_trainer.py" in bundle["observations"]["file_hashes"]
    assert bundle["api_cost_usd"] == 0.0
