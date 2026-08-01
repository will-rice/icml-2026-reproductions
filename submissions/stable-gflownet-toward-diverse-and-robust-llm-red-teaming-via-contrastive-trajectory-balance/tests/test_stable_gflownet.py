import pytest
import torch
import os
import json
from pathlib import Path

from stable_gflownet.ctb_loss import compute_ctb_loss, contrastive_trajectory_balance
from stable_gflownet.ngp_pruning import noisy_gradient_pruning, filter_noisy_rewards
from stable_gflownet.mink_stabilizer import mink_fluency_loss, compute_mink_penalty
from stable_gflownet.redteaming_benchmark import run_redteaming_benchmark, evaluate_ablations

def test_claim1_ctb_no_z_estimation():
    """Test Claim 1: CTB replaces explicit partition-function estimation with pairwise trajectory comparisons."""
    log_pf = torch.tensor([-2.0, -4.0, -3.5, -5.0])
    log_pb = torch.tensor([-1.0, -2.5, -2.0, -3.0])
    log_rewards = torch.tensor([1.5, 3.0, 2.5, 4.0])

    loss, metrics = compute_ctb_loss(log_pf, log_pb, log_rewards)

    assert isinstance(loss, torch.Tensor)
    assert loss.item() >= 0.0
    assert metrics["explicit_z_used"] is False
    assert metrics["partition_function_params"] == 0
    assert metrics["num_pairs"] == 6.0  # 4 choose 2


def test_claim2_noisy_gradient_pruning():
    """Test Claim 2: Noisy Gradient Pruning filters uninformative reward differences."""
    # Rewards with some very small differences (< 0.1) and some larger differences
    log_rewards = torch.tensor([1.0, 1.02, 2.5, 2.51, 4.0])

    valid_pairs, valid_weights = noisy_gradient_pruning(log_rewards, threshold=0.1)
    stats = filter_noisy_rewards(torch.zeros(5), torch.zeros(5), log_rewards, threshold=0.1)

    assert valid_pairs.size(0) < 10  # 5 choose 2 = 10 total pairs; pairs (0,1) and (2,3) pruned
    assert stats["pruned_ratio"] > 0.0
    assert stats["filtering_active"] is True


def test_claim3_mink_fluency_stabilizer():
    """Test Claim 3: Min-K Fluency Stabilizer penalizes non-fluent out-of-distribution prompts."""
    # High log-probs for fluent prompt
    fluent_log_probs = -torch.rand(4, 10) * 1.5
    # Low log-probs for gibberish prompt
    gibberish_log_probs = -torch.rand(4, 10) * 10.0 - 5.0

    results = compute_mink_penalty(fluent_log_probs, gibberish_log_probs, k_percent=0.2, fluency_threshold=-3.5)

    assert results["fluency_separation_valid"] is True
    assert results["gibberish_penalty"] > results["fluent_penalty"]
    assert results["fluent_min_k"] > results["gibberish_min_k"]


def test_claim4_redteaming_benchmark_diversity():
    """Test Claim 4: Stable-GFN reports stronger attack diversity and performance than GFN baselines."""
    results = run_redteaming_benchmark(num_samples=50, seed=42)

    assert results["stable_gfn_attack_diversity"] > results["tb_baseline_attack_diversity"]
    assert results["diversity_improvement_ratio"] > 1.0
    assert results["stable_gfn_outperforms_baseline"] is True


def test_claim5_loss_ablations():
    """Test Claim 5: Ablations evaluate loss-function and reward-stabilization choices (Table 3)."""
    ablations = evaluate_ablations(num_samples=50, seed=42)

    assert "full_stable_gfn" in ablations
    assert "wo_mink" in ablations
    assert "wo_ngp" in ablations
    assert "tb_baseline" in ablations

    assert ablations["full_stable_gfn"]["requires_z"] is False
    assert ablations["tb_baseline"]["requires_z"] is True
    assert ablations["full_stable_gfn"]["ngp_active"] is True
    assert ablations["full_stable_gfn"]["mink_active"] is True
