import sys
import os
import pytest

# Ensure root package directory is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from weak_diffusion_priors.theory import simulate_theorem_3_1_posterior_concentration
from weak_diffusion_priors.inverse_problem import evaluate_table_1_inverse_problem_baselines
from generate_evidence import generate_evidence


def test_theorem_3_1_simulation():
    res = simulate_theorem_3_1_posterior_concentration(n_dim=64, seed=123)
    assert res["theorem_3_1_verified"] is True
    assert len(res["sweep_results"]) > 0
    high_ratio_error = res["sweep_results"][-1]["reconstruction_error_weak"]
    low_ratio_error = res["sweep_results"][0]["reconstruction_error_weak"]
    assert high_ratio_error < low_ratio_error


def test_table_1_inverse_problem_eval():
    res = evaluate_table_1_inverse_problem_baselines(signal_length=128, num_samples=20, seed=123)
    assert res["claim_1_verified"] is True
    high_regime = res["table_1_metrics"]["High_Informative_m_n_0.90"]
    assert high_regime["psnr_ratio_weak_vs_strong"] >= 0.80


def test_generate_evidence():
    evidence = generate_evidence()
    assert evidence["summary"]["all_claims_verified"] is True
    assert len(evidence["claims"]) == 2

    evidence_path = os.path.join(os.path.dirname(__file__), "..", "evidence", "evidence.json")
    assert os.path.exists(evidence_path)
