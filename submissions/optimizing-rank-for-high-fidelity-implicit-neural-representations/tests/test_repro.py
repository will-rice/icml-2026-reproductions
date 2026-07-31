"""Test suite for Optimizing Rank for High-Fidelity INRs reproduction."""

import json
from pathlib import Path
import pytest
from optimizing_rank_inr_repro.benchmarks import (
    run_claim1_stable_rank_degradation_test,
    run_claim2_image_overfitting_test,
    run_claim3_sparse_ct_test,
    run_claim4_multidomain_extension_test,
    run_all_benchmarks,
)


def test_claim1_stable_rank_degradation():
    res = run_claim1_stable_rank_degradation_test()
    assert res["status"] == "verified"
    assert res["rank_preserved_by_muon"] is True
    assert res["final_muon_stable_rank"] > res["final_adam_stable_rank"]


def test_claim2_image_overfitting():
    res = run_claim2_image_overfitting_test()
    assert res["status"] == "verified"
    assert res["all_architectures_improved"] is True


def test_claim3_sparse_ct():
    res = run_claim3_sparse_ct_test()
    assert res["status"] == "verified"
    assert res["psnr_gain_db"] >= 0.0


def test_claim4_multidomain_extension():
    res = run_claim4_multidomain_extension_test()
    assert res["status"] == "verified"
    assert res["all_domains_improved"] is True


def test_determinism():
    res1 = run_claim1_stable_rank_degradation_test(steps=50, seed=123)
    res2 = run_claim1_stable_rank_degradation_test(steps=50, seed=123)
    assert res1 == res2


def test_evidence_file():
    ev_path = Path(__file__).parent.parent / "evidence" / "evidence.json"
    assert ev_path.exists(), "evidence.json must exist"
    with open(ev_path) as f:
        data = json.load(f)
    assert data["paper_id"] == "2azIa9tfl3"
    assert len(data["claims"]) == 4
    assert data["summary"]["all_claims_verified"] is True
