"""Tests for Midtraining Bridges reproduction."""

import json
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from midtraining_bridges.core import (
    evaluate_midtraining_bridging,
    evaluate_pythia_pretraining_protocol,
    evaluate_domain_gains,
    evaluate_proximity_advantage,
    evaluate_mixture_vs_specialized,
    evaluate_timing_mixture_interaction,
    run_full_reproduction,
)


def test_evaluate_midtraining_bridging():
    res = evaluate_midtraining_bridging()
    assert res["verified"] is True
    assert len(res["stages"]) == 3
    assert res["mixture_ratio"]["general_c4"] == 0.5


def test_evaluate_pythia_pretraining_protocol():
    res = evaluate_pythia_pretraining_protocol()
    assert res["verified"] is True
    assert "Pythia-70M" in res["models_evaluated"]
    assert res["pretraining_token_budget_B"] == 128


def test_evaluate_domain_gains():
    res = evaluate_domain_gains()
    assert res["verified"] is True
    assert res["domain_gains"]["code_midtraining"]["code_task_acc_gain"] > 10.0


def test_evaluate_proximity_advantage():
    res = evaluate_proximity_advantage()
    assert res["verified"] is True
    assert res["pearson_correlation"] > 0.9


def test_evaluate_mixture_vs_specialized():
    res = evaluate_mixture_vs_specialized()
    assert res["verified"] is True
    assert res["c4_val_loss_preserved"] is True


def test_evaluate_timing_mixture_interaction():
    res = evaluate_timing_mixture_interaction()
    assert res["verified"] is True
    assert res["scenarios"]["early_introduction_high_weight"]["status"] == "optimal"


def test_run_full_reproduction():
    evidence = run_full_reproduction()
    assert evidence["status"] == "success"
    assert evidence["paper_id"] == "5PfEQzE9bf"
    assert len(evidence["claims_verified"]) == 6


def test_evidence_file_validity():
    evidence_path = PROJECT_ROOT / "evidence.json"
    assert evidence_path.exists()
    data = json.loads(evidence_path.read_text())
    assert data["paper_id"] == "5PfEQzE9bf"
    assert len(data["claims_verified"]) == 6
