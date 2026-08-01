import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


PROJECT = Path(__file__).resolve().parents[1]
MODULE_PATH = PROJECT / "generate_evidence.py"


def load_module():
    assert MODULE_PATH.exists(), "generate_evidence.py must implement segmented-execution evidence generation"
    spec = importlib.util.spec_from_file_location("segmented_evidence", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_tbptt_gradient_matches_explicit_truncated_objective():
    module = load_module()

    result = module.compare_tbptt_to_truncated_objective(
        num_segments=5,
        segment_width=4,
        state_width=3,
        seed=13,
    )

    assert result["max_abs_gradient_error"] < 1e-9
    assert result["parameter_count"] > 0


def test_retrieved_prefix_is_forward_only_and_detached_from_gradient():
    module = load_module()

    result = module.check_retrieval_gradient_isolation(seed=19)

    assert result["retrieved_prefix_gradient_norm"] == pytest.approx(0.0, abs=1e-12)
    assert result["carried_state_gradient_norm"] > 0.0
    assert result["retrieval_changes_forward_output"] is True


def test_memory_scaling_reaches_paper_direction_at_128k():
    module = load_module()

    curve = module.compute_memory_scaling(
        context_lengths=[4096, 8192, 16384, 32768, 65536, 131072],
        segment_length=4096,
        carried_tail=1024,
        retrieved_prefix=4096,
        active_long_layers=8,
        total_layers=32,
    )

    ratios = [point["full_to_segmented_ratio"] for point in curve]
    assert ratios == sorted(ratios)
    assert ratios[-1] >= 5.5
    assert ratios[-1] <= 7.0


def test_evidence_summary_covers_all_bound_claims(tmp_path):
    module = load_module()

    output = tmp_path / "evidence_summary.json"
    summary = module.generate_evidence(output_path=output)

    loaded = json.loads(output.read_text(encoding="utf-8"))
    assert loaded == summary
    assert loaded["paper_id"] == "PoRigyDOcC"
    assert loaded["attempt_id"] == "18872478-4b49-464f-b63c-0ee39d354284"
    assert len(loaded["claims"]) == 6
    assert {claim["challenge_claim_sha256"] for claim in loaded["claims"]} == {
        "25f3ccef60346b8971c84ae4c1198d71df76526f54ddcc120fb772587ffbbbd4",
        "33d8cc56b976c169c14c76ef6e694d9e3a621db03a950d4f38c39c85a1cdd53b",
        "8266ddd98e721ca3097a423634a36e7172a09f5c37ee4e5f10ff36d45ef20728",
        "e6ef1ca7199ebaa42b363cd4677889ab91c4d5771563ad54aba748ffc5fbdcd7",
        "76e754cb3680c6a2392c6547586848a704fc161b1d83279c3737aa4b6abaf68a",
        "391d5f95dceaeca03b50bf7ab344648fbf9517646d884cf93c9c4f2332d1a1c0",
    }
    assert all("status" in claim and "observations" in claim for claim in loaded["claims"])
    assert np.isfinite(loaded["checks"]["tbptt_gradient"]["max_abs_gradient_error"])
