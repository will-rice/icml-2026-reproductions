import importlib
import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def _evidence_module():
    try:
        return importlib.import_module("direct_repro.evidence")
    except ModuleNotFoundError as error:
        pytest.fail(f"missing evidence module: {error}")


def test_flat_direction_preference_increases_surrogate_objective():
    evidence = _evidence_module()

    result = evidence.evaluate_flat_direction_preference()

    assert result["claim_status"] == "verified"
    assert result["flat_direction_delta"] > 0.0
    assert result["stiff_direction_delta"] < result["flat_direction_delta"]
    assert result["flat_eigenvalue"] < result["flatness_threshold"]


def test_direct_sample_priority_prefers_high_loss_long_sequences():
    evidence = _evidence_module()

    result = evidence.evaluate_sample_priority()

    assert result["claim_status"] == "verified"
    assert result["high_loss_long_sequence_id"] in result["top_selected_ids"]
    assert result["short_length_probe_top_id"] != result["high_loss_long_sequence_id"]
    assert result["deterministic"] is True


def test_evidence_generation_preserves_existing_timestamp(tmp_path):
    evidence = _evidence_module()

    first = evidence.run_evidence_generation(tmp_path)
    second = evidence.run_evidence_generation(tmp_path)

    assert second["generated_at"] == first["generated_at"]
