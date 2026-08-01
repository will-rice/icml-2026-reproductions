import json
import math
from pathlib import Path

from memorization_repro.evidence import (
    ATTEMPT_ID,
    UPSTREAM_REVISION,
    build_evidence_bundle,
    generate_uniform_token_sequences,
    memorized_bits_from_nll,
    run_evidence,
)

EXPECTED_ATTEMPT_ID = "8de87cc9-1d39-49a6-b552-4b4dd7e67e0e"
EXPECTED_UPSTREAM_REVISION = "arxiv:2505.24832v3"


def test_memorized_bits_clamps_examples_above_uniform_baseline():
    bits = memorized_bits_from_nll(
        mean_nll_bits_per_token=[0.5, 1.0, 3.0],
        sequence_length=5,
        vocab_size=4,
    )

    assert bits == 12.5


def test_uniform_token_generation_is_seeded_and_reproducible():
    tokens = generate_uniform_token_sequences(
        num_sequences=3,
        sequence_length=4,
        vocab_size=8,
        seed=123,
    )

    assert tokens == [
        [0, 4, 1, 6],
        [4, 1, 0, 6],
        [5, 5, 0, 2],
    ]


def test_build_evidence_bundle_keeps_measured_and_paper_values_separate():
    bundle = build_evidence_bundle(
        attempt_id="attempt-1",
        paper_id="paper-1",
        upstream_revision=EXPECTED_UPSTREAM_REVISION,
        experiments=[
            {
                "model_name": "tiny",
                "parameter_count": 128,
                "dataset_size": 4,
                "memorized_bits": 64.0,
                "bits_per_parameter": 0.5,
                "mean_train_nll_bits_per_token": 1.25,
            }
        ],
    )

    assert bundle["claims"][0]["status"] in {"toy", "inconclusive"}
    assert bundle["claims"][1]["status"] in {"toy", "inconclusive"}
    assert bundle["paper_reported_context"]["plateau_bits_per_parameter"] == 3.6
    assert "plateau_bits_per_parameter" not in bundle["measurements"][0]
    assert bundle["provenance"]["upstream_revision"] == EXPECTED_UPSTREAM_REVISION


def test_run_evidence_writes_finite_machine_readable_results(tmp_path):
    output_dir = tmp_path / "evidence"

    bundle = run_evidence(
        output_dir=output_dir,
        train_steps=2,
        dataset_sizes=[4],
        model_specs=[{"name": "tiny", "d_model": 8, "n_layers": 1, "n_heads": 2}],
    )

    results = json.loads((output_dir / "results.json").read_text(encoding="utf-8"))
    provenance = json.loads((output_dir / "provenance.json").read_text(encoding="utf-8"))

    assert results == bundle
    assert provenance["attempt_id"] == EXPECTED_ATTEMPT_ID
    assert provenance["upstream_revision"] == EXPECTED_UPSTREAM_REVISION
    assert len(results["measurements"]) == 1
    assert math.isfinite(results["measurements"][0]["memorized_bits"])
    assert math.isfinite(results["measurements"][0]["bits_per_parameter"])


def test_current_attempt_metadata_is_embedded_in_default_bundle():
    assert ATTEMPT_ID == EXPECTED_ATTEMPT_ID
    assert UPSTREAM_REVISION == EXPECTED_UPSTREAM_REVISION


def test_committed_bundle_records_current_attempt_metadata():
    committed = json.loads(
        (
            Path(__file__).resolve().parents[1]
            / "evidence"
            / "bundle.json"
        ).read_text(encoding="utf-8")
    )

    assert committed["attempt_id"] == EXPECTED_ATTEMPT_ID
    assert committed["upstream_revision"] == EXPECTED_UPSTREAM_REVISION
