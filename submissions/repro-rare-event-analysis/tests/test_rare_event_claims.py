import importlib.util
import json
from pathlib import Path

import numpy as np


SUBMISSION_ROOT = Path(__file__).resolve().parents[1]


def test_text_process_enumerates_normalized_distribution():
    from rare_event_llm.process import TextProcess, enumerate_sequences

    records = enumerate_sequences(TextProcess.default(length=6))

    assert len(records) == 4**6
    assert abs(sum(np.exp(record.log_probability) for record in records) - 1.0) < 1e-12
    assert min(record.observable for record in records) < max(
        record.observable for record in records
    )


def test_annealed_biased_sampling_shifts_observable_tails():
    from rare_event_llm.process import TextProcess, enumerate_sequences
    from rare_event_llm.samplers import biased_sample, transition_path_sample

    records = enumerate_sequences(TextProcess.default(length=6))
    neutral = biased_sample(records, beta=0.0, sample_count=2000, seed=1)
    high = biased_sample(records, beta=2.0, sample_count=2000, seed=2)
    low = biased_sample(records, beta=-2.0, sample_count=2000, seed=3)
    annealed = transition_path_sample(
        records, beta_schedule=[0.0, 0.75, 1.5, 2.0], steps=2000, seed=4
    )

    assert high.mean_observable > neutral.mean_observable
    assert low.mean_observable < neutral.mean_observable
    assert annealed[-1].mean_observable > neutral.mean_observable


def test_mbar_reconstructs_exact_observable_histogram():
    from rare_event_llm.evidence import run_reproduction

    bundle = run_reproduction()
    comparison = bundle["measurements"]["mbar_histogram"]

    assert comparison["l1_error"] < 0.08
    assert comparison["max_bin_error"] < 0.04


def test_evidence_bundle_binds_selected_claims():
    from rare_event_llm.evidence import build_evidence_bundle

    bundle = build_evidence_bundle()

    assert bundle["paper_id"] == "2RJN5vDHG0"
    assert [claim["challenge_claim_sha256"] for claim in bundle["claims"]] == [
        "676d9200dfdcf3098ae37cbc8d61fdcffa374c5a1eeb39002379094b2aeb74bf",
        "a98926e8474dace301f85c8cd53326326195eb0dd80174f4c44ddb0899edea26",
        "b4a93b328d67630945555047ec3e9d16b5fababa66ad04b30f81436f2a045b07",
        "96cfc0cb94241dcf49d14841b69f766e8942dc5931d0310a5a4d4d72300b8f34",
    ]
    assert all(claim["status"] in {"toy", "verified"} for claim in bundle["claims"])


def test_error_analysis_compares_direct_and_mbar_intervals():
    from rare_event_llm.evidence import run_reproduction

    result = run_reproduction()
    error = result["measurements"]["error_analysis"]

    assert error["direct_ci_width"] > error["mbar_ci_width"]
    assert error["mbar_ci_low"] <= error["exact_tail_probability"]
    assert error["exact_tail_probability"] <= error["mbar_ci_high"]


def test_generate_evidence_writes_claim_bound_bundle():
    from rare_event_llm.evidence import write_evidence_bundle

    output = SUBMISSION_ROOT / "evidence" / "bundle.json"
    bundle = write_evidence_bundle(output)
    persisted = json.loads(output.read_text(encoding="utf-8"))

    assert persisted == bundle
    assert persisted["upstream_revision"] == (
        "arxiv:2602.06791v2+arxiv-source-sha256:"
        "54d5438fbe581dc0cdae014290bb5db62d4aae446a7debddf3b0866c888f1d9a"
    )


def test_space_assets_exist_and_import():
    pages = sorted((SUBMISSION_ROOT / "pages").glob("*.md"))
    total_characters = sum(len(path.read_text(encoding="utf-8").strip()) for path in pages)

    assert total_characters >= 200

    spec = importlib.util.spec_from_file_location(
        "rare_event_space_app", SUBMISSION_ROOT / "app.py"
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert hasattr(module, "demo")
