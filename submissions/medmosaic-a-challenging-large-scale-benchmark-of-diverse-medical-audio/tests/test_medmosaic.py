from pathlib import Path

from medmosaic_repro.core import (
    EXPECTED_QA_TYPE_COUNTS,
    build_evidence_bundle,
    load_pinned_index,
    summarize_index,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_pinned_dataset_index_counts_rows_and_qa_types():
    frame = load_pinned_index()
    summary = summarize_index(frame)

    assert summary["row_count"] == 4661
    assert summary["qa_type_counts"] == EXPECTED_QA_TYPE_COUNTS
    assert summary["difficulty_counts"] == {
        "hard": 2329,
        "medium": 1396,
        "easy": 930,
        "null": 6,
    }
    assert summary["audio_folder_counts"]["sound_only"] == 1417
    assert summary["audio_folder_counts"]["speech_sound"] == 1087


def test_index_records_have_medmosaic_question_answer_shape():
    summary = summarize_index(load_pinned_index())

    assert summary["rows_with_audio_path"] == 4661
    assert summary["rows_with_question"] == 4655
    assert summary["rows_with_ground_truth"] == 4655
    assert summary["standard_mcq_rows"] == 3964
    assert summary["standard_mcq_rows_with_ten_options"] == 3964
    assert summary["ground_truth_answer_marker_rows"] == 3964
    assert summary["multi_turn_rows"] == 6
    assert summary["multi_turn_turns"] == 18
    assert summary["multi_turn_turns_with_answer_markers"] == 18


def test_evidence_bundle_marks_only_artifact_supported_claims():
    bundle = build_evidence_bundle(load_pinned_index())
    statuses = {
        claim["challenge_claim_sha256"]: claim["status"]
        for claim in bundle["claims"]
    }

    assert statuses["7892ea8680906a105db8288b33fb04b6639d319047078eadd54a461c89d0dc09"] == "falsified"
    assert statuses["ab4285e365e2ada727ae3ccdb719a9c64ef075ed0fa1a6327daeb3d0955134cc"] == "verified"
    assert statuses["951f2c0da2384aa8130bf0d3a2901537d2cb167416f2888dd9b38d69b2720538"] == "inconclusive"
    assert statuses["6a2e0745b8a74511913de8cf606b6f3ea8504f58db131c96406ac365b2671b8c"] == "inconclusive"
    assert statuses["1ce471213ab62dad6a22534ed55da70a22b4664e8afdd603bd3d94507109b0a3"] == "inconclusive"
    assert statuses["e7429633bf76cc769e78c999e472562cdcf2d0ea0c11017548dc54d4bfdf6254"] == "inconclusive"
    assert bundle["reproduced_model_measurements"] == []
    assert bundle["reproduced_expert_review_measurements"] == []


def test_space_metadata_and_report_bind_attempt():
    readme = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
    report = (PROJECT_ROOT / "pages" / "report.md").read_text(encoding="utf-8")

    assert "icml2026-repro" in readme
    assert "paper-OMdQJQwp26" in readme
    assert "MedMosaic" in report
    assert "a6ea67bd4a65b87248c6651e559656b2c31fa669" in report
    assert "4,661" in report
    assert "46,701" in report
