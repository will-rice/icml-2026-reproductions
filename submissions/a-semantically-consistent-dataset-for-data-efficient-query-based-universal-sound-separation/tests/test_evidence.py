import json

from hive_repro.evidence import (
    CLAIMS,
    UPSTREAM_PINS,
    audit_hub_artifacts,
    audit_result_artifacts,
    build_evidence_bundle,
)


def test_hub_audit_records_dataset_and_checkpoint_metadata():
    audit = audit_hub_artifacts(
        {
            "metadata_dataset": {
                "repo": "ShandaAI/Hive",
                "sha": "abc",
                "files": {"train/data.parquet": 100, "validation/data.parquet": 10, "test/data.parquet": 1},
            },
            "audio_archive": {
                "repo": "JusperLee/Hive-ALL",
                "sha": "def",
                "files": {"test/2mix/tar_000001.tar": 2649610240},
            },
            "audiosep_model": {
                "repo": "AlayaLab/AudioSep-hive",
                "sha": "ghi",
                "files": {"audiosep_hive.ckpt": 10, "config.yaml": 1},
            },
            "flowsep_model": {
                "repo": "AlayaLab/FlowSep-hive",
                "sha": "jkl",
                "files": {"flowsep_hive.ckpt": 10, "config.yaml": 1},
            },
        }
    )

    assert audit["metadata_parquets"] == "present"
    assert audit["audio_archive_tars"] == 1
    assert audit["hive_checkpoints"] == "present"


def test_result_audit_reports_missing_table_artifacts():
    audit = audit_result_artifacts({"README.md", "hive_dataset/README.md", "infer_audiosep.py"})

    assert audit["table3_result_files"] == []
    assert audit["table6_result_files"] == []


def test_bundle_records_claim_statuses_and_exact_pins():
    bundle = build_evidence_bundle(
        source_files={
            "pipeline/code/01_audio_chunking.py": "chunk raw audio",
            "pipeline/code/02_filter_single_label.py": "single label",
            "pipeline/code/03_filter_single_event_qwen.py": "Qwen3 Omni single-event filter",
            "pipeline/code/04_audioset_label_audiotag.py": "AudioTag ontology",
            "pipeline/code/05_leaf_label_qwen.py": "leaf labels with hive_ontology",
            "pipeline/code/06_superres_apollo.py": "Apollo super-resolution to 44.1kHz",
            "hive_dataset/README.md": "2,442 hours 19.6M mixtures 283 classes logic-based co-occurrence matrix",
            "infer_audiosep.py": "ShandaAI/AudioSep-hive",
            "infer_flowsep.py": "ShandaAI/FlowSep-hive",
        },
        repo_files={"README.md", "hive_dataset/README.md", "infer_audiosep.py", "infer_flowsep.py"},
        ontology=[
            {"id": "root", "name": "Root", "child_ids": ["a"]},
            {"id": "a", "name": "Class", "child_ids": []},
        ],
        hub_artifacts={
            "metadata_dataset": {"repo": "ShandaAI/Hive", "sha": "abc", "files": {"train/data.parquet": 1}},
            "audio_archive": {"repo": "JusperLee/Hive-ALL", "sha": "def", "files": {"test/2mix/tar_000001.tar": 1}},
            "audiosep_model": {"repo": "AlayaLab/AudioSep-hive", "sha": "ghi", "files": {"audiosep_hive.ckpt": 1}},
            "flowsep_model": {"repo": "AlayaLab/FlowSep-hive", "sha": "jkl", "files": {"flowsep_hive.ckpt": 1}},
        },
    )

    assert bundle["paper_id"] == "vCc2NAe0OS"
    assert bundle["upstream_pins"] == UPSTREAM_PINS
    assert [result["claim_sha256"] for result in bundle["claim_results"]] == [
        claim["challenge_claim_sha256"] for claim in CLAIMS
    ]
    assert [result["status"] for result in bundle["claim_results"]] == [
        "verified",
        "toy",
        "inconclusive",
        "inconclusive",
        "toy",
        "inconclusive",
    ]
    json.dumps(bundle)
