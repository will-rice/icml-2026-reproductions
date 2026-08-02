from hive_repro.evidence import audit_pipeline, ontology_stats


def test_pipeline_audit_detects_six_stage_hive_pipeline():
    files = {
        "pipeline/code/01_audio_chunking.py": "chunk raw audio",
        "pipeline/code/02_filter_single_label.py": "single label",
        "pipeline/code/03_filter_single_event_qwen.py": "Qwen3 Omni single-event filter",
        "pipeline/code/04_audioset_label_audiotag.py": "AudioTag ontology",
        "pipeline/code/05_leaf_label_qwen.py": "leaf labels with hive_ontology",
        "pipeline/code/06_superres_apollo.py": "Apollo super-resolution to 44.1kHz",
    }

    audit = audit_pipeline(files)

    assert audit["stage_count"] == 6
    assert audit["super_resolution"] == "present"
    assert audit["semantic_acoustic_alignment"] == "present"


def test_ontology_stats_reports_naive_count_mismatch():
    ontology = [
        {"id": "root", "name": "Root", "child_ids": ["a", "b"]},
        {"id": "a", "name": "Dog", "child_ids": []},
        {"id": "b", "name": "Cat", "child_ids": []},
    ]

    stats = ontology_stats(ontology, claimed_classes=3)

    assert stats["node_count"] == 3
    assert stats["leaf_like_count"] == 2
    assert stats["matches_claimed_classes"] is False
