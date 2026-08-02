import json
from pathlib import Path

from avgen_repro.evidence import (
    CLAIMS,
    UPSTREAM_PINS,
    audit_artifact_availability,
    audit_metric_inventory,
    audit_prompt_inventory,
    build_evidence_bundle,
    parse_readme_leaderboard,
    summarize_leaderboard_failure_modes,
)


PROMPTS = {
    "ads.json": [
        {"content": "Ad One", "prompt": "A beach rental ad with spoken text and on-screen slogan."},
        {"content": "Ad Two", "prompt": "A product ad with synchronized voiceover."},
    ],
    "sports.json": [
        {"content": "Golf: The Bunker Explosion", "prompt": "Club strikes sand with a dull thump."}
    ],
}

AGGREGATE_SOURCE = """
GROUP_WEIGHTS = {"basic": 0.2, "cross": 0.2, "fine": 0.6}
GROUP_DIMENSIONS = {
    "basic": ("Vis", "Aud"),
    "cross": ("AV", "Lip"),
    "fine": ("Text", "Face", "Music", "Speech", "Lo-Phy", "Hi-Phy", "Holistic"),
}
"""

README_TABLE = """
| Model | Components | Vis | Aud (PQ) | AV | Lip | Text | Face | Music | Speech | Lo-Phy | Hi-Phy | Holistic | Total |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| StrongVisual | Demo | 0.970 | 7.40 | 0.20 | 2.00 | 12.00 | 50.00 | 4.00 | 92.00 | 3.00 | 60.00 | 70.00 | 64.00 |
| Balanced | Demo | 0.900 | 6.80 | 0.25 | 3.00 | 72.00 | 55.00 | 22.00 | 88.00 | 4.00 | 72.00 | 82.00 | 70.00 |
"""


def test_prompt_inventory_counts_categories_prompts_and_complexity():
    audit = audit_prompt_inventory(PROMPTS)

    assert audit["category_count"] == 2
    assert audit["prompt_count"] == 3
    assert audit["category_counts"] == {"ads": 2, "sports": 1}
    assert audit["prompts_with_audio_cues"] >= 2
    assert audit["sha256_by_file"]["ads.json"]


def test_metric_inventory_reads_aggregate_groups_and_expected_dimensions():
    audit = audit_metric_inventory(AGGREGATE_SOURCE, {"eval/Ocr/batch_eval.py", "eval/speech/batch_eval.py"})

    assert audit["group_count"] == 3
    assert audit["metric_count"] == 11
    assert audit["groups"]["fine"] == ["Text", "Face", "Music", "Speech", "Lo-Phy", "Hi-Phy", "Holistic"]
    assert audit["module_presence"]["text"] == "present"
    assert audit["module_presence"]["speech"] == "present"


def test_leaderboard_parser_and_failure_mode_summary():
    rows = parse_readme_leaderboard(README_TABLE)
    summary = summarize_leaderboard_failure_modes(rows)

    assert len(rows) == 2
    assert rows[0]["model"] == "StrongVisual"
    assert summary["models_with_high_basic_and_low_fine"] == ["StrongVisual"]
    assert summary["lowest_mean_dimensions"][0]["dimension"] == "Music"


def test_bundle_records_pins_claims_and_unavailable_artifacts():
    bundle = build_evidence_bundle(
        prompts=PROMPTS,
        aggregate_source=AGGREGATE_SOURCE,
        readme=README_TABLE,
        repo_files={"scripts/eval_scale_stability_from_cached.py"},
        hf_files={"prompts/ads.json", "metadata.parquet"},
    )

    assert bundle["paper_id"] == "aJdgt8xDMy"
    assert bundle["snapshot_id"] == "41692f328d154e4fad790fb8c89aa276452ce49b8aaa18064abb9c47a897d622"
    assert bundle["upstream_pins"] == UPSTREAM_PINS
    assert [result["claim_sha256"] for result in bundle["claim_results"]] == [
        claim["challenge_claim_sha256"] for claim in CLAIMS
    ]
    assert bundle["audits"]["artifact_availability"]["human_correlation_artifacts"] == []
    assert bundle["claim_results"][4]["status"] == "inconclusive"
    json.dumps(bundle)


def test_artifact_availability_requires_raw_human_and_repeat_outputs():
    audit = audit_artifact_availability(
        repo_files={
            "scripts/eval_scale_stability_from_cached.py",
            "README.md",
            "prompts/ads.json",
        },
        hf_files={
            "metadata.parquet",
            "prompts/ads.json",
            "Veo_3.1_fast/ads/example.mp4",
        },
    )

    assert audit["generated_video_present"] is True
    assert audit["stability_code_present"] is True
    assert audit["repeat_output_artifacts"] == []
    assert audit["human_correlation_artifacts"] == []


def test_served_pages_meet_controller_scoring_gate():
    pages = sorted((Path(__file__).resolve().parents[1] / "pages").glob("*.md"))
    texts = [path.read_text(encoding="utf-8") for path in pages]
    numeric_lines = sum(
        1
        for text in texts
        for line in text.splitlines()
        if any(character.isdigit() for character in line)
    )

    assert len(pages) >= 2
    assert sum(len(text.strip()) for text in texts) >= 200
    assert numeric_lines >= 15
