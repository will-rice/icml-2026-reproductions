import json
from pathlib import Path

from r4t_repro.evidence import (
    CLAIMS,
    UPSTREAM_PINS,
    audit_source_manifest,
    audit_tables,
    build_evidence_bundle,
    compile_synthetic_supervision,
    toy_set_reward,
)


def test_source_manifest_detects_manuscript_assets_not_executable_artifacts():
    files = {
        "main.tex",
        "task_1_result.tex",
        "task_2_result.tex",
        "query_fanout_efficiency.pdf",
        "bq_reward_gemma.png",
        "framework.pdf",
    }

    audit = audit_source_manifest(files)

    assert audit["has_main_tex"] is True
    assert audit["has_table1_tex"] is True
    assert audit["has_table2_tex"] is True
    assert audit["has_latency_figure"] is True
    assert audit["python_files"] == []
    assert audit["dataset_files"] == []


def test_table_audit_counts_r4t_rows_but_marks_values_as_paper_reported():
    tables = audit_tables(
        task1_tex="R4T-FOLM R4T-Diffusion No Fan-out Best-of-N",
        task2_tex="R4T-FOLM (Qwen) R4T-Diffusion (Qwen) Gemini-2.5-Flash",
    )

    assert tables["task1_r4t_mentions"] == 2
    assert tables["task2_r4t_mentions"] == 2
    assert tables["table_values_are_recomputed"] is False


def test_toy_reward_penalizes_collapse_and_compiles_supervision():
    database = [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]]
    query = [1.0, 0.0]
    collapsed = [[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]]
    diverse = [[1.0, 0.0], [0.0, 1.0], [0.7, 0.7]]

    assert toy_set_reward(query, diverse, database)["total"] > toy_set_reward(query, collapsed, database)["total"]

    supervision = compile_synthetic_supervision({"festival outfit": diverse})
    assert supervision[0]["query"] == "festival outfit"
    assert supervision[0]["target_count"] == 3


def test_bundle_records_exact_claim_statuses():
    bundle = build_evidence_bundle(
        source_files={
            "main.tex": "train a fan-out policy synthetic supervision diffusion model groundedness diversity alignment Figure 5",
            "task_1_result.tex": "R4T-FOLM R4T-Diffusion",
            "task_2_result.tex": "R4T-FOLM (Qwen) R4T-Diffusion (Qwen)",
            "query_fanout_efficiency.pdf": "binary",
            "bq_reward_gemma.png": "binary",
        }
    )

    assert bundle["attempt_id"] == "8a83f44b-e3db-4c2b-acf7-d233a750fdcc"
    assert bundle["paper_id"] == "4P9cEcinYP"
    assert bundle["upstream_pins"] == UPSTREAM_PINS
    assert [result["claim_sha256"] for result in bundle["claim_results"]] == [
        claim["challenge_claim_sha256"] for claim in CLAIMS
    ]
    assert [result["status"] for result in bundle["claim_results"]] == [
        "toy",
        "inconclusive",
        "inconclusive",
        "toy",
        "inconclusive",
    ]
    json.dumps(bundle)


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
