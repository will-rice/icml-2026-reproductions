import json
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from generate_evidence import (  # noqa: E402
    CLAIM_BINDINGS,
    parse_dataset_card,
    parse_similarity_table,
    build_evidence_bundle,
    main,
)


FIXTURES = Path(__file__).parent / "fixtures"


def test_parse_dataset_card_detects_qualitative_and_throughput_buckets():
    audit = parse_dataset_card(FIXTURES / "README.md")

    assert audit["qualitative_examples"] == 880
    assert audit["throughput_buckets"] == ["1k", "2k", "8k", "16k", "32k"]
    assert audit["throughput_examples_per_bucket"] == {
        "1k": 1536,
        "2k": 1536,
        "8k": 1536,
        "16k": 1536,
        "32k": 1536,
    }


def test_similarity_table_supports_greedy_lower_than_specbench_and_random():
    rows = parse_similarity_table(FIXTURES / "README.md")

    compared = [row for row in rows if row["category"] != "QA"]
    assert compared
    assert all(row["speed_greedy"] < row["specbench"] for row in compared)
    assert all(row["speed_greedy"] < row["speed_random"] for row in compared)


def test_bundle_marks_claims_conservatively_without_result_tables():
    bundle = build_evidence_bundle(
        dataset_readme=FIXTURES / "README.md",
        repo_dir=FIXTURES,
    )

    assert [claim["challenge_claim_sha256"] for claim in bundle["claims"]] == [
        binding["challenge_claim_sha256"] for binding in CLAIM_BINDINGS
    ]
    statuses = {claim["claim_index"]: claim["status"] for claim in bundle["claims"]}
    assert statuses[1] == "verified"
    assert statuses[2] == "verified"
    assert statuses[3] in {"toy", "inconclusive"}
    assert "No machine-readable Table 1 result artifact" in bundle["claims"][2]["evidence"]


def test_cli_writes_deterministic_json_with_final_newline(tmp_path):
    out = tmp_path / "bundle.json"
    args = [
        "--dataset-readme",
        str(FIXTURES / "README.md"),
        "--repo-dir",
        str(FIXTURES),
        "--output",
        str(out),
    ]

    assert main(args) == 0
    first = out.read_bytes()
    assert first.endswith(b"\n")
    assert main(args) == 0
    assert out.read_bytes() == first


def test_space_source_has_valid_metadata_and_scoring_pages():
    readme = (PROJECT / "README.md").read_text(encoding="utf-8")
    metadata = readme.split("---", 2)[1]
    pages = sorted((PROJECT / "pages").glob("*.md"))
    substantive = sum(len(path.read_text(encoding="utf-8").strip()) for path in pages)

    assert "emoji: " in metadata
    assert "icml2026-repro" in metadata
    assert "paper-Rl2uQlCoQX" in metadata
    assert substantive >= 200
