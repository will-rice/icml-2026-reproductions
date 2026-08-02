from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]


def test_generate_evidence_records_pinned_dataset_metadata_and_claim_results(tmp_path: Path) -> None:
    output = tmp_path / "textatlas5m_results.json"
    result = subprocess.run(
        [
            sys.executable,
            str(PROJECT / "generate_evidence.py"),
            "--output",
            str(output),
        ],
        cwd=PROJECT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["attempt_id"] == "c21b9754-4227-4943-b26f-ac9afd5712a4"
    assert payload["paper"]["paper_id"] == "5vufrrbi4N"
    assert payload["snapshot_id"] == "367354e797e820ffc39729ded717a4c02df0c72eee85303de50ba3c181ddde47"
    assert payload["upstream"]["hf_dataset_commit"] == "f9f2a0f5000fbb078f718197acb45cfb9ceed551"
    assert payload["upstream"]["code_commit"] == "f13e9926689de1bc4d671b3f21a1c62255be738d"

    dataset = payload["dataset_metadata"]
    assert dataset["license"] == "mit"
    assert dataset["is_private"] is False
    assert dataset["is_gated"] is False
    assert dataset["config_count"] == 10
    assert dataset["total_examples"] > 5_000_000
    for key in [
        "synthetic_clean_text",
        "interleaved_text_vision",
        "styled_synthetic_scenes",
        "ppt",
        "book_covers",
        "papers",
        "textsceneshq",
    ]:
        assert key in dataset["domain_map"]

    claims = {claim["id"]: claim for claim in payload["claims"]}
    assert set(claims) == {
        "textatlaseval_size",
        "dataset_domain_coverage",
        "textatlas5m_scale_and_annotations",
        "model_eval_metrics",
        "finetuning_improvement",
    }
    assert claims["textatlas5m_scale_and_annotations"]["challenge_claim_sha256"] == (
        "012c780c0991f038c727283e4810bf52d104984e1d7d15795cf31b06583df262"
    )
    assert claims["textatlas5m_scale_and_annotations"]["status"] in {
        "metadata_verified",
        "metadata_falsified",
        "unavailable",
    }
    assert claims["model_eval_metrics"]["source"] != "paper_prose" or claims["model_eval_metrics"]["status"] == "unavailable"
    assert claims["finetuning_improvement"]["source"] != "paper_prose" or claims["finetuning_improvement"]["status"] == "unavailable"

    pages = sorted((PROJECT / "pages").glob("*.md"))
    assert len(pages) >= 2
    numeric_lines = sum(
        1
        for page in pages
        for line in page.read_text(encoding="utf-8").splitlines()
        if any(character.isdigit() for character in line)
    )
    assert numeric_lines >= 15
