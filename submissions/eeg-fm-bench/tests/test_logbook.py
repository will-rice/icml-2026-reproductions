from __future__ import annotations

import json
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]


def test_logbook_has_canonical_claim_aligned_structure() -> None:
    """Catches missing/reordered claim pages or unlabeled evidence boundaries."""
    metadata = json.loads(
        (PROJECT / ".trackio" / "logbook" / "logbook.json").read_text(
            encoding="utf-8"
        )
    )
    assert metadata["root"]["title"] == "EEG-FM-Bench released-artifact audit"
    assert [page["title"] for page in metadata["root"]["children"]] == [
        "Executive summary",
        "Fourteen datasets and ten paradigms",
        "Standardized preprocessing reproducibility",
        "Three-strategy evaluation harness",
        "Conclusion",
    ]
    page_text = "\n".join(
        path.read_text(encoding="utf-8")
        for path in sorted((PROJECT / ".trackio" / "logbook" / "pages").glob("*/page.md"))
    )
    for claim_id in (
        "fourteen-dataset-ten-paradigm-curation",
        "standardized-preprocessing-reproducibility",
        "three-strategy-evaluation-harness",
    ):
        assert claim_id in page_text
    assert "paper-reported context" in page_text
    assert "computed evidence" in page_text
    assert "unavailable" in page_text
    assert '"type": "figure"' in page_text
    assert '"pinned": true' in page_text
    assert "<title>EEG-FM-Bench released-artifact audit</title>" in page_text


def test_poster_is_self_contained_and_truthful() -> None:
    """Catches a network-dependent poster or unsupported leaderboard claim."""
    poster = (PROJECT / "poster.html").read_text(encoding="utf-8")
    embed = (PROJECT / "poster_embed.html").read_text(encoding="utf-8")

    assert poster == embed
    assert "<script" not in poster.lower()
    assert "http://" not in poster.lower()
    assert "https://" not in poster.lower()
    assert "14 datasets" in poster
    assert "8 release-defined task types" in poster
    assert "paper context: 10 canonical paradigms" in poster
    assert "GPU leaderboard: unavailable" in poster
    assert "released-artifact audit" in poster
