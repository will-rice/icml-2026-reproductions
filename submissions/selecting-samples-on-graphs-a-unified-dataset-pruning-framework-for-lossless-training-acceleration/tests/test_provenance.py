import hashlib
from pathlib import Path

import pytest

from graph_pruning_repro.provenance import (
    PAPER,
    TARGET_CLAIMS,
    TRANSCRIPTION_SET_SHA256,
    load_transcriptions,
    transcription_set_sha256,
    verify_pdf,
)


def test_pdf_identity_and_digest_rejection(tmp_path: Path) -> None:
    assert PAPER["revision"] == "arxiv:2606.12913v2"
    assert PAPER["pdf_byte_count"] == 683737
    bad = tmp_path / "paper.pdf"
    bad.write_bytes(b"not the pinned PDF")
    with pytest.raises(ValueError, match="pinned PDF byte count"):
        verify_pdf(bad)


def test_transcriptions_are_complete_and_checksummed() -> None:
    root = Path(__file__).parents[1]
    records = load_transcriptions(root)
    equations = {record["equation"] for record in records}
    assert {
        "2",
        "3",
        "4",
        "5",
        "6",
        "7",
        "8",
        "10-11",
        "12-14",
        "Appendix E inline",
        "26",
        "27",
        "28-38",
        "Algorithm 1",
    } <= equations
    expected_keys = {
        "record_id",
        "equation",
        "pdf_page",
        "section",
        "normalized_expression",
        "source_excerpt_path",
        "source_excerpt_byte_count",
        "source_excerpt_sha256",
        "reviewed_by",
    }
    seen_paths = set()
    for record in records:
        assert set(record) == expected_keys
        path = root / record["source_excerpt_path"]
        assert path.is_relative_to(root / "paper_transcriptions")
        assert record["source_excerpt_path"] not in seen_paths
        seen_paths.add(record["source_excerpt_path"])
        excerpt = path.read_bytes()
        excerpt.decode("utf-8")
        assert record["source_excerpt_byte_count"] == len(excerpt)
        assert record["source_excerpt_sha256"] == hashlib.sha256(excerpt).hexdigest()
        assert record["reviewed_by"][0] == "codex-graph-pruning-design-author-v2"
        assert len(record["reviewed_by"]) == 2
        assert record["reviewed_by"][1] != record["reviewed_by"][0]
    assert transcription_set_sha256(records) == TRANSCRIPTION_SET_SHA256


def test_target_claims_are_exact() -> None:
    assert TARGET_CLAIMS == (
        "The paper casts dataset pruning as a graph problem with node weights "
        "for intrinsic importance and edge weights for extrinsic "
        "diversity/interaction, yielding a Maximum Weight Clique formulation "
        "(Section 3.3).",
        "Under mild conditions, the unified objective becomes submodular and "
        "admits a greedy approximation guarantee (Section 3.6; Appendix F).",
    )
