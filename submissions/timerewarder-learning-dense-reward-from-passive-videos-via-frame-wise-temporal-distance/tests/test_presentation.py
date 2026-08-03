import json
from html import unescape
from pathlib import Path

import pytest

from timerewarder_repro.presentation import (
    claim_rows,
    load_verified_evidence,
    render_poster,
)


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "artifacts" / "evidence.json"


def test_verified_bundle_drives_claim_rows_and_poster() -> None:
    bundle = load_verified_evidence(EVIDENCE)
    rows = claim_rows(bundle)
    poster = unescape(render_poster(bundle))

    assert len(rows) == 6
    assert [row[1] for row in rows] == [
        "verified",
        "verified",
        "verified",
        "partial",
        "unavailable",
        "unavailable",
    ]
    assert bundle["measurement_sha256"] in poster
    for row in rows:
        assert row[1] in poster
        if row[1] in {"partial", "unavailable"}:
            assert row[3] in poster
    assert "five-video-per-task released-model protocol" in poster
    assert "diagnostic-only" in poster


def test_verified_loader_rejects_tampering(tmp_path: Path) -> None:
    bundle = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    bundle["claims"][0]["status"] = "contradicted"
    tampered = tmp_path / "evidence.json"
    tampered.write_text(json.dumps(bundle), encoding="utf-8")

    with pytest.raises(ValueError, match="measurement hash"):
        load_verified_evidence(tampered)


def test_readme_and_generated_poster_share_canonical_statuses() -> None:
    bundle = load_verified_evidence(EVIDENCE)
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    poster = unescape((ROOT / "poster.html").read_text(encoding="utf-8"))

    assert readme.startswith("---\ntitle: TimeRewarder Reproduction Evidence\n")
    for text in (readme, poster):
        assert bundle["measurement_sha256"] in text
        for index, claim in enumerate(bundle["claims"], start=1):
            assert f"Claim {index}" in text
            assert claim["status"] in text
            if claim["status"] in {"partial", "unavailable"}:
                assert claim["limitations"] in text
    assert "No deployment or official verdict is included" in readme
