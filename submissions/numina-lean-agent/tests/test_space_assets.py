from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

from numina_lean.space_assets import EVIDENCE_FILENAMES, render_assets


PROJECT = Path(__file__).parents[1]
EVIDENCE = PROJECT / "evidence"
UPSTREAM_REVISION = (
    "github:project-numina/numina-lean-agent@"
    "1c9af8a52e715f22fede766425ba3d3b95526132+"
    "project-numina/Numina-Putnam2025@"
    "60d33c8ba19af905bd731e938ebde1c5b8c76519+"
    "project-numina/BrascampLieb@"
    "413f2bfd31100187eb6c2d632c9cbf12e3115494"
)
CLAIM_IDS = ["putnam-12-12", "brascamp-lieb-formalization"]


def test_manifest_hashes_exactly_five_normalized_evidence_files() -> None:
    manifest = json.loads((PROJECT / "manifest.json").read_text())

    assert manifest["schema_version"] == 1
    assert manifest["scope"] == (
        "released-proof verification; not agent re-execution or official verdict"
    )
    assert manifest["upstream_revision"] == UPSTREAM_REVISION
    assert manifest["claim_ids"] == CLAIM_IDS
    assert list(manifest["evidence_files"]) == list(EVIDENCE_FILENAMES)
    for filename in EVIDENCE_FILENAMES:
        path = EVIDENCE / filename
        assert path.read_text().endswith("\n")
        assert json.dumps(
            json.loads(path.read_text()), indent=2, sort_keys=True
        ) + "\n" == path.read_text()
        assert manifest["evidence_files"][filename]["sha256"] == hashlib.sha256(
            path.read_bytes()
        ).hexdigest()


def test_assets_are_deterministic_and_derived_from_claims(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()

    render_assets(EVIDENCE, first)
    render_assets(EVIDENCE, second)

    for filename in ("manifest.json", "report.md", "poster.html", "index.html"):
        assert (first / filename).read_bytes() == (second / filename).read_bytes()
        assert (PROJECT / filename).read_bytes() == (first / filename).read_bytes()

    claims = json.loads((EVIDENCE / "claims.json").read_text())
    report = (first / "report.md").read_text()
    poster = (first / "poster.html").read_text()
    index = (first / "index.html").read_text()
    for claim in claims:
        assert claim["claim_id"] in report
        assert claim["supported_component"] in report
        assert claim["supported_component"] in poster
        assert claim["supported_component"] in index
    assert "12 released companion proofs" in report
    assert "BrascampLieb.upperBound" in poster


def test_space_card_is_valid_static_metadata() -> None:
    readme = (PROJECT / "README.md").read_text()

    assert readme.startswith("---\n")
    metadata = readme.split("---\n", 2)[1]
    assert "sdk: static\n" in metadata
    assert "app_file: index.html\n" in metadata
    assert "  - icml2026-repro\n" in metadata
    assert "  - paper-0bTEd4LpQr\n" in metadata
    assert "Numina-Lean-Agent" in readme
    assert UPSTREAM_REVISION in readme


def test_assets_state_scope_and_license_limitations_truthfully() -> None:
    combined = "\n".join(
        (PROJECT / filename).read_text()
        for filename in ("README.md", "report.md", "poster.html", "index.html")
    )

    assert "released-proof verification" in combined
    assert "partial support" in combined
    assert "not an agent rerun" in combined
    assert "not an official verdict" in combined
    assert "no LICENSE file" in combined
    assert "does not redistribute" in combined

    forbidden_suffixes = {
        ".lean",
        ".olean",
        ".log",
        ".tar",
        ".gz",
        ".zip",
    }
    tracked = subprocess.run(
        ["git", "ls-files", str(PROJECT)],
        cwd=PROJECT.parents[1],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    tracked_files = [PROJECT.parents[1] / path for path in tracked]
    assert not any(path.suffix in forbidden_suffixes for path in tracked_files)
    assert not any("__pycache__" in path.parts for path in tracked_files)
