from __future__ import annotations

import json
import re
from pathlib import Path


PROJECT_ROOT = Path(__file__).parents[1]


def _accepted_artifact_bytes() -> dict[Path, bytes]:
    evidence_path = PROJECT_ROOT / "evidence" / "evidence.json"
    evidence = json.loads(evidence_path.read_text())
    paths = [evidence_path]
    paths.extend(
        PROJECT_ROOT / witness["artifact_path"]
        for witness in evidence["witnesses"]
    )
    return {path: path.read_bytes() for path in paths}


def _parse_pass_counts(status: str) -> tuple[int, int]:
    match = re.fullmatch(r"PASS actual=(\d+) ceiling=(\d+)", status)
    assert match is not None
    return int(match.group(1)), int(match.group(2))


def test_space_import_is_offline_and_does_not_mutate_evidence(
    monkeypatch,
) -> None:
    before = _accepted_artifact_bytes()
    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("network used")
        ),
    )

    from app import EVIDENCE_PATH, STARTUP_VALIDATED, demo

    assert EVIDENCE_PATH.name == "evidence.json"
    assert STARTUP_VALIDATED is True
    assert demo is not None
    assert _accepted_artifact_bytes() == before


def test_space_exposes_required_evidence_panels() -> None:
    from app import PANEL_NAMES, demo

    assert {
        "Summary",
        "Variants",
        "Witnesses",
        "Proof ledger",
        "Unavailable claims",
    } <= set(PANEL_NAMES)
    config = str(demo.get_config_file())
    for panel_name in PANEL_NAMES:
        assert panel_name in config


def test_space_exposes_canonical_downloads() -> None:
    from app import DOWNLOAD_PATHS

    relative_paths = {path.as_posix() for path in DOWNLOAD_PATHS}
    assert {
        "evidence/evidence.json",
        "NOTICE.md",
        "LICENSE",
        "LICENSES/CC-BY-NC-SA-4.0.txt",
    } <= relative_paths
    evidence = json.loads(
        (PROJECT_ROOT / "evidence" / "evidence.json").read_text()
    )
    assert {
        witness["artifact_path"] for witness in evidence["witnesses"]
    } <= relative_paths


def test_space_recompute_uses_bounded_evidence_path(tmp_path: Path) -> None:
    from app import recompute

    status, evidence_path = recompute(tmp_path)
    actual, ceiling = _parse_pass_counts(status)
    assert ceiling == 13_833_860
    assert 0 <= actual <= ceiling
    assert evidence_path == tmp_path / "evidence.json"
    assert evidence_path.exists()
