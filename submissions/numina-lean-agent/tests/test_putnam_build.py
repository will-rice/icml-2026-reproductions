import json
from pathlib import Path

import pytest

from numina_lean import putnam_audit


COMPANION_SHA = "60d33c8ba19af905bd731e938ebde1c5b8c76519"
LEAN_TOOLCHAIN = "leanprover/lean4:v4.26.0"
UPSTREAM_REVISION = (
    "github:project-numina/numina-lean-agent@"
    "1c9af8a52e715f22fede766425ba3d3b95526132+"
    "project-numina/Numina-Putnam2025@"
    "60d33c8ba19af905bd731e938ebde1c5b8c76519+"
    "project-numina/BrascampLieb@"
    "413f2bfd31100187eb6c2d632c9cbf12e3115494"
)
PROOF_NAMES = [
    f"putnam_2025_{problem}"
    for problem in (
        "a1",
        "a2",
        "a3",
        "a4",
        "a5",
        "a6",
        "b1",
        "b2",
        "b3",
        "b4",
        "b5",
        "b6",
    )
]
EVIDENCE = Path(__file__).parents[1] / "evidence"


def load_evidence(filename: str) -> object:
    return json.loads((EVIDENCE / filename).read_text())


def test_build_result_exists() -> None:
    assert (EVIDENCE / "putnam_build.json").exists(), "Run build pipeline first"


def test_build_succeeded_at_pinned_revision() -> None:
    data = load_evidence("putnam_build.json")

    assert data["exit_code"] == 0
    assert data["pinned_sha"] == COMPANION_SHA
    assert data["lean_toolchain"] == LEAN_TOOLCHAIN
    assert data["upstream_revision"] == UPSTREAM_REVISION


def test_all_12_proofs_have_no_sorry_ax() -> None:
    data = load_evidence("putnam_axioms.json")

    assert sorted(data) == PROOF_NAMES
    for proof_name in PROOF_NAMES:
        assert "sorryAx" not in data[proof_name]["axioms"]
        assert data[proof_name]["upstream_revision"] == UPSTREAM_REVISION


def test_axiom_audit_is_normalized_and_tracks_quot_sound() -> None:
    path = EVIDENCE / "putnam_axioms.json"
    data = load_evidence(path.name)

    assert "observed_at" not in data
    assert path.read_text() == json.dumps(data, indent=2, sort_keys=True) + "\n"
    assert all(item["axioms"] == sorted(set(item["axioms"])) for item in data.values())
    assert any("Quot.sound" in item["axioms"] for item in data.values())


def test_failed_axiom_query_invalidates_stale_putnam_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    evidence_dir = tmp_path / "evidence"
    evidence_dir.mkdir()
    stale_axioms = evidence_dir / "putnam_axioms.json"
    stale_axioms.write_text('{"stale": true}\n')

    monkeypatch.setattr(putnam_audit, "ensure_checkout", lambda checkout: None)
    monkeypatch.setattr(putnam_audit, "verify_pins", lambda checkout: None)
    monkeypatch.setattr(putnam_audit, "PROOF_NAMES", ["putnam_2025_a1"])
    proof_dir = tmp_path / "checkout" / "NuminaPutnam2025"
    proof_dir.mkdir(parents=True)
    (proof_dir / "putnam_2025_a1.lean").write_text(
        "#print axioms putnam_2025_a1\n"
    )

    def fake_run(
        command: list[str], *, cwd: Path
    ) -> putnam_audit.subprocess.CompletedProcess[str]:
        return putnam_audit.subprocess.CompletedProcess(
            command,
            0 if command == ["lake", "build"] else 1,
            stdout="",
            stderr="query failed",
        )

    monkeypatch.setattr(
        putnam_audit.subprocess,
        "run",
        lambda command, **kwargs: putnam_audit.subprocess.CompletedProcess(command, 0),
    )
    monkeypatch.setattr(putnam_audit, "run", fake_run)

    with pytest.raises(RuntimeError, match="axiom command failed"):
        putnam_audit.audit(tmp_path / "checkout", evidence_dir)

    assert not stale_axioms.exists()
