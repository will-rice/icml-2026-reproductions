import json
import subprocess
from pathlib import Path

import pytest

from numina_lean import brascamp_lieb_audit


BL_SHA = "413f2bfd31100187eb6c2d632c9cbf12e3115494"
LEAN_TOOLCHAIN = "leanprover/lean4:v4.28.0"
MATHLIB_SHA = "8f9d9cff6bd728b17a24e163c9402775d9e6a365"
MAIN_THEOREM = "BrascampLieb.upperBound"
UPSTREAM_REVISION = (
    "github:project-numina/numina-lean-agent@"
    "1c9af8a52e715f22fede766425ba3d3b95526132+"
    "project-numina/Numina-Putnam2025@"
    "60d33c8ba19af905bd731e938ebde1c5b8c76519+"
    "project-numina/BrascampLieb@"
    "413f2bfd31100187eb6c2d632c9cbf12e3115494"
)
EVIDENCE = Path(__file__).parents[1] / "evidence"


def load_evidence(filename: str) -> object:
    return json.loads((EVIDENCE / filename).read_text())


def test_brascamp_lieb_build_succeeded_at_pinned_revision() -> None:
    data = load_evidence("brascamp_lieb_build.json")

    assert data["exit_code"] == 0
    assert data["pinned_sha"] == BL_SHA
    assert data["lean_toolchain"] == LEAN_TOOLCHAIN
    assert data["mathlib_revision"] == "v4.28.0"
    assert data["mathlib_sha"] == MATHLIB_SHA
    assert data["command"] == ["lake", "build"]
    assert data["scope"] == "released-proof verification; not agent re-execution"
    assert data["upstream_revision"] == UPSTREAM_REVISION
    assert data["source_audit"] == {
        "file_count": 21,
        "files_with_sorry": {},
        "method": "nested-comment/string-aware sorry token scan",
        "sorry_count": 0,
    }


def test_brascamp_lieb_main_theorem_has_no_sorry_ax() -> None:
    data = load_evidence("brascamp_lieb_axioms.json")
    result = data[MAIN_THEOREM]

    assert sorted(data) == [MAIN_THEOREM]
    assert result["command"] == [
        "lake",
        "env",
        "lean",
        "axiom_check_bl.lean",
    ]
    assert result["exit_code"] == 0
    assert result["lean_toolchain"] == LEAN_TOOLCHAIN
    assert result["mathlib_sha"] == MATHLIB_SHA
    assert result["pinned_sha"] == BL_SHA
    assert result["scope"] == "released-proof verification; not agent re-execution"
    assert result["formalization_scope"] == (
        "BrascampLieb.upperBound Gaussian supremum bound from the Numina paper "
        "Appendix A.1; not the full analytic function-space theorem"
    )
    assert "sorryAx" not in result["axioms"]
    assert result["upstream_revision"] == UPSTREAM_REVISION


def test_brascamp_lieb_axiom_audit_is_normalized() -> None:
    path = EVIDENCE / "brascamp_lieb_axioms.json"
    data = load_evidence(path.name)
    axioms = data[MAIN_THEOREM]["axioms"]

    assert "observed_at" not in data
    assert path.read_text() == json.dumps(data, indent=2, sort_keys=True) + "\n"
    assert axioms == sorted(set(axioms))


def test_brascamp_lieb_evidence_is_locally_authored_json_only() -> None:
    evidence_files = [path for path in EVIDENCE.glob("**/*") if path.is_file()]

    assert evidence_files
    assert all(path.suffix == ".json" for path in evidence_files)


def test_failed_axiom_query_invalidates_stale_brascamp_lieb_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    evidence_dir = tmp_path / "evidence"
    evidence_dir.mkdir()
    stale_axioms = evidence_dir / "brascamp_lieb_axioms.json"
    stale_axioms.write_text('{"stale": true}\n')
    stale_claims = evidence_dir / "claims.json"
    stale_claims.write_text('{"stale": true}\n')

    monkeypatch.setattr(brascamp_lieb_audit, "ensure_checkout", lambda checkout: None)
    monkeypatch.setattr(brascamp_lieb_audit, "verify_pins", lambda checkout: None)
    monkeypatch.setattr(
        brascamp_lieb_audit, "tracked_lean_sources", lambda checkout: []
    )
    monkeypatch.setattr(
        brascamp_lieb_audit.subprocess,
        "run",
        lambda command, **kwargs: brascamp_lieb_audit.subprocess.CompletedProcess(
            command, 0, stdout="", stderr=""
        ),
    )
    monkeypatch.setattr(
        brascamp_lieb_audit,
        "run_axiom_query",
        lambda checkout: brascamp_lieb_audit.subprocess.CompletedProcess(
            ["lake", "env", "lean", brascamp_lieb_audit.QUERY_FILENAME],
            1,
            stdout="",
            stderr="query failed",
        ),
    )

    with pytest.raises(RuntimeError, match="axiom command failed"):
        brascamp_lieb_audit.audit(tmp_path / "checkout", evidence_dir)

    assert not stale_axioms.exists()
    assert not stale_claims.exists()


def test_brascamp_lieb_checkout_must_be_clean(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=checkout, check=True)
    (checkout / "Main.lean").write_text("theorem clean : True := by trivial\n")
    subprocess.run(["git", "add", "Main.lean"], cwd=checkout, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Numina test",
            "-c",
            "user.email=numina-test@example.invalid",
            "commit",
            "-qm",
            "fixture",
        ],
        cwd=checkout,
        check=True,
    )
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=checkout,
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()
    monkeypatch.setattr(brascamp_lieb_audit, "PINNED_SHA", head)
    (checkout / "untracked.lean").write_text("theorem dirty : True := by trivial\n")

    with pytest.raises(RuntimeError, match="checkout is not clean"):
        brascamp_lieb_audit.ensure_checkout(checkout)
