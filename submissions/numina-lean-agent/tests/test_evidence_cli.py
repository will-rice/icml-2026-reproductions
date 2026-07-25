from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


PROJECT = Path(__file__).parents[1]
SOURCE_EVIDENCE = PROJECT / "evidence"
SOURCE_ROOT = PROJECT / "src"
INPUT_FILENAMES = (
    "putnam_build.json",
    "putnam_axioms.json",
    "brascamp_lieb_build.json",
    "brascamp_lieb_axioms.json",
)
OFFICIAL_CLAIMS = {
    "putnam-12-12": (
        "Using Claude Opus 4.5, Numina-Lean-Agent solves all 12 Putnam 2025 "
        "problems, matching AXIOM's 12/12 in the comparison table (Table 1)."
    ),
    "brascamp-lieb-formalization": (
        "The paper reports successful formalization of the Brascamp-Lieb theorem "
        "through interaction with mathematicians (Abstract)."
    ),
}
UPSTREAM_REVISION = (
    "github:project-numina/numina-lean-agent@"
    "1c9af8a52e715f22fede766425ba3d3b95526132+"
    "project-numina/Numina-Putnam2025@"
    "60d33c8ba19af905bd731e938ebde1c5b8c76519+"
    "project-numina/BrascampLieb@"
    "413f2bfd31100187eb6c2d632c9cbf12e3115494"
)


def copy_inputs(destination: Path) -> None:
    destination.mkdir()
    for filename in INPUT_FILENAMES:
        shutil.copyfile(SOURCE_EVIDENCE / filename, destination / filename)


def run_cli(evidence_dir: Path) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    existing_pythonpath = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        str(SOURCE_ROOT)
        if not existing_pythonpath
        else f"{SOURCE_ROOT}{os.pathsep}{existing_pythonpath}"
    )
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "numina_lean.cli",
            "--evidence-dir",
            str(evidence_dir),
        ],
        cwd=PROJECT,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )


def test_evidence_cli_writes_deterministic_claims_from_computed_inputs(
    tmp_path: Path,
) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    copy_inputs(first)
    copy_inputs(second)

    first_result = run_cli(first)
    second_result = run_cli(second)

    assert first_result.returncode == 0, first_result.stderr
    assert second_result.returncode == 0, second_result.stderr
    assert (first / "claims.json").read_bytes() == (
        second / "claims.json"
    ).read_bytes()

    claims = json.loads((first / "claims.json").read_text())
    assert [claim["claim_id"] for claim in claims] == list(OFFICIAL_CLAIMS)
    assert {claim["claim_id"]: claim["claim"] for claim in claims} == OFFICIAL_CLAIMS
    assert all(claim["status"] == "supports" for claim in claims)
    assert all(
        claim["evidence_kind"] == "released-proof-verification" for claim in claims
    )
    assert all(claim["upstream_revision"] == UPSTREAM_REVISION for claim in claims)
    assert all(claim["computed_observation"] for claim in claims)
    assert all(claim["limitations"] for claim in claims)
    assert all(
        set(claim["input_files"])
        in (
            {"putnam_build.json", "putnam_axioms.json"},
            {"brascamp_lieb_build.json", "brascamp_lieb_axioms.json"},
        )
        for claim in claims
    )
    assert all(
        len(input_data["sha256"]) == 64
        for claim in claims
        for input_data in claim["input_files"].values()
    )

    by_id = {claim["claim_id"]: claim for claim in claims}
    assert by_id["putnam-12-12"]["computed_observation"] == {
        "build_exit_code": 0,
        "kernel_checked_proof_count": 12,
        "proof_names": [
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
        ],
        "sorry_ax_count": 0,
    }
    assert by_id["brascamp-lieb-formalization"]["computed_observation"] == {
        "axioms": ["Classical.choice", "Quot.sound", "propext"],
        "build_exit_code": 0,
        "formalization_scope": (
            "BrascampLieb.upperBound Gaussian supremum bound from the Numina paper "
            "Appendix A.1; not the full analytic function-space theorem"
        ),
        "sorry_ax_present": False,
        "theorem": "BrascampLieb.upperBound",
    }

    assert (SOURCE_EVIDENCE / "claims.json").read_bytes() == (
        first / "claims.json"
    ).read_bytes()


def test_evidence_cli_rejects_mixed_provenance_and_removes_stale_output(
    tmp_path: Path,
) -> None:
    evidence_dir = tmp_path / "evidence"
    copy_inputs(evidence_dir)
    axiom_path = evidence_dir / "brascamp_lieb_axioms.json"
    axiom_data = json.loads(axiom_path.read_text())
    axiom_data["BrascampLieb.upperBound"]["upstream_revision"] = "tampered"
    axiom_path.write_text(json.dumps(axiom_data, indent=2, sort_keys=True) + "\n")
    claims_path = evidence_dir / "claims.json"
    claims_path.write_text('{"stale": true}\n')

    result = run_cli(evidence_dir)

    assert result.returncode != 0
    assert "upstream_revision" in result.stderr
    assert not claims_path.exists()
