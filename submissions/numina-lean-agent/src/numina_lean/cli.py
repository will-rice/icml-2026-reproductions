"""Generate deterministic claim evidence from the released-proof audits."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

from numina_lean import UPSTREAM_REVISION, invalidate_evidence
from numina_lean.brascamp_lieb_audit import (
    FORMALIZATION_SCOPE,
    LEAN_TOOLCHAIN as BRASCAMP_LIEB_TOOLCHAIN,
    MAIN_THEOREM,
    MATHLIB_REVISION as BRASCAMP_LIEB_MATHLIB_REVISION,
    MATHLIB_SHA as BRASCAMP_LIEB_MATHLIB_SHA,
    PINNED_SHA as BRASCAMP_LIEB_SHA,
    QUERY_FILENAME,
    REPOSITORY_URL as BRASCAMP_LIEB_REPOSITORY_URL,
)
from numina_lean.putnam_audit import LEAN_TOOLCHAIN as PUTNAM_TOOLCHAIN
from numina_lean.putnam_audit import MATHLIB_REVISION as PUTNAM_MATHLIB_REVISION
from numina_lean.putnam_audit import MATHLIB_SHA as PUTNAM_MATHLIB_SHA
from numina_lean.putnam_audit import PINNED_SHA as PUTNAM_SHA
from numina_lean.putnam_audit import PROOF_NAMES
from numina_lean.putnam_audit import REPOSITORY_URL as PUTNAM_REPOSITORY_URL


PUTNAM_CLAIM = (
    "Using Claude Opus 4.5, Numina-Lean-Agent solves all 12 Putnam 2025 "
    "problems, matching AXIOM's 12/12 in the comparison table (Table 1)."
)
BRASCAMP_LIEB_CLAIM = (
    "The paper reports successful formalization of the Brascamp-Lieb theorem "
    "through interaction with mathematicians (Abstract)."
)


class EvidenceError(RuntimeError):
    """Raised when claim inputs cannot be safely combined."""


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text())
    except FileNotFoundError as error:
        raise EvidenceError(f"missing evidence input: {path.name}") from error
    except json.JSONDecodeError as error:
        raise EvidenceError(f"invalid JSON evidence input: {path.name}") from error


def input_record(path: Path) -> dict[str, str]:
    return {"sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def require_mapping(value: Any, filename: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise EvidenceError(f"{filename} must contain a JSON object")
    return value


def require_provenance(
    records: list[tuple[str, dict[str, Any]]],
    *,
    command_for_label: dict[str, list[str]],
    lean_toolchain: str,
    mathlib_revision: str,
    mathlib_sha: str,
    pinned_sha: str,
    repository_url: str,
) -> None:
    for label, record in records:
        expected = {
            "command": command_for_label[label],
            "lean_toolchain": lean_toolchain,
            "mathlib_revision": mathlib_revision,
            "mathlib_sha": mathlib_sha,
            "pinned_sha": pinned_sha,
            "repository_url": repository_url,
            "scope": "released-proof verification; not agent re-execution",
            "upstream_revision": UPSTREAM_REVISION,
        }
        for field, value in expected.items():
            if record.get(field) != value:
                raise EvidenceError(f"{label} has an unexpected {field}")
        if type(record.get("exit_code")) is not int:
            raise EvidenceError(f"{label} has an invalid exit_code")


def require_axiom_list(record: dict[str, Any], label: str) -> list[str]:
    axioms = record.get("axioms")
    if (
        not isinstance(axioms, list)
        or not all(isinstance(name, str) for name in axioms)
        or axioms != sorted(set(axioms))
    ):
        raise EvidenceError(f"{label} has an invalid axioms list")
    return axioms


def require_source_audit(
    record: dict[str, Any], label: str, *, file_count: int
) -> dict[str, Any]:
    source_audit = record.get("source_audit")
    if not isinstance(source_audit, dict):
        raise EvidenceError(f"{label} has an invalid source_audit")
    files_with_sorry = source_audit.get("files_with_sorry")
    sorry_count = source_audit.get("sorry_count")
    valid_counts = isinstance(files_with_sorry, dict) and all(
        isinstance(path, str) and type(count) is int and count > 0
        for path, count in files_with_sorry.items()
    )
    if (
        source_audit.get("file_count") != file_count
        or source_audit.get("method")
        != "nested-comment/string-aware sorry token scan"
        or not valid_counts
        or type(sorry_count) is not int
        or sorry_count < 0
        or sorry_count != sum(files_with_sorry.values())
    ):
        raise EvidenceError(f"{label} has an invalid source_audit")
    return source_audit


def putnam_claim(evidence_dir: Path) -> dict[str, Any]:
    build_path = evidence_dir / "putnam_build.json"
    axioms_path = evidence_dir / "putnam_axioms.json"
    build = require_mapping(read_json(build_path), build_path.name)
    axioms = require_mapping(read_json(axioms_path), axioms_path.name)
    if list(axioms) != sorted(PROOF_NAMES):
        raise EvidenceError("putnam_axioms.json does not contain exactly 12 proofs")

    axiom_records: list[tuple[str, dict[str, Any]]] = []
    for proof_name in PROOF_NAMES:
        record = axioms.get(proof_name)
        if not isinstance(record, dict):
            raise EvidenceError(f"missing axiom record for {proof_name}")
        if record.get("exit_code") != 0:
            raise EvidenceError(f"axiom query did not succeed for {proof_name}")
        label = f"putnam_axioms.json:{proof_name}"
        require_axiom_list(record, label)
        axiom_records.append((label, record))
    command_for_label = {"putnam_build.json": ["lake", "build"]}
    command_for_label.update(
        {
            label: [
                "lake",
                "env",
                "lean",
                f"NuminaPutnam2025/{proof_name}.lean",
            ]
            for (label, _), proof_name in zip(
                axiom_records, PROOF_NAMES, strict=True
            )
        }
    )
    require_provenance(
        [("putnam_build.json", build), *axiom_records],
        command_for_label=command_for_label,
        lean_toolchain=PUTNAM_TOOLCHAIN,
        mathlib_revision=PUTNAM_MATHLIB_REVISION,
        mathlib_sha=PUTNAM_MATHLIB_SHA,
        pinned_sha=PUTNAM_SHA,
        repository_url=PUTNAM_REPOSITORY_URL,
    )

    sorry_ax_count = sum(
        "sorryAx" in require_axiom_list(record, label)
        for label, record in axiom_records
    )
    source_sorry_count = require_source_audit(
        build, "putnam_build.json", file_count=12
    )["sorry_count"]
    build_exit_code = build.get("exit_code")
    supports = (
        build_exit_code == 0
        and sorry_ax_count == 0
        and source_sorry_count == 0
    )
    return {
        "claim": PUTNAM_CLAIM,
        "claim_id": "putnam-12-12",
        "computed_observation": {
            "build_exit_code": build_exit_code,
            "kernel_checked_proof_count": len(axiom_records),
            "proof_names": PROOF_NAMES,
            "sorry_ax_count": sorry_ax_count,
            "source_sorry_count": source_sorry_count,
        },
        "evidence_kind": "released-proof-verification",
        "input_files": {
            build_path.name: input_record(build_path),
            axioms_path.name: input_record(axioms_path),
        },
        "limitations": [
            "Does not rerun Numina-Lean-Agent or Claude Opus 4.5.",
            (
                "Verifies the released companion proofs, not the agent-attribution "
                "or comparison-table experiment."
            ),
        ],
        "status": "partial-support" if supports else "does-not-support",
        "supported_component": (
            "The 12 released companion proofs kernel-check without sorryAx."
        ),
        "upstream_revision": UPSTREAM_REVISION,
    }


def brascamp_lieb_claim(evidence_dir: Path) -> dict[str, Any]:
    build_path = evidence_dir / "brascamp_lieb_build.json"
    axioms_path = evidence_dir / "brascamp_lieb_axioms.json"
    build = require_mapping(read_json(build_path), build_path.name)
    axioms = require_mapping(read_json(axioms_path), axioms_path.name)
    if list(axioms) != [MAIN_THEOREM]:
        raise EvidenceError(
            "brascamp_lieb_axioms.json does not contain exactly the main theorem"
        )
    theorem = axioms[MAIN_THEOREM]
    if not isinstance(theorem, dict):
        raise EvidenceError(f"missing axiom record for {MAIN_THEOREM}")
    if theorem.get("exit_code") != 0:
        raise EvidenceError(f"axiom query did not succeed for {MAIN_THEOREM}")
    if theorem.get("formalization_scope") != FORMALIZATION_SCOPE:
        raise EvidenceError("Brascamp-Lieb formalization_scope does not match")
    require_provenance(
        [
            ("brascamp_lieb_build.json", build),
            (f"brascamp_lieb_axioms.json:{MAIN_THEOREM}", theorem),
        ],
        command_for_label={
            "brascamp_lieb_build.json": ["lake", "build"],
            f"brascamp_lieb_axioms.json:{MAIN_THEOREM}": [
                "lake",
                "env",
                "lean",
                QUERY_FILENAME,
            ],
        },
        lean_toolchain=BRASCAMP_LIEB_TOOLCHAIN,
        mathlib_revision=BRASCAMP_LIEB_MATHLIB_REVISION,
        mathlib_sha=BRASCAMP_LIEB_MATHLIB_SHA,
        pinned_sha=BRASCAMP_LIEB_SHA,
        repository_url=BRASCAMP_LIEB_REPOSITORY_URL,
    )

    axiom_names = require_axiom_list(
        theorem, f"brascamp_lieb_axioms.json:{MAIN_THEOREM}"
    )
    source_sorry_count = require_source_audit(
        build, "brascamp_lieb_build.json", file_count=21
    )["sorry_count"]
    sorry_ax_present = "sorryAx" in axiom_names
    build_exit_code = build.get("exit_code")
    supports = (
        build_exit_code == 0
        and not sorry_ax_present
        and source_sorry_count == 0
    )
    return {
        "claim": BRASCAMP_LIEB_CLAIM,
        "claim_id": "brascamp-lieb-formalization",
        "computed_observation": {
            "axioms": axiom_names,
            "build_exit_code": build_exit_code,
            "formalization_scope": FORMALIZATION_SCOPE,
            "sorry_ax_present": sorry_ax_present,
            "source_sorry_count": source_sorry_count,
            "theorem": MAIN_THEOREM,
        },
        "evidence_kind": "released-proof-verification",
        "input_files": {
            build_path.name: input_record(build_path),
            axioms_path.name: input_record(axioms_path),
        },
        "limitations": [
            (
                "Checks the released Gaussian supremum bound, not the full "
                "analytic function-space Brascamp-Lieb theorem."
            ),
            (
                "Does not verify interaction with mathematicians or rerun "
                "Numina-Lean-Agent."
            ),
            "The released formal statement assumes nonzero ambient dimension.",
        ],
        "status": "partial-support" if supports else "does-not-support",
        "supported_component": (
            "The released BrascampLieb.upperBound Gaussian supremum declaration "
            "kernel-checks without sorryAx."
        ),
        "upstream_revision": UPSTREAM_REVISION,
    }


def build_claims(evidence_dir: Path) -> list[dict[str, Any]]:
    return [putnam_claim(evidence_dir), brascamp_lieb_claim(evidence_dir)]


def write_claims(path: Path, claims: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        temporary.write_text(json.dumps(claims, indent=2, sort_keys=True) + "\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def parser() -> argparse.ArgumentParser:
    argument_parser = argparse.ArgumentParser(description=__doc__)
    argument_parser.add_argument(
        "--evidence-dir",
        type=Path,
        default=Path("evidence"),
        help="directory containing audit JSON and receiving claims.json",
    )
    return argument_parser


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    evidence_dir = args.evidence_dir.resolve()
    claims_path = evidence_dir / "claims.json"
    invalidate_evidence(claims_path)
    try:
        claims = build_claims(evidence_dir)
    except EvidenceError as error:
        parser().exit(2, f"error: {error}\n")
    write_claims(claims_path, claims)
    return 0


if __name__ == "__main__":
    sys.exit(main())
