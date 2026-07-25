"""Build and audit the pinned Brascamp-Lieb formalization."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path

from numina_lean import (
    RELEASED_PROOF_SCOPE,
    UPSTREAM_REVISION,
    invalidate_evidence,
    scan_lean_sources,
    tracked_lean_sources,
    verify_clean_checkout,
)


PINNED_SHA = "413f2bfd31100187eb6c2d632c9cbf12e3115494"
REPOSITORY_URL = "https://github.com/project-numina/BrascampLieb.git"
LEAN_TOOLCHAIN = "leanprover/lean4:v4.28.0"
MATHLIB_REVISION = "v4.28.0"
MATHLIB_SHA = "8f9d9cff6bd728b17a24e163c9402775d9e6a365"
MAIN_THEOREM = "BrascampLieb.upperBound"
QUERY_FILENAME = "axiom_check_bl.lean"
QUERY_SOURCE = """import BrascampLieb.Code.MainTheorems

#print axioms BrascampLieb.upperBound
"""
AXIOM_PATTERN = re.compile(r"depends on axioms:\s*\[([^\]]*)\]")
FORMALIZATION_SCOPE = (
    "BrascampLieb.upperBound Gaussian supremum bound from the Numina paper "
    "Appendix A.1; not the full analytic function-space theorem"
)


def run(command: list[str], *, cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, cwd=cwd, check=False, text=True, capture_output=True)


def ensure_checkout(checkout: Path) -> None:
    if not checkout.exists():
        checkout.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["git", "clone", "--filter=blob:none", REPOSITORY_URL, str(checkout)],
            check=True,
        )
        subprocess.run(
            ["git", "checkout", "--detach", PINNED_SHA], cwd=checkout, check=True
        )

    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=checkout,
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()
    if head != PINNED_SHA:
        raise RuntimeError(f"checkout is at {head}, expected {PINNED_SHA}")
    verify_clean_checkout(checkout)


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def verify_pins(checkout: Path) -> None:
    toolchain = (checkout / "lean-toolchain").read_text().strip()
    manifest = json.loads((checkout / "lake-manifest.json").read_text())
    mathlib = next(
        package for package in manifest["packages"] if package["name"] == "mathlib"
    )
    if toolchain != LEAN_TOOLCHAIN:
        raise RuntimeError(f"unexpected Lean toolchain: {toolchain}")
    if mathlib["inputRev"] != MATHLIB_REVISION or mathlib["rev"] != MATHLIB_SHA:
        raise RuntimeError("unexpected Mathlib revision")


def parse_axioms(stdout: str) -> list[str]:
    match = AXIOM_PATTERN.search(stdout)
    if match is None or f"'{MAIN_THEOREM}'" not in stdout:
        raise RuntimeError(f"could not parse axiom output for {MAIN_THEOREM}")
    return sorted(
        {name.strip() for name in match.group(1).split(",") if name.strip()}
    )


def run_axiom_query(checkout: Path) -> subprocess.CompletedProcess[str]:
    query_path = checkout / QUERY_FILENAME
    if query_path.exists():
        raise RuntimeError(f"refusing to overwrite existing {query_path}")
    try:
        query_path.write_text(QUERY_SOURCE)
        return run(["lake", "env", "lean", QUERY_FILENAME], cwd=checkout)
    finally:
        query_path.unlink(missing_ok=True)


def audit(checkout: Path, evidence_dir: Path) -> int:
    build_path = evidence_dir / "brascamp_lieb_build.json"
    axioms_path = evidence_dir / "brascamp_lieb_axioms.json"
    invalidate_evidence(build_path, axioms_path, evidence_dir / "claims.json")
    ensure_checkout(checkout)
    verify_pins(checkout)
    source_audit = scan_lean_sources(checkout, tracked_lean_sources(checkout))
    subprocess.run(["lake", "exe", "cache", "get"], cwd=checkout, check=True)
    build = run(["lake", "build"], cwd=checkout)
    write_json(
        build_path,
        {
            "command": ["lake", "build"],
            "exit_code": build.returncode,
            "lean_toolchain": LEAN_TOOLCHAIN,
            "mathlib_sha": MATHLIB_SHA,
            "mathlib_revision": MATHLIB_REVISION,
            "pinned_sha": PINNED_SHA,
            "repository_url": REPOSITORY_URL,
            "scope": RELEASED_PROOF_SCOPE,
            "source_audit": source_audit,
            "upstream_revision": UPSTREAM_REVISION,
        },
    )
    if build.returncode:
        return build.returncode

    result = run_axiom_query(checkout)
    if result.returncode:
        raise RuntimeError(f"axiom command failed for {MAIN_THEOREM}")
    write_json(
        axioms_path,
        {
            MAIN_THEOREM: {
                "axioms": parse_axioms(result.stdout),
                "command": ["lake", "env", "lean", QUERY_FILENAME],
                "exit_code": result.returncode,
                "formalization_scope": FORMALIZATION_SCOPE,
                "lean_toolchain": LEAN_TOOLCHAIN,
                "mathlib_revision": MATHLIB_REVISION,
                "mathlib_sha": MATHLIB_SHA,
                "pinned_sha": PINNED_SHA,
                "repository_url": REPOSITORY_URL,
                "scope": RELEASED_PROOF_SCOPE,
                "upstream_revision": UPSTREAM_REVISION,
            }
        },
    )
    return 0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkout", type=Path, required=True)
    parser.add_argument("--evidence-dir", type=Path, required=True)
    args = parser.parse_args()
    raise SystemExit(audit(args.checkout.resolve(), args.evidence_dir.resolve()))


if __name__ == "__main__":
    main()
