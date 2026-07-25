"""Build and audit the pinned Numina Putnam 2025 companion proofs."""

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
    verify_clean_checkout,
)


PINNED_SHA = "60d33c8ba19af905bd731e938ebde1c5b8c76519"
REPOSITORY_URL = "https://github.com/project-numina/Numina-Putnam2025.git"
LEAN_TOOLCHAIN = "leanprover/lean4:v4.26.0"
MATHLIB_REVISION = "v4.26.0"
MATHLIB_SHA = "2df2f0150c275ad53cb3c90f7c98ec15a56a1a67"
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
AXIOM_PATTERN = re.compile(r"depends on axioms:\s*\[([^\]]*)\]")


def run(command: list[str], *, cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, cwd=cwd, check=False, text=True, capture_output=True)


def ensure_checkout(checkout: Path) -> None:
    if not checkout.exists():
        checkout.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["git", "clone", "--filter=blob:none", REPOSITORY_URL, str(checkout)],
            check=True,
        )
    subprocess.run(["git", "fetch", "origin", PINNED_SHA], cwd=checkout, check=True)
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


def parse_axioms(stdout: str, proof_name: str) -> list[str]:
    match = AXIOM_PATTERN.search(stdout)
    if match is None or f"'{proof_name}'" not in stdout:
        raise RuntimeError(f"could not parse axiom output for {proof_name}")
    return sorted(
        {name.strip() for name in match.group(1).split(",") if name.strip()}
    )


def audit(checkout: Path, evidence_dir: Path) -> int:
    build_path = evidence_dir / "putnam_build.json"
    axioms_path = evidence_dir / "putnam_axioms.json"
    invalidate_evidence(build_path, axioms_path, evidence_dir / "claims.json")
    ensure_checkout(checkout)
    verify_pins(checkout)
    proof_paths = [
        Path("NuminaPutnam2025") / f"{proof_name}.lean"
        for proof_name in PROOF_NAMES
    ]
    source_audit = scan_lean_sources(checkout, proof_paths)
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

    axioms: dict[str, dict[str, object]] = {}
    for proof_name, relative_file in zip(PROOF_NAMES, proof_paths, strict=True):
        if f"#print axioms {proof_name}" not in (checkout / relative_file).read_text():
            raise RuntimeError(f"missing committed axiom command in {relative_file}")
        result = run(["lake", "env", "lean", str(relative_file)], cwd=checkout)
        if result.returncode:
            raise RuntimeError(f"axiom command failed for {relative_file}")
        axioms[proof_name] = {
            "axioms": parse_axioms(result.stdout, proof_name),
            "command": ["lake", "env", "lean", str(relative_file)],
            "exit_code": result.returncode,
            "lean_toolchain": LEAN_TOOLCHAIN,
            "mathlib_revision": MATHLIB_REVISION,
            "mathlib_sha": MATHLIB_SHA,
            "pinned_sha": PINNED_SHA,
            "repository_url": REPOSITORY_URL,
            "scope": RELEASED_PROOF_SCOPE,
            "upstream_revision": UPSTREAM_REVISION,
        }
    write_json(axioms_path, axioms)
    return 0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkout", type=Path, required=True)
    parser.add_argument("--evidence-dir", type=Path, required=True)
    args = parser.parse_args()
    raise SystemExit(audit(args.checkout.resolve(), args.evidence_dir.resolve()))


if __name__ == "__main__":
    main()
