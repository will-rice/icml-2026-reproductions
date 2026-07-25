"""Released-proof verification for Numina-Lean-Agent."""

from __future__ import annotations

import subprocess
from pathlib import Path


UPSTREAM_REVISION = (
    "github:project-numina/numina-lean-agent@"
    "1c9af8a52e715f22fede766425ba3d3b95526132+"
    "project-numina/Numina-Putnam2025@"
    "60d33c8ba19af905bd731e938ebde1c5b8c76519+"
    "project-numina/BrascampLieb@"
    "413f2bfd31100187eb6c2d632c9cbf12e3115494"
)
RELEASED_PROOF_SCOPE = "released-proof verification; not agent re-execution"


def invalidate_evidence(*paths: Path) -> None:
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.unlink(missing_ok=True)


def verify_clean_checkout(checkout: Path) -> None:
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=checkout,
        check=True,
        text=True,
        capture_output=True,
    ).stdout
    if status:
        raise RuntimeError(f"checkout is not clean: {status.strip()}")
