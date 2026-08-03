#!/usr/bin/env python3
"""Generate the DiReCT reproduction evidence bundle."""

from pathlib import Path
import sys


PROJECT_DIR = Path(__file__).resolve().parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from direct_repro.evidence import run_evidence_generation


def main() -> None:
    bundle = run_evidence_generation(PROJECT_DIR / "evidence")
    print(f"Wrote evidence bundle for {bundle['paper_id']}")


if __name__ == "__main__":
    main()
