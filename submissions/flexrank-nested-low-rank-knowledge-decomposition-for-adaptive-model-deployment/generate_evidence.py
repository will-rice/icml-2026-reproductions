#!/usr/bin/env python3
"""Generate reproduction evidence bundle for FlexRank."""

from pathlib import Path
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from flexrank_repro.evidence import run_evidence_generation


def main():
    output_dir = SCRIPT_DIR / "evidence"
    print(f"Generating evidence in {output_dir}...")
    bundle = run_evidence_generation(output_dir)
    print("Evidence generation completed successfully.")
    print(f"Bundle contents: {bundle}")


if __name__ == "__main__":
    main()
