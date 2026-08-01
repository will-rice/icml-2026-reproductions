from __future__ import annotations

import argparse
from pathlib import Path

from grokking_repro.evidence import run_evidence


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="evidence")
    parser.add_argument("--steps", type=int, default=160)
    args = parser.parse_args()
    run_evidence(
        output_dir=Path(__file__).resolve().parent / args.output_dir,
        steps=args.steps,
    )


if __name__ == "__main__":
    main()
