from __future__ import annotations

import argparse
from pathlib import Path

from memorization_repro.evidence import run_evidence


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="evidence")
    parser.add_argument("--train-steps", type=int, default=2)
    args = parser.parse_args()
    run_evidence(
        output_dir=Path(__file__).resolve().parent / args.output_dir,
        train_steps=args.train_steps,
    )


if __name__ == "__main__":
    main()
