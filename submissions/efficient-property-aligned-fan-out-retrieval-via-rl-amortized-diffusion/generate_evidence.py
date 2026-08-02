from __future__ import annotations

import argparse
from pathlib import Path

from r4t_repro.evidence import build_evidence_bundle, write_evidence


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate R4T reproduction evidence.")
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output or Path(__file__).resolve().parent / "evidence" / "bundle.json"
    bundle = build_evidence_bundle()
    write_evidence(bundle, output)
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
