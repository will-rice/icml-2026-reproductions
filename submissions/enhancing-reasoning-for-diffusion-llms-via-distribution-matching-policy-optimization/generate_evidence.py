from __future__ import annotations

import argparse
from pathlib import Path

from dmpo_repro.evidence import write_bundle


ROOT = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate DMPO reproduction evidence.")
    parser.add_argument("--upstream", type=Path, default=None, help="Pinned DMPO checkout; cloned if omitted.")
    parser.add_argument("--output", type=Path, default=ROOT / "evidence" / "bundle.json")
    args = parser.parse_args()
    bundle = write_bundle(args.output, args.upstream)
    print(f"wrote {args.output}")
    print(f"claim statuses: {[claim['status'] for claim in bundle['claims']]}")


if __name__ == "__main__":
    main()
