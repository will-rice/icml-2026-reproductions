#!/usr/bin/env python3
"""Generate canonical offline AGoQ evidence from pinned checked-in inputs."""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from agoq_repro.evidence import write_evidence


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "evidence.json",
        help="Canonical JSON output path",
    )
    args = parser.parse_args()
    output = args.output.resolve()
    write_evidence(PROJECT_ROOT, output)
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    print(f"Wrote {output} sha256={digest}")


if __name__ == "__main__":
    main()
