#!/usr/bin/env python3
"""Generate the WIRE reproduction evidence bundle."""

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from wire_graph_rope.claims import write_evidence_bundle  # noqa: E402


def main() -> None:
    output = PROJECT_ROOT / "evidence" / "bundle.json"
    write_evidence_bundle(output)
    print(output)


if __name__ == "__main__":
    main()
