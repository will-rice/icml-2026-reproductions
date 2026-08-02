from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from synermedgen_repro import build_evidence_bundle


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="evidence")
    args = parser.parse_args()

    output = Path(args.output_dir) / "bundle.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(build_evidence_bundle(), indent=2) + "\n")


if __name__ == "__main__":
    main()
