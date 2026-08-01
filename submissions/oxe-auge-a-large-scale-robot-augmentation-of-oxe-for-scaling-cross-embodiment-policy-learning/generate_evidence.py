from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from oxe_auge_repro.pipeline import generate_evidence_bundle


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate OXE-AugE evidence bundle")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    bundle = generate_evidence_bundle()
    out_file = Path(args.output)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(bundle, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Wrote evidence bundle to {out_file}")


if __name__ == "__main__":
    main()
