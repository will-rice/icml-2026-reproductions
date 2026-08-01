from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from interplay_repro.evidence import write_evidence


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Interplay-LM evidence bundle")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    bundle = write_evidence(Path(args.output))
    print(json.dumps({"output": args.output, "claims": len(bundle["claim_results"])}, sort_keys=True))


if __name__ == "__main__":
    main()
