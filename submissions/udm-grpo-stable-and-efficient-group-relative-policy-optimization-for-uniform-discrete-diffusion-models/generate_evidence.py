from __future__ import annotations

import argparse
import json
from pathlib import Path

from udm_grpo_repro import build_evidence_bundle


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="evidence/bundle.json")
    args = parser.parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(build_evidence_bundle(), indent=2) + "\n")


if __name__ == "__main__":
    main()
