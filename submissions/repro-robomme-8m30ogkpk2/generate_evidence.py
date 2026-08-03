from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from robomme_repro.evidence import build_bundle


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=Path("/tmp/icml-robomme-repo"),
        help="Pinned RoboMME benchmark repository checkout.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("evidence/bundle.json"),
        help="Path for the deterministic evidence bundle.",
    )
    args = parser.parse_args()
    bundle = build_bundle(args.artifact_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(bundle, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
