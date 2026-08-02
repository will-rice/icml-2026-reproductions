from __future__ import annotations

import argparse
from pathlib import Path

from hive_repro.evidence import build_evidence_bundle, offline_fixture, write_evidence


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Hive reproduction evidence.")
    parser.add_argument("--offline-fixture", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output or Path(__file__).resolve().parent / "evidence" / "bundle.json"
    if args.offline_fixture:
        source_files, repo_files, ontology, hub_artifacts = offline_fixture()
        bundle = build_evidence_bundle(source_files, repo_files, ontology, hub_artifacts)
    else:
        bundle = build_evidence_bundle()
    write_evidence(bundle, output)
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
