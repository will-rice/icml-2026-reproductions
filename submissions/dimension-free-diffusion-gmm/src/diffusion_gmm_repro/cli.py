"""CLI subcommands for running restartable evidence cells, assembling schema-v2 bundles, and verifying results."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from diffusion_gmm_repro.claims import LIVE_CLAIMS
from diffusion_gmm_repro.runner import (
    ExperimentConfig,
    assemble_bundle,
    run_cells,
)


def parser() -> argparse.ArgumentParser:
    main_parser = argparse.ArgumentParser(description=__doc__)
    subparsers = main_parser.add_subparsers(dest="subcommand", required=True)

    # pilot subcommand
    pilot_parser = subparsers.add_parser("pilot", help="Run pilot evidence cells.")
    pilot_parser.add_argument("--output-dir", type=Path, default=Path("evidence/v2"))
    pilot_parser.add_argument("--small-test", action="store_true", help="Run fast reduced test cells.")
    pilot_parser.add_argument("--deadline", type=float, default=None, help="Deadline in seconds from now.")
    pilot_parser.add_argument("--max-rss-gb", type=float, default=None, help="Max RSS memory limit in GB.")

    # scaled subcommand
    scaled_parser = subparsers.add_parser("scaled", help="Run full scaled evidence cells.")
    scaled_parser.add_argument("--output-dir", type=Path, default=Path("evidence/v2"))
    scaled_parser.add_argument("--small-test", action="store_true", help="Run fast reduced test cells.")
    scaled_parser.add_argument("--deadline", type=float, default=None, help="Deadline in seconds from now.")
    scaled_parser.add_argument("--max-rss-gb", type=float, default=None, help="Max RSS memory limit in GB.")

    # assemble subcommand
    assemble_parser = subparsers.add_parser("assemble", help="Assemble evidence shards into schema v2 bundle.")
    assemble_parser.add_argument("--output-dir", type=Path, default=Path("evidence/v2"))
    assemble_parser.add_argument("--mode", choices=["pilot", "scaled"], default="pilot")

    # verify subcommand
    verify_parser = subparsers.add_parser("verify", help="Verify assembled schema v2 evidence bundle.")
    verify_parser.add_argument("--output-dir", type=Path, default=Path("evidence/v2"))

    return main_parser


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)

    if args.subcommand in ("pilot", "scaled"):
        config = ExperimentConfig.pilot() if args.subcommand == "pilot" else ExperimentConfig.scaled()
        deadline_mono = time.monotonic() + args.deadline if args.deadline is not None else None
        max_rss_bytes = int(args.max_rss_gb * 1024**3) if args.max_rss_gb is not None else None

        res = run_cells(
            config,
            output_dir=args.output_dir,
            small_test=args.small_test,
            deadline_monotonic=deadline_mono,
            max_rss_bytes=max_rss_bytes,
        )
        print(f"{args.subcommand} cells finished with status: {res['status']}")
        return 0

    elif args.subcommand == "assemble":
        config = ExperimentConfig.pilot() if args.mode == "pilot" else ExperimentConfig.scaled()
        bundle = assemble_bundle(config, output_dir=args.output_dir)
        print(f"Assembled schema v2 bundle with {len(bundle['cells'])} cells at {args.output_dir}")
        return 0

    elif args.subcommand == "verify":
        results_file = args.output_dir / "results.json"
        csv_file = args.output_dir / "measurements.csv"

        if not results_file.exists():
            print(f"Error: {results_file} does not exist", file=sys.stderr)
            return 1
        if not csv_file.exists():
            print(f"Error: {csv_file} does not exist", file=sys.stderr)
            return 1

        data = json.loads(results_file.read_text(encoding="utf-8"))
        if data.get("schema_version") != 2:
            print(f"Error: unexpected schema version {data.get('schema_version')}", file=sys.stderr)
            return 1

        catalog_digests = [c["digest"] for c in data.get("claim_catalog", {}).get("claims", [])]
        expected_digests = [c.digest for c in LIVE_CLAIMS]
        if catalog_digests != expected_digests:
            print("Error: claim digests mismatch", file=sys.stderr)
            return 1

        print(f"Schema v2 evidence bundle at {args.output_dir} verified successfully.")
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())
