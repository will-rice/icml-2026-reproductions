#!/usr/bin/env python3
"""Generate or verify the WeDLM machine-readable evidence bundle."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from wedlm_repro.evidence import build_evidence_bundle


def get_git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            check=True,
            text=True,
        )
        return result.stdout.strip()
    except Exception:
        return "unknown"


def comparable(bundle: dict) -> dict:
    ignored = {"timestamp", "git_commit", "environment"}
    return {key: value for key, value in bundle.items() if key not in ignored}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="verify evidence_summary.json without rewriting it")
    args = parser.parse_args()

    out_file = SCRIPT_DIR / "evidence_summary.json"
    bundle = build_evidence_bundle(
        timestamp=datetime.now(timezone.utc).isoformat(),
        git_commit=get_git_commit(),
    )

    if args.check:
        if not out_file.exists():
            print(f"missing {out_file}", file=sys.stderr)
            return 1
        existing = json.loads(out_file.read_text())
        if comparable(existing) != comparable(bundle):
            print("evidence_summary.json is stale", file=sys.stderr)
            return 1
        print(f"verified {out_file}")
        return 0

    out_file.write_text(json.dumps(bundle, indent=2, sort_keys=True) + "\n")
    print(f"wrote {out_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
