from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

from venusbench_mobile_repro.evidence import (
    EVIDENCE_PATH,
    EXPECTED_UPSTREAM_COMMIT,
    UPSTREAM_BRANCH,
    UPSTREAM_REPOSITORY,
    build_evidence_bundle,
)


def _clone_upstream() -> Path:
    destination = Path(tempfile.mkdtemp(prefix="venusbench-mobile-upstream-"))
    subprocess.run(
        [
            "git",
            "clone",
            "--branch",
            UPSTREAM_BRANCH,
            "--depth",
            "1",
            UPSTREAM_REPOSITORY,
            str(destination),
        ],
        check=True,
    )
    return destination


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-root",
        type=Path,
        help="Pinned UI-Venus checkout. Defaults to cloning the VenusBench-Mobile branch.",
    )
    parser.add_argument("--output", type=Path, default=EVIDENCE_PATH)
    args = parser.parse_args()

    temporary_source = args.source_root is None
    source_root = args.source_root or _clone_upstream()
    try:
        command_log = [
            (
                "git clone --branch VenusBench-Mobile --depth 1 "
                "https://github.com/inclusionAI/UI-Venus.git"
            ),
            f"git -C {source_root} rev-parse HEAD -> {EXPECTED_UPSTREAM_COMMIT}",
            "uv run pytest tests/test_evidence_bundle.py",
        ]
        bundle = build_evidence_bundle(source_root=source_root, command_log=command_log)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(bundle, indent=2) + "\n", encoding="utf-8")
        print(f"Wrote evidence bundle to {args.output}")
    finally:
        if temporary_source:
            shutil.rmtree(source_root, ignore_errors=True)


if __name__ == "__main__":
    main()
