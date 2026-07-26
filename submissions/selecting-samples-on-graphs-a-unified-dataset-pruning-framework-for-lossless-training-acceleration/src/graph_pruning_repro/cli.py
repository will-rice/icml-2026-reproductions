"""Command-line interface for canonical evidence generation and acceptance."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from .evidence import build_evidence, validate_evidence


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="graph-pruning-repro")
    commands = parser.add_subparsers(dest="command", required=True)

    recompute = commands.add_parser("recompute")
    recompute.add_argument("output_dir", type=Path)
    recompute.add_argument("--source-revision", required=True)

    validate = commands.add_parser("validate")
    validate.add_argument("evidence", type=Path)

    render = commands.add_parser("render")
    render.add_argument("evidence", type=Path)
    render.add_argument("output_dir", type=Path)
    return parser


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _validation_root(evidence_path: Path) -> Path:
    if evidence_path.parent.name == "evidence":
        candidate = evidence_path.parent.parent
        if (candidate / "paper_transcriptions").is_dir():
            return candidate
    return evidence_path.parent


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        if arguments.command == "recompute":
            evidence = build_evidence(
                arguments.output_dir,
                source_revision=arguments.source_revision,
            )
            command = next(
                record
                for record in evidence["commands"]
                if record["id"] == "recompute"
            )
            print(
                f"completed actual={command['actual']} "
                f"ceiling={command['ceiling']}"
            )
            return 0

        if arguments.command == "validate":
            validate_evidence(
                arguments.evidence,
                _project_root() / "evidence" / "schema.json",
                _validation_root(arguments.evidence),
            )
            print("schema and full-replay semantic acceptance: PASS")
            return 0

        try:
            from .render import render_poster, render_report
        except ImportError as exc:
            raise ValueError(
                "render implementation is introduced by approved Task 7"
            ) from exc
        evidence = json.loads(arguments.evidence.read_text())
        arguments.output_dir.mkdir(parents=True, exist_ok=True)
        (arguments.output_dir / "report.md").write_text(
            render_report(evidence)
        )
        (arguments.output_dir / "poster.html").write_text(
            render_poster(evidence)
        )
        return 0
    except (OSError, TypeError, ValueError) as exc:
        print(f"acceptance failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
