import argparse
import json
import os
from pathlib import Path
import sys
import tempfile

from conditional_dpo_repro.cpo import run_cpo_margin_lane
from conditional_dpo_repro.equivalence import run_equivalence_lane
from conditional_dpo_repro.evidence import (
    build_evidence,
    canonical_json_bytes,
    validate_evidence,
)
from conditional_dpo_repro.failure_modes import (
    run_relative_advantage_lane,
    run_undesirable_space_lane,
)
from conditional_dpo_repro.soft_margin import run_soft_margin_lane

LANE_MAP = {
    "equivalence": run_equivalence_lane,
    "relative-advantage": run_relative_advantage_lane,
    "undesirable-space": run_undesirable_space_lane,
    "cpo-margin": run_cpo_margin_lane,
    "soft-margin": run_soft_margin_lane,
}


def atomic_write(path: Path, payload: bytes) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def command_generate(args: argparse.Namespace) -> int:
    project_root = args.project_root.resolve()
    value = build_evidence(project_root)
    validate_evidence(value, project_root / "schema/evidence-v1.schema.json")
    atomic_write(args.output, canonical_json_bytes(value))
    return 0


def command_validate(args: argparse.Namespace) -> int:
    project_root = args.project_root.resolve()
    payload = json.loads(args.evidence.read_text("utf-8"))
    validate_evidence(payload, project_root / "schema/evidence-v1.schema.json")
    return 0


def command_run_lane(args: argparse.Namespace) -> int:
    func = LANE_MAP.get(args.lane)
    if not func:
        raise ValueError(f"unknown lane: {args.lane}")
    result = func()
    payload = canonical_json_bytes(result)
    if args.output:
        atomic_write(args.output, payload)
    else:
        sys.stdout.buffer.write(payload)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="conditional-dpo-repro",
        description="Finite-response reproduction CLI for conditional DPO/RLHF claims",
    )
    subparsers = parser.add_subparsers(dest="subcommand", required=True)

    gen_parser = subparsers.add_parser("generate", help="Generate evidence bundle")
    gen_parser.add_argument(
        "--project-root",
        type=Path,
        default=Path("."),
        help="Path to project root",
    )
    gen_parser.add_argument(
        "--output",
        type=Path,
        default=Path("evidence.json"),
        help="Path to output evidence JSON",
    )
    gen_parser.set_defaults(handler=command_generate)

    val_parser = subparsers.add_parser("validate", help="Validate evidence bundle")
    val_parser.add_argument(
        "--project-root",
        type=Path,
        default=Path("."),
        help="Path to project root",
    )
    val_parser.add_argument(
        "--evidence",
        type=Path,
        default=Path("evidence.json"),
        help="Path to evidence JSON",
    )
    val_parser.set_defaults(handler=command_validate)

    lane_parser = subparsers.add_parser("run-lane", help="Run a specific evaluation lane")
    lane_parser.add_argument(
        "lane",
        choices=sorted(LANE_MAP.keys()),
        help="Name of lane to run",
    )
    lane_parser.add_argument(
        "--output",
        type=Path,
        help="Optional path to output JSON",
    )
    lane_parser.set_defaults(handler=command_run_lane)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.handler(args)
    except (KeyError, OSError, TypeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
