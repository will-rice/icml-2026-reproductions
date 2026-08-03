"""Console script entrypoint for lrr-repro CLI."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

from lrr_repro.evidence import (
    build_evidence,
    build_worker_proposal,
    canonical_json,
    validate_evidence,
)
from lrr_repro.provenance import load_verified_inputs, read_manifest, validate_manifest


def command_acquire(args: argparse.Namespace) -> int:
    from scripts.acquire_upstream import acquire_all

    acquire_all(args.project_root, args.cache_dir)
    return 0


def command_audit(args: argparse.Namespace) -> int:
    evidence = build_evidence(args.project_root, args.cache_dir)
    validate_evidence(evidence, args.schema)
    payload = canonical_json(evidence)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(payload)
    print(f"Wrote canonical evidence to {args.output}")
    return 0


def command_validate(args: argparse.Namespace) -> int:
    evidence_bytes = args.evidence.read_bytes()
    evidence = json.loads(evidence_bytes.decode("utf-8"))
    validate_evidence(evidence, args.schema)
    digest = hashlib.sha256(evidence_bytes).hexdigest()

    val_record = {
        "status": "passed",
        "evidence_sha256": digest,
        "schema_path": str(args.schema.relative_to(args.project_root))
        if args.schema.is_relative_to(args.project_root)
        else str(args.schema),
    }

    if args.validation_output:
        args.validation_output.parent.mkdir(parents=True, exist_ok=True)
        args.validation_output.write_bytes(canonical_json(val_record))
        print(f"Wrote validation record to {args.validation_output}")
    else:
        print(f"Validation passed for evidence SHA-256: {digest}")
    return 0


def command_propose(args: argparse.Namespace) -> int:
    evidence_bytes = args.evidence.read_bytes()
    proposal = build_worker_proposal(
        evidence_bytes, args.source_commit, args.source_tree
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(canonical_json(proposal))
    print(f"Wrote worker proposal to {args.output}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="lrr-repro",
        description="Learning Randomized Reductions reproduction CLI",
    )
    subparsers = parser.add_subparsers(dest="subcommand", required=True)

    # acquire
    p_acq = subparsers.add_parser("acquire", help="Acquire upstream inputs")
    p_acq.add_argument(
        "--manifest",
        type=Path,
        default=Path("evidence/inputs/upstream_manifest.json"),
    )
    p_acq.add_argument(
        "--cache-dir", type=Path, default=Path(".cache/upstream")
    )
    p_acq.add_argument("--project-root", type=Path, default=Path("."))

    # audit
    p_audit = subparsers.add_parser("audit", help="Run offline reproduction audit")
    p_audit.add_argument("--project-root", type=Path, default=Path("."))
    p_audit.add_argument(
        "--cache-dir", type=Path, default=Path(".cache/upstream")
    )
    p_audit.add_argument(
        "--schema", type=Path, default=Path("schema/evidence-v1.schema.json")
    )
    p_audit.add_argument(
        "--output", type=Path, default=Path("evidence/results.json")
    )

    # validate
    p_val = subparsers.add_parser("validate", help="Validate evidence JSON against schema")
    p_val.add_argument(
        "evidence", type=Path, nargs="?", default=Path("evidence/results.json")
    )
    p_val.add_argument(
        "--schema", type=Path, default=Path("schema/evidence-v1.schema.json")
    )
    p_val.add_argument("--project-root", type=Path, default=Path("."))
    p_val.add_argument(
        "--validation-output",
        type=Path,
        default=None,
        help="Optional path to save validation.json",
    )

    # propose
    p_prop = subparsers.add_parser("propose", help="Generate worker proposal JSON")
    p_prop.add_argument(
        "--evidence", type=Path, default=Path("evidence/results.json")
    )
    p_prop.add_argument("--source-commit", type=str, required=True)
    p_prop.add_argument("--source-tree", type=str, required=True)
    p_prop.add_argument(
        "--output", type=Path, default=Path("evidence/worker-proposal.json")
    )

    args = parser.parse_args(argv)

    if args.subcommand == "acquire":
        return command_acquire(args)
    elif args.subcommand == "audit":
        return command_audit(args)
    elif args.subcommand == "validate":
        return command_validate(args)
    elif args.subcommand == "propose":
        return command_propose(args)
    return 1


if __name__ == "__main__":
    sys.exit(main())
