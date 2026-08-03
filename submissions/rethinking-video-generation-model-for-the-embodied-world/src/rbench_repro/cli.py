"""CLI for RBench reproduction: acquire, validate-inputs, audit, validate, propose."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

from rbench_repro.evidence import (
    ATTEMPT_ID,
    PAPER_ID,
    AuditInputs,
    build_evidence,
    validate_evidence,
)
from rbench_repro.model import canonical_json, sha256_bytes


def build_worker_proposal(
    evidence_bytes: bytes, source_commit: str, source_tree: str
) -> dict[str, object]:
    """Build a non-authoritative worker proposal for controller validation."""
    if len(source_commit) != 40 or len(source_tree) != 40:
        raise ValueError("source_commit and source_tree must be 40 hex chars")
    return {
        "schema_version": 1,
        "paper_id": PAPER_ID,
        "attempt_id": ATTEMPT_ID,
        "requested_action": "controller_validation",
        "external_mutations": [],
        "source_commit": source_commit,
        "source_tree": source_tree,
        "evidence_sha256": sha256_bytes(evidence_bytes),
    }


def command_acquire(args: argparse.Namespace) -> int:
    """Acquire pinned sources from upstream repositories."""
    from rbench_repro.acquisition import acquire_all

    acquire_all(
        cache_root=Path(args.cache_dir),
        output_manifest=Path(args.manifest),
        acquired_at=args.acquired_at,
    )
    return 0


def command_validate_inputs(args: argparse.Namespace) -> int:
    """Rehash every cached file and verify against the manifest."""
    from rbench_repro.acquisition import load_acquired

    manifest_path = Path(args.manifest)
    cache_dir = Path(args.cache_dir)
    sources = load_acquired(manifest_path, cache_dir)
    print(f"validated {len(sources)} sources", file=sys.stderr)
    return 0


def command_audit(args: argparse.Namespace) -> int:
    """Run a fully offline audit producing schema-valid evidence."""
    from rbench_repro.acquisition import load_acquired
    from rbench_repro.census import run_census
    from rbench_repro.leaderboard import (
        audit_leaderboard,
        compare_cohorts,
        derive_formula,
        derive_groups,
        infer_displayed_mean_formula,
    )
    from rbench_repro.source_audit import audit_failure_modes, trace_metrics
    from rbench_repro.census import read_manifested_bytes

    manifest_path = Path(args.manifest)
    cache_dir = Path(args.cache_dir)
    schema_path = Path(args.schema)
    output_path = Path(args.output)

    sources = load_acquired(manifest_path, cache_dir)

    census = run_census(sources)
    metrics = trace_metrics(sources)

    # Derive formula from leaderboard Space sources
    paper_source = sources["rbench-leaderboard-paper-era"]
    current_source = sources["rbench-leaderboard-current"]
    utils_source = read_manifested_bytes(paper_source, "utils.py")
    app_source = read_manifested_bytes(paper_source, "app.py")
    formula_provenance = "source-traced"
    try:
        formula = derive_formula(utils_source, app_source)
    except ValueError:
        paper_records = json.loads(
            read_manifested_bytes(paper_source, "leaderboard.json")
        )
        if (
            not isinstance(paper_records, list)
            or not paper_records
            or not isinstance(paper_records[0], dict)
        ):
            raise ValueError("invalid leaderboard records")
        formula = infer_displayed_mean_formula(
            paper_records,
            tuple(
                key
                for key in paper_records[0]
                if key not in {"model", "avg"}
            ),
        )
        formula_provenance = "artifact-inferred"
    groups = derive_groups(read_manifested_bytes(current_source, "app.py"))

    paper_lb = audit_leaderboard(
        paper_source, "paper-era", "leaderboard.json", formula, groups
    )
    current_lb = audit_leaderboard(
        current_source, "later", "leaderboard.json", formula, groups
    )
    comparison = compare_cohorts(paper_lb, current_lb)
    failure_modes = audit_failure_modes(sources)

    lock_path = Path(__file__).resolve().parents[2] / "uv.lock"
    lock_sha = sha256_bytes(lock_path.read_bytes()) if lock_path.is_file() else "unknown"

    inputs = AuditInputs(
        sources=tuple(s.manifest for s in sources.values()),
        census=census,
        metrics=metrics,
        formula=formula,
        leaderboards=(paper_lb, current_lb),
        comparison=comparison,
        failure_modes=failure_modes,
        category_evidence=census.category_sets,
        package_lock_sha256=lock_sha,
        formula_provenance=formula_provenance,
    )

    evidence = build_evidence(inputs, args.generated_at, args.tool_revision)
    validate_evidence(evidence, schema_path)
    _atomic_write(output_path, canonical_json(evidence))
    return 0


def command_validate(args: argparse.Namespace) -> int:
    """Validate a committed evidence bundle against the schema."""
    evidence_path = Path(args.evidence)
    schema_path = Path(args.schema)
    evidence = json.loads(evidence_path.read_bytes())
    validate_evidence(evidence, schema_path)
    print(f"valid: {evidence_path.name}", file=sys.stderr)
    return 0


def command_propose(args: argparse.Namespace) -> int:
    """Generate a non-authoritative worker proposal."""
    evidence_path = Path(args.evidence)
    output_path = Path(args.output)
    evidence_bytes = evidence_path.read_bytes()
    proposal = build_worker_proposal(
        evidence_bytes, args.source_commit, args.source_tree
    )
    _atomic_write(output_path, canonical_json(proposal))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rbench-repro")
    subparsers = parser.add_subparsers(dest="command")

    # acquire
    acquire = subparsers.add_parser("acquire")
    acquire.add_argument("--cache-dir", required=True)
    acquire.add_argument("--manifest", required=True)
    acquire.add_argument("--acquired-at", required=True)
    acquire.set_defaults(handler=command_acquire)

    # validate-inputs
    vi = subparsers.add_parser("validate-inputs")
    vi.add_argument("--manifest", required=True)
    vi.add_argument("--cache-dir", required=True)
    vi.set_defaults(handler=command_validate_inputs)

    # audit
    audit = subparsers.add_parser("audit")
    audit.add_argument("--manifest", required=True)
    audit.add_argument("--cache-dir", required=True)
    audit.add_argument("--schema", required=True)
    audit.add_argument("--output", required=True)
    audit.add_argument("--generated-at", required=True)
    audit.add_argument("--tool-revision", required=True)
    audit.set_defaults(handler=command_audit)

    # validate
    validate = subparsers.add_parser("validate")
    validate.add_argument("evidence")
    validate.add_argument("--schema", required=True)
    validate.set_defaults(handler=command_validate)

    # propose
    propose = subparsers.add_parser("propose")
    propose.add_argument("--evidence", required=True)
    propose.add_argument("--source-commit", required=True)
    propose.add_argument("--source-tree", required=True)
    propose.add_argument("--output", required=True)
    propose.set_defaults(handler=command_propose)

    return parser


def main(argv: list[str] | None = None) -> int:
    import jsonschema

    parser = build_parser()
    args = parser.parse_args(argv)
    if not hasattr(args, "handler"):
        parser.print_help()
        return 1
    try:
        return args.handler(args)
    except (ValueError, OSError, jsonschema.ValidationError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1


def _atomic_write(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as file:
            file.write(value)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


if __name__ == "__main__":
    sys.exit(main())
