"""Command-line entry point for TimeRewarder trust-boundary operations."""

import argparse
import hashlib
import json
import logging
import os
import sys
import tempfile
from collections.abc import Sequence
from pathlib import Path

from timerewarder_repro.acquisition import acquire_inert_sources
from timerewarder_repro.checkpoint import (
    checkpoint_entry,
    load_checkpoint_registry,
)
from timerewarder_repro.conversion import (
    ConversionRejected,
    approve_conversion,
    convert_checkpoint,
)
from timerewarder_repro.evaluation import evaluate_representative
from timerewarder_repro.evidence import build_evidence_bundle
from timerewarder_repro.fixture import run_fixture


def convert_registered_checkpoint(
    *,
    task: str,
    registry_path: Path,
    cache_dir: Path,
    output_dir: Path,
    converter: str,
) -> dict[str, object]:
    """Convert one registry-selected checkpoint and emit no approval."""
    registry = load_checkpoint_registry(registry_path)
    entry = checkpoint_entry(registry, task)
    project_root = registry_path.resolve().parent.parent
    checkpoint = cache_dir / str(entry["file"])
    rejection_path = (
        project_root
        / "artifacts"
        / "conversion-rejections"
        / f"{Path(str(entry['file'])).stem}.json"
    )
    try:
        _verify_checkpoint(checkpoint, entry)
        expected_approval = project_root / str(entry["approval"])
        if expected_approval.exists():
            raise ConversionRejected("approval_exists")
        schema_path = project_root / str(registry["schema"]["path"])
        task_output = output_dir / Path(str(entry["file"])).stem
        task_output.mkdir(parents=True, exist_ok=True)
        request = {
            "task": task,
            "checkpoint_file": entry["file"],
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": entry["lfs_sha256"],
            "checkpoint_bytes": entry["size_bytes"],
            "model_repository": entry["repository"],
            "model_revision": entry["model_revision"],
            "schema": str(schema_path),
            "schema_sha256": entry["schema_sha256"],
            "schema_bytes": schema_path.stat().st_size,
            "output_dir": str(task_output),
            "runtime": str(project_root / "conversion" / ".venv"),
            "package_root": str(project_root / "src"),
            "python_root": sys.base_prefix,
            "converter": converter,
        }
        request_path = _temporary_json(request, task_output)
        try:
            receipt = convert_checkpoint(request_path)
        finally:
            request_path.unlink(missing_ok=True)
        receipt_path = project_root / str(entry["receipt"])
        _write_json_atomic(receipt, receipt_path)
        rejection_path.unlink(missing_ok=True)
        return receipt
    except ConversionRejected as error:
        rejection = {
            "format": "timerewarder-conversion-rejection-v1",
            "task": task,
            "checkpoint": {
                "file": entry["file"],
                "repository": entry["repository"],
                "model_revision": entry["model_revision"],
                "lfs_sha256": entry["lfs_sha256"],
                "size_bytes": entry["size_bytes"],
            },
            "failure_category": error.gate,
        }
        _write_json_atomic(rejection, rejection_path)
        raise


def review_registered_conversion(
    *,
    task: str,
    registry_path: Path,
    receipt_path: Path,
    output_path: Path,
    reviewer: str,
    approval_path: Path,
) -> dict[str, object]:
    """Validate and atomically record one independently reviewed output."""
    registry = load_checkpoint_registry(registry_path)
    entry = checkpoint_entry(registry, task)
    project_root = registry_path.resolve().parent.parent
    expected_receipt = project_root / str(entry["receipt"])
    expected_approval = project_root / str(entry["approval"])
    if receipt_path.resolve() != expected_receipt.resolve():
        raise ValueError("receipt path does not match registry")
    if approval_path.resolve() != expected_approval.resolve():
        raise ValueError("approval path does not match registry")
    if approval_path.exists():
        raise ValueError("approval record already exists")
    receipt = _read_json(receipt_path)
    if (
        receipt.get("task") != task
        or receipt.get("checkpoint_file") != entry["file"]
        or receipt.get("checkpoint_sha256") != entry["lfs_sha256"]
        or receipt.get("checkpoint_bytes") != entry["size_bytes"]
        or receipt.get("model_revision") != entry["model_revision"]
        or receipt.get("schema_sha256") != entry["schema_sha256"]
    ):
        raise ValueError("receipt identity does not match registry")
    approval = approve_conversion(receipt_path, reviewer, output_path)
    _write_json_atomic(approval, approval_path)
    return approval


def main(argv: Sequence[str] | None = None) -> None:
    """Run one explicitly selected trust-boundary operation."""
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    acquire = commands.add_parser("acquire", help="acquire verified inert source bytes")
    acquire.add_argument("--manifest", type=Path, required=True)
    acquire.add_argument("--output", type=Path, required=True)
    acquire.add_argument("--receipt", type=Path, required=True)

    convert = commands.add_parser("convert", help="convert one pinned checkpoint")
    convert.add_argument("--task", required=True)
    convert.add_argument("--registry", type=Path, required=True)
    convert.add_argument("--cache-dir", type=Path, required=True)
    convert.add_argument("--output-dir", type=Path, required=True)
    convert.add_argument("--converter", required=True)

    review = commands.add_parser(
        "review-conversion", help="record an independent conversion approval"
    )
    review.add_argument("--task", required=True)
    review.add_argument("--registry", type=Path, required=True)
    review.add_argument("--receipt", type=Path, required=True)
    review.add_argument("--output", type=Path, required=True)
    review.add_argument("--reviewer", required=True)
    review.add_argument("--approval", type=Path, required=True)

    representative = commands.add_parser(
        "representative", help="run the fixed released-video protocol"
    )
    representative.add_argument("--registry", type=Path, required=True)
    representative.add_argument("--dataset-manifest", type=Path, required=True)
    representative.add_argument("--schema", type=Path, required=True)
    representative.add_argument("--cache-dir", type=Path, required=True)
    representative.add_argument("--output", type=Path, required=True)

    fixture = commands.add_parser("fixture", help="rerun the diagnostic fixture")
    fixture.add_argument("--output", type=Path, required=True)

    evidence = commands.add_parser(
        "build-evidence", help="recompute the canonical six-claim bundle"
    )
    evidence.add_argument("--manifest", type=Path, required=True)
    evidence.add_argument("--acquisition", type=Path, required=True)
    evidence.add_argument("--registry", type=Path, required=True)
    evidence.add_argument("--source-root", type=Path, required=True)
    evidence.add_argument("--representative", type=Path, required=True)
    evidence.add_argument("--output", type=Path, required=True)

    arguments = parser.parse_args(argv)
    if arguments.command == "acquire":
        result = acquire_inert_sources(
            arguments.manifest, arguments.output, arguments.receipt
        )
    elif arguments.command == "convert":
        result = convert_registered_checkpoint(
            task=arguments.task,
            registry_path=arguments.registry,
            cache_dir=arguments.cache_dir,
            output_dir=arguments.output_dir,
            converter=arguments.converter,
        )
    elif arguments.command == "review-conversion":
        result = review_registered_conversion(
            task=arguments.task,
            registry_path=arguments.registry,
            receipt_path=arguments.receipt,
            output_path=arguments.output,
            reviewer=arguments.reviewer,
            approval_path=arguments.approval,
        )
    elif arguments.command == "representative":
        result = evaluate_representative(
            arguments.registry,
            arguments.dataset_manifest,
            arguments.schema,
            arguments.cache_dir,
        )
        _write_canonical_json_atomic(result, arguments.output)
    elif arguments.command == "fixture":
        result = run_fixture()
        _write_canonical_json_atomic(result, arguments.output)
    else:
        result = build_evidence_bundle(
            arguments.manifest,
            arguments.acquisition,
            arguments.registry,
            arguments.source_root,
            arguments.representative,
        )
        _write_canonical_json_atomic(result, arguments.output)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logging.info(json.dumps(result, sort_keys=True))


def _verify_checkpoint(path: Path, entry: dict[str, object]) -> None:
    if (
        not path.is_file()
        or path.is_symlink()
        or path.stat().st_size != entry["size_bytes"]
        or _sha256_file(path) != entry["lfs_sha256"]
    ):
        raise ConversionRejected("input_identity", "checkpoint")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _temporary_json(value: object, directory: Path) -> Path:
    descriptor, name = tempfile.mkstemp(
        prefix=".conversion-request-", suffix=".json", dir=directory
    )
    path = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, sort_keys=True, separators=(",", ":"))
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise
    return path


def _write_json_atomic(value: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _write_canonical_json_atomic(value: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(
                value,
                handle,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _read_json(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("receipt is unreadable") from error
    if not isinstance(value, dict):
        raise ValueError("receipt schema")
    return value
