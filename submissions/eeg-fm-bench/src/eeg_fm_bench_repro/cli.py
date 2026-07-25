"""Generate the deterministic EEG-FM-Bench evidence bundle."""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import platform
import tempfile
from pathlib import Path
from typing import Any

import numpy
import torch

from .census import run_census_audit
from .harness_audit import run_harness_audit
from .preproc_audit import run_preproc_audit
from .upstream import PROVENANCE_PATH, ensure_paper_pdf, ensure_repo_snapshot


def _claim(record: dict[str, Any], thresholds: dict[str, Any]) -> dict[str, Any]:
    status = record.get("status")
    if status is None:
        computed = record["computed"]
        context = record["paper_context"]
        status = (
            "verified"
            if computed["dataset_count"] == context["dataset_count"]
            and computed["paradigm_count"] == context["paradigm_count"]
            else "partial"
        )
    return {
        "claim_id": record["claim_id"],
        "kind": record["kind"],
        "status": status,
        "observations": record,
        "thresholds": thresholds,
    }


def build_bundle(cache_dir: Path) -> dict[str, Any]:
    """Acquire pinned inputs and compute all three selected claim records."""

    snapshot = ensure_repo_snapshot(cache_dir)
    ensure_paper_pdf(cache_dir)
    census = run_census_audit(snapshot)
    preprocessing = run_preproc_audit(snapshot)
    harness = run_harness_audit(snapshot)
    return {
        "schema_version": 1,
        "scope": "released_artifact_audit_not_leaderboard_reproduction",
        "claims": [
            _claim(census, {"dataset_count": 14, "paradigm_count": 10}),
            _claim(
                preprocessing,
                {
                    "datasets_minimum": 2,
                    "repeat_identical": True,
                    "target_sfreq": 256.0,
                },
            ),
            _claim(
                harness,
                {
                    "strategies": 3,
                    "finite_cpu_steps": 3,
                    "all_upstream_contract_checks": True,
                },
            ),
        ],
        "unavailable_claims": [
            {
                "claim": "gpu-foundation-model leaderboard performance",
                "status": "unavailable",
                "reason": "Requires GPU foundation-model runs and gated raw EEG datasets.",
            },
            {
                "claim": "representation and qualitative analyses",
                "status": "unavailable",
                "reason": "Requires checkpoints and raw datasets outside this CPU artifact audit.",
            },
        ],
        "provenance": json.loads(PROVENANCE_PATH.read_text(encoding="utf-8")),
        "environment": {
            "python": platform.python_version(),
            "numpy": numpy.__version__,
            "torch": torch.__version__,
        },
    }


def _json_bytes(bundle: dict[str, Any]) -> bytes:
    return (json.dumps(bundle, indent=2, sort_keys=True) + "\n").encode()


def _csv_bytes(bundle: dict[str, Any]) -> bytes:
    output = io.StringIO(newline="")
    writer = csv.DictWriter(
        output,
        fieldnames=["claim_id", "kind", "status", "observations", "thresholds"],
        lineterminator="\n",
    )
    writer.writeheader()
    for claim in bundle["claims"]:
        writer.writerow(
            {
                "claim_id": claim["claim_id"],
                "kind": claim["kind"],
                "status": claim["status"],
                "observations": json.dumps(
                    claim["observations"], sort_keys=True, separators=(",", ":")
                ),
                "thresholds": json.dumps(
                    claim["thresholds"], sort_keys=True, separators=(",", ":")
                ),
            }
        )
    return output.getvalue().encode()


def _atomic_write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def write_bundle(cache_dir: Path, output_dir: Path) -> None:
    bundle = build_bundle(cache_dir)
    _atomic_write(output_dir / "results.json", _json_bytes(bundle))
    _atomic_write(output_dir / "measurements.csv", _csv_bytes(bundle))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate deterministic EEG-FM-Bench artifact-audit evidence."
    )
    parser.add_argument("--cache-dir", type=Path, default=Path(".cache/upstream"))
    parser.add_argument("--output-dir", type=Path, default=Path("evidence"))
    return parser


def main() -> None:
    arguments = _parser().parse_args()
    write_bundle(arguments.cache_dir, arguments.output_dir)


if __name__ == "__main__":
    main()
