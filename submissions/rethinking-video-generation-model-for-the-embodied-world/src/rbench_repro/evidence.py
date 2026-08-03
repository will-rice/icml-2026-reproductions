"""Closed evidence renderer for the RBench reproduction."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import jsonschema

from rbench_repro.acquisition import SourceManifest
from rbench_repro.census import CensusResult
from rbench_repro.leaderboard import (
    CohortComparison,
    Formula,
    LeaderboardResult,
)
from rbench_repro.model import CLAIMS, canonical_json, sha256_bytes
from rbench_repro.source_audit import FailureModeResult, MetricTrace

PAPER_ID = "p5QSlnwume"
ATTEMPT_ID = "8c21f2dc-a357-422e-9c1b-79a4d417e3dc"


@dataclass(slots=True)
class AuditInputs:
    sources: tuple[SourceManifest, ...]
    census: CensusResult | None
    metrics: tuple[MetricTrace, ...]
    formula: Formula | None
    leaderboards: tuple[LeaderboardResult, ...]
    comparison: CohortComparison | None
    failure_modes: tuple[FailureModeResult, ...]
    category_evidence: dict[str, tuple[str, ...]]
    package_lock_sha256: str
    formula_provenance: str = "source-traced"


def build_evidence(
    inputs: AuditInputs, generated_at: str, tool_revision: str
) -> dict[str, object]:
    claims = (
        _claim_1_status(inputs.census, inputs.metrics, inputs.leaderboards),
        _claim_2_status(
            _cohort(inputs.leaderboards, "paper-era"),
            _cohort(inputs.leaderboards, "later"),
            inputs.comparison,
            inputs.category_evidence,
        ),
        _claim_3_status(inputs.failure_modes),
    )
    return _render_bundle(inputs, claims, generated_at, tool_revision)


def validate_evidence(value: object, schema_path: Path) -> None:
    import json

    schema = json.loads(schema_path.read_text())
    jsonschema.validate(instance=value, schema=schema)


def resolve_json_pointer(value: object, pointer: str) -> object:
    parts = pointer.strip("/").split("/")
    current = value
    for part in parts:
        if isinstance(current, dict):
            current = current[part]
        elif isinstance(current, list):
            current = current[int(part)]
        else:
            raise KeyError(f"cannot resolve pointer segment: {part}")
    return current


# ── Claim status logic ───────────────────────────────────────────────


def _claim_1_status(
    census: CensusResult | None,
    metrics: tuple[MetricTrace, ...],
    leaderboards: tuple[LeaderboardResult, ...],
) -> dict[str, object]:
    """Claim 1: five-task/four-embodiment census with sub-metrics."""
    observations: list[dict[str, object]] = []
    limitations: list[str] = []
    status = "verified"

    if census is not None:
        task_count = sum(
            1 for cat in census.categories if cat["partition"] == "task"
        )
        emb_count = sum(
            1 for cat in census.categories if cat["partition"] == "embodiment"
        )
        observations.append(
            {
                "summary": f"Census: {task_count} task categories, {emb_count} embodiment categories, {census.total_records} total prompts",
                "detail": census.to_dict(),
            }
        )
    else:
        status = "partial"
        limitations.append("census unavailable")

    if metrics:
        task_metrics = [m for m in metrics if m.partition == "task"]
        emb_metrics = [m for m in metrics if m.partition == "embodiment"]
        observations.append(
            {
                "summary": f"Source-traced {len(task_metrics)} task metrics and {len(emb_metrics)} embodiment metrics",
                "detail": [m.to_dict() for m in metrics],
            }
        )
    else:
        if status == "verified":
            status = "partial"
        limitations.append("no source-traced metrics available")

    if leaderboards:
        for lb in leaderboards:
            observations.append(
                {
                    "summary": f"Leaderboard ({lb.cohort}): {lb.valid_count} valid models",
                    "detail": lb.to_dict(),
                }
            )

    limitations.append(
        "video generation was not rerun; metric values are structural not empirical"
    )

    return {
        "claim": CLAIMS[0],
        "status": status,
        "observations": observations,
        "limitations": limitations,
    }


def _claim_2_status(
    paper: LeaderboardResult | None,
    later: LeaderboardResult | None,
    comparison: CohortComparison | None,
    category_evidence: dict[str, tuple[str, ...]],
) -> dict[str, object]:
    """Claim 2: 25-model leaderboard with cohort distinction."""
    observations: list[dict[str, object]] = []
    limitations: list[str] = []
    status = "verified"

    if paper is not None:
        observations.append(
            {
                "summary": f"Paper-era cohort: {paper.unique_exact_count} unique models",
                "detail": {
                    "ordered_names": list(paper.ordered_names),
                    "discrepancies": [dict(d) for d in paper.discrepancies],
                },
            }
        )
    if later is not None:
        observations.append(
            {
                "summary": f"Later cohort: {later.unique_exact_count} unique models",
                "detail": {
                    "ordered_names": list(later.ordered_names),
                    "discrepancies": [dict(d) for d in later.discrepancies],
                },
            }
        )
    if comparison is not None:
        observations.append(
            {
                "summary": f"Cohort comparison: {len(comparison.added_models)} added, {len(comparison.removed_models)} removed",
                "detail": comparison.to_dict(),
            }
        )

    if category_evidence:
        observations.append(
            {
                "summary": f"Cross-source category alignment from {len(category_evidence)} category sets",
                "detail": {
                    name: list(values)
                    for name, values in sorted(category_evidence.items())
                },
            }
        )
    else:
        if status == "verified":
            status = "partial"
        limitations.append("cross-source category evidence unavailable")

    limitations.append(
        "human correlation was not reproduced; model scores are from artifact audit only"
    )

    return {
        "claim": CLAIMS[1],
        "status": status,
        "observations": observations,
        "limitations": limitations,
    }


def _claim_3_status(
    failure_modes: tuple[FailureModeResult, ...],
) -> dict[str, object]:
    """Claim 3: failure-mode capture (partial — no real-video eval)."""
    observations: list[dict[str, object]] = []
    limitations: list[str] = [
        "real-video evaluation was not rerun; failure modes are source-traced only"
    ]

    if failure_modes:
        operationalized = [
            fm for fm in failure_modes if fm.status == "operationalized"
        ]
        declared = [fm for fm in failure_modes if fm.status == "declared_only"]
        missing = [fm for fm in failure_modes if fm.status == "missing"]
        observations.append(
            {
                "summary": (
                    f"{len(operationalized)} operationalized, "
                    f"{len(declared)} declared-only, {len(missing)} missing "
                    "failure modes"
                ),
                "detail": [fm.to_dict() for fm in failure_modes],
            }
        )
        if operationalized or declared:
            status = "partial"
        else:
            status = "inconclusive"
            limitations.append("no named failure mode was found in pinned artifacts")
    else:
        status = "inconclusive"
        limitations.append("no failure mode evidence available")

    return {
        "claim": CLAIMS[2],
        "status": status,
        "observations": observations,
        "limitations": limitations,
    }


# ── Internal rendering ───────────────────────────────────────────────


def _cohort(
    leaderboards: tuple[LeaderboardResult, ...], cohort: str
) -> LeaderboardResult | None:
    return next((lb for lb in leaderboards if lb.cohort == cohort), None)


def _render_bundle(
    inputs: AuditInputs,
    claims: tuple[dict[str, object], ...],
    generated_at: str,
    tool_revision: str,
) -> dict[str, object]:
    sources = [
        {
            "label": source.label,
            "repo_id": source.repo_id,
            "requested_revision": source.requested_revision,
            "resolved_revision": source.resolved_revision,
            "license_id": source.license_id,
            "redistributable": source.redistributable,
        }
        for source in inputs.sources
    ]

    unavailable = [
        "Video generation model inference was not rerun",
        "Human correlation study was not reproduced",
        "Semantic quality metric validation requires GPU inference",
    ]
    if inputs.formula_provenance == "artifact-inferred":
        unavailable.append(
            "Leaderboard aggregation formula was not source-traced; the mean "
            "rule is inferred only as an internal consistency check over "
            "rounded paper-era artifact fields"
        )

    # Artifact index with JSON pointers and content hashes.
    bundle: dict[str, object] = {
        "schema_version": 1,
        "paper_id": PAPER_ID,
        "attempt_id": ATTEMPT_ID,
        "generated_at": generated_at,
        "tool_revision": tool_revision,
        "sources": sources,
        "claims": list(claims),
        "environment": {"package_lock_sha256": inputs.package_lock_sha256},
        "unavailable": unavailable,
        "contradictions": [],
    }

    # Add detailed audit data when available.
    if inputs.census is not None:
        bundle["census"] = inputs.census.to_dict()
    if inputs.metrics:
        bundle["metrics"] = [m.to_dict() for m in inputs.metrics]
    if inputs.formula is not None:
        bundle["formula"] = {
            "columns": list(inputs.formula.columns),
            "operation": inputs.formula.operation,
            "source_precision": inputs.formula.source_precision,
            "output_precision": inputs.formula.output_precision,
            "rounding": inputs.formula.rounding,
            "absolute_tolerance": inputs.formula.absolute_tolerance,
        }
    if inputs.leaderboards:
        bundle["leaderboards"] = [lb.to_dict() for lb in inputs.leaderboards]
    if inputs.comparison is not None:
        bundle["comparison"] = inputs.comparison.to_dict()
    if inputs.failure_modes:
        bundle["failure_modes"] = [fm.to_dict() for fm in inputs.failure_modes]

    # Build artifact index from the data sections.
    artifacts = []
    artifact_sections = (
        ("census", "/census"),
        ("metrics", "/metrics"),
        ("formula", "/formula"),
        ("leaderboards", "/leaderboards"),
        ("comparison", "/comparison"),
        ("failure_modes", "/failure_modes"),
    )
    for label, pointer in artifact_sections:
        if label in bundle:
            pointed = resolve_json_pointer(bundle, pointer)
            artifacts.append(
                {
                    "label": label,
                    "json_pointer": pointer,
                    "sha256": sha256_bytes(canonical_json(pointed)),
                }
            )

    bundle["artifacts"] = artifacts

    return bundle
