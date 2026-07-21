"""Recompute benchmark and construction evidence from the NAPE release."""

import json
import statistics
from pathlib import Path

from icml_2026_repro.audit import (
    GITHUB_REVISION,
    REPOSITORY_ROOT,
    read_git_head,
    read_git_worktree_status,
)

PAPER_TRAJECTORIES = 52
PAPER_OPERATIONS = 11_907
PAPER_MINIMUM_SEQUENCE_LENGTH = 35
PAPER_MAXIMUM_SEQUENCE_LENGTH = 821
PAPER_MEAN_SEQUENCE_LENGTH = 228.98076923076923
PAPER_ROUNDED_MEAN_SEQUENCE_LENGTH = 229
PAPER_MEDIAN_SEQUENCE_LENGTH = 164
REQUIRED_RAW_FILENAMES = (
    "operations.txt",
    "predictable_state.json",
    "sheet_image.png",
    "spreadsheet.xlsx",
)
PIPELINE_SOURCES = {
    "symbolic_sequencing": "src/next_action_pred_eval/generation/sequencing/engine.py",
    "region_annotation": "src/next_action_pred_eval/generation/regions/analyzer.py",
    "llm_refinement": "src/next_action_pred_eval/generation/refinement/pipeline.py",
}


def read_trajectory(trajectory_path: Path) -> tuple[str, int]:
    """Validate one trajectory JSON file and return its identifier and length."""
    trajectory = json.loads(trajectory_path.read_text(encoding="utf-8"))
    if not isinstance(trajectory, dict):
        raise ValueError(f"{trajectory_path.name}: trajectory JSON must be an object")
    name = trajectory.get("name")
    if not isinstance(name, str):
        raise ValueError(f"{trajectory_path.name}: name must be a string")
    operations = trajectory.get("operations")
    if (
        not isinstance(operations, list)
        or not operations
        or not all(isinstance(operation, str) for operation in operations)
    ):
        raise ValueError(f"{trajectory_path.name}: operations must be a non-empty list of strings")
    return name, len(operations)


def read_trajectories(trajectory_paths: list[Path]) -> tuple[set[str], list[int]]:
    """Validate trajectory identities and return their identifiers and lengths."""
    trajectory_ids: set[str] = set()
    sequence_lengths: list[int] = []
    for trajectory_path in trajectory_paths:
        name, sequence_length = read_trajectory(trajectory_path)
        if name in trajectory_ids:
            raise ValueError(f"duplicate trajectory name: {name}")
        if name != trajectory_path.stem:
            raise ValueError(f"filename/name mismatch: {trajectory_path.name}")
        trajectory_ids.add(name)
        sequence_lengths.append(sequence_length)
    return trajectory_ids, sequence_lengths


def verify_nape_checkout(nape_path: Path) -> None:
    """Require the exact clean NAPE checkout used for benchmark evidence."""
    revision = read_git_head(nape_path)
    if revision != GITHUB_REVISION:
        raise ValueError(f"NAPE checkout is at {revision}, expected {GITHUB_REVISION}")
    status = read_git_worktree_status(nape_path)
    if status:
        raise ValueError(f"NAPE checkout is dirty: {', '.join(status.splitlines())}")


def audit_benchmark_release(nape_path: Path) -> dict[str, object]:
    """Recompute NAPE release statistics and audit construction artifacts."""
    verify_nape_checkout(nape_path)
    trajectory_paths = sorted((nape_path / "data" / "trajectories").glob("*.json"))
    if not trajectory_paths:
        raise ValueError("no trajectory JSON files found")

    trajectory_ids, sequence_lengths = read_trajectories(trajectory_paths)

    raw_path = nape_path / "data" / "raw"
    if not raw_path.is_dir():
        raise ValueError("missing data/raw directory")
    raw_ids = {path.name for path in raw_path.iterdir() if path.is_dir()}
    if trajectory_ids != raw_ids:
        raise ValueError("trajectory/raw ID sets differ")
    for trajectory_id in trajectory_ids:
        missing_files = [
            filename
            for filename in REQUIRED_RAW_FILENAMES
            if not (raw_path / trajectory_id / filename).is_file()
        ]
        if missing_files:
            raise ValueError(
                f"missing required raw files for {trajectory_id}: {', '.join(missing_files)}"
            )
    for source_path in PIPELINE_SOURCES.values():
        if not (nape_path / source_path).is_file():
            raise ValueError(f"missing pipeline source: {source_path}")

    mean_sequence_length = statistics.mean(sequence_lengths)
    observed = {
        "trajectories": len(sequence_lengths),
        "operations": sum(sequence_lengths),
        "minimum_sequence_length": min(sequence_lengths),
        "maximum_sequence_length": max(sequence_lengths),
        "mean_sequence_length": mean_sequence_length,
        "paper_rounded_mean": round(mean_sequence_length),
        "median_sequence_length": statistics.median(sequence_lengths),
    }
    paper_observed = {
        "trajectories": PAPER_TRAJECTORIES,
        "operations": PAPER_OPERATIONS,
        "minimum_sequence_length": PAPER_MINIMUM_SEQUENCE_LENGTH,
        "maximum_sequence_length": PAPER_MAXIMUM_SEQUENCE_LENGTH,
        "mean_sequence_length": PAPER_MEAN_SEQUENCE_LENGTH,
        "paper_rounded_mean": PAPER_ROUNDED_MEAN_SEQUENCE_LENGTH,
        "median_sequence_length": PAPER_MEDIAN_SEQUENCE_LENGTH,
    }
    if observed != paper_observed:
        raise ValueError("observed benchmark statistics do not match paper values")
    artifact_audit = {
        "matched_trajectory_ids": len(trajectory_ids),
        "required_files_per_trajectory": list(REQUIRED_RAW_FILENAMES),
    }
    return {
        "claim": (
            "NAPE contains 52 trajectories and 11,907 actions with sequence lengths 35-821 "
            "(mean 229, median 164), produced by symbolic sequencing, LLM refinement, and "
            "human annotation."
        ),
        "source_revision": GITHUB_REVISION,
        "input_paths": ["external/NAPE/data/trajectories", "external/NAPE/data/raw"],
        "counting_definition": (
            "One trajectory is one JSON file; one action is one string in its operations array."
        ),
        "observed": observed,
        "artifact_audit": artifact_audit,
        "construction_pipeline": {
            "symbolic_sequencing": {
                "source_path": PIPELINE_SOURCES["symbolic_sequencing"],
                "evidence_scope": "executable_source",
            },
            "region_annotation": {
                "source_path": PIPELINE_SOURCES["region_annotation"],
                "evidence_scope": "executable_source",
            },
            "llm_refinement": {
                "source_path": PIPELINE_SOURCES["llm_refinement"],
                "evidence_scope": "executable_source",
            },
            "human_annotation": {
                "source_path": None,
                "evidence_scope": "release_provenance_only",
            },
        },
        "evidence_scope": "released_artifact_recomputation_and_source_audit",
        "verdict": "reproduced",
    }


def build_benchmark_evidence() -> dict[str, object]:
    """Build benchmark evidence from the exact clean pinned NAPE checkout."""
    return audit_benchmark_release(REPOSITORY_ROOT / "external" / "NAPE")
