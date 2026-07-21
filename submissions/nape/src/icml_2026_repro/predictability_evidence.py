"""Recompute released predictability evidence from the NAPE release."""

import json
import math
import statistics
from pathlib import Path

from icml_2026_repro.audit import (
    GITHUB_REVISION,
    REPOSITORY_ROOT,
    read_git_head,
    read_git_worktree_status,
)


def verify_nape_checkout(nape_path: Path) -> None:
    """Require the exact clean NAPE checkout used for predictability evidence."""
    revision = read_git_head(nape_path)
    if revision != GITHUB_REVISION:
        raise ValueError(f"NAPE checkout is at {revision}, expected {GITHUB_REVISION}")
    status = read_git_worktree_status(nape_path)
    if status:
        raise ValueError(f"NAPE checkout is dirty: {', '.join(status.splitlines())}")


def read_predictable_state(trajectory_path: Path) -> dict[str, object]:
    """Read one released predictable-state object."""
    predictable_state_path = trajectory_path / "predictable_state.json"
    if not predictable_state_path.is_file():
        raise ValueError(f"{trajectory_path.name}: missing predictable_state.json")
    predictable_state = json.loads(predictable_state_path.read_text(encoding="utf-8"))
    if not isinstance(predictable_state, dict):
        raise ValueError(f"{trajectory_path.name}: JSON must be an object")
    return predictable_state


def read_trajectory_name(
    predictable_state: dict[str, object], trajectory_path: Path, trajectory_ids: set[str]
) -> None:
    """Validate and record a released trajectory identifier."""
    trajectory_name = predictable_state.get("trajectory_name")
    if not isinstance(trajectory_name, str):
        raise ValueError(f"{trajectory_path.name}: trajectory_name must be a string")
    if trajectory_name in trajectory_ids:
        raise ValueError(f"duplicate trajectory_name: {trajectory_name}")
    if trajectory_name != trajectory_path.name:
        raise ValueError(f"{trajectory_path.name}: trajectory_name does not match directory")
    trajectory_ids.add(trajectory_name)


def read_integer(
    predictable_state: dict[str, object], field_name: str, trajectory_path: Path
) -> int:
    """Read one non-boolean integer field from a released output."""
    value = predictable_state.get(field_name)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{trajectory_path.name}: {field_name} must be an integer")
    return value


def read_coverage_pct(predictable_state: dict[str, object], trajectory_path: Path) -> float:
    """Read one finite bounded coverage percentage from a released output."""
    coverage_pct = predictable_state.get("coverage_pct")
    if (
        isinstance(coverage_pct, bool)
        or not isinstance(coverage_pct, int | float)
        or not math.isfinite(coverage_pct)
    ):
        raise ValueError(f"{trajectory_path.name}: coverage_pct must be a finite number")
    if not 0 <= coverage_pct <= 100:
        raise ValueError(f"{trajectory_path.name}: coverage_pct must be within [0, 100]")
    return float(coverage_pct)


def count_predictable_properties(
    predictable_state: dict[str, object], trajectory_path: Path
) -> int:
    """Validate the released sheet/cell/property structure and count entries."""
    predictable_properties = predictable_state.get("predictable_properties")
    if not isinstance(predictable_properties, dict):
        raise ValueError(f"{trajectory_path.name}: predictable_properties must be an object")
    property_entry_count = 0
    for sheet_name, cells in predictable_properties.items():
        if not isinstance(sheet_name, str) or not isinstance(cells, dict):
            raise ValueError(
                f"{trajectory_path.name}: predictable_properties sheet must be an object"
            )
        for cell_address, properties in cells.items():
            if not isinstance(cell_address, str) or not isinstance(properties, list):
                raise ValueError(
                    f"{trajectory_path.name}: predictable_properties cell must be a list of strings"
                )
            if not all(isinstance(property_name, str) for property_name in properties):
                raise ValueError(
                    f"{trajectory_path.name}: predictable_properties cell must be a list of strings"
                )
            property_entry_count += len(properties)
    return property_entry_count


def read_predictable_row(trajectory_path: Path, trajectory_ids: set[str]) -> tuple[int, int, float]:
    """Validate one released output and return its aggregate values."""
    predictable_state = read_predictable_state(trajectory_path)
    read_trajectory_name(predictable_state, trajectory_path, trajectory_ids)
    predictable_count = read_integer(predictable_state, "predictable_count", trajectory_path)
    final_state_size = read_integer(predictable_state, "final_state_size", trajectory_path)
    if final_state_size <= 0:
        raise ValueError(f"{trajectory_path.name}: final_state_size must be positive")
    if not 0 <= predictable_count <= final_state_size:
        raise ValueError(
            f"{trajectory_path.name}: predictable_count must be within final_state_size"
        )
    coverage_pct = read_coverage_pct(predictable_state, trajectory_path)
    if count_predictable_properties(predictable_state, trajectory_path) != predictable_count:
        raise ValueError(
            f"{trajectory_path.name}: predictable_count does not match predictable_properties"
        )
    if not math.isclose(coverage_pct, 100 * predictable_count / final_state_size, abs_tol=1e-2):
        raise ValueError(f"{trajectory_path.name}: coverage_pct does not match counts")
    return predictable_count, final_state_size, coverage_pct


def aggregate_predictability(raw_path: Path) -> dict[str, object]:
    """Aggregate validated released predictable-state outputs."""
    verify_nape_checkout(raw_path.parents[1])
    trajectory_paths = sorted(path for path in raw_path.iterdir() if path.is_dir())
    rows: list[tuple[int, int, float]] = []
    trajectory_ids: set[str] = set()
    for trajectory_path in trajectory_paths:
        rows.append(read_predictable_row(trajectory_path, trajectory_ids))

    if not rows:
        raise ValueError("no predictable-state outputs found")
    total_predictable = sum(row[0] for row in rows)
    total_final = sum(row[1] for row in rows)
    coverages = [row[2] for row in rows]
    return {
        "claim": (
            "Approximately 68% of final spreadsheet properties are empirically predictable "
            "by the released frontier-model oracle pipeline."
        ),
        "source_revision": GITHUB_REVISION,
        "input_path": "external/NAPE/data/raw/*/predictable_state.json",
        "counting_definition": (
            "Weighted coverage is 100 times the sum of predictable_count divided by the sum "
            "of final_state_size."
        ),
        "observed": {
            "trajectories": len(rows),
            "predictable_properties": total_predictable,
            "final_state_properties": total_final,
            "weighted_coverage_pct": 100 * total_predictable / total_final,
            "mean_coverage_pct": statistics.mean(coverages),
            "median_coverage_pct": statistics.median(coverages),
            "trajectories_above_50_pct": sum(value > 50 for value in coverages),
        },
        "evidence_scope": "released_oracle_output_recomputation",
        "limitation": "The original paid frontier-model oracle calls were not rerun.",
        "verdict": "reproduced_from_released_outputs",
    }


def build_predictability_evidence() -> dict[str, object]:
    """Build predictability evidence from the exact clean pinned NAPE checkout."""
    nape_path = REPOSITORY_ROOT / "external" / "NAPE"
    verify_nape_checkout(nape_path)
    return aggregate_predictability(nape_path / "data" / "raw")
