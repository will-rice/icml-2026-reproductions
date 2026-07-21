# Claim 1: Benchmark and construction


---
<!-- trackio-cell
{"type": "code", "id": "cell_20fd77758075", "created_at": "2026-07-21T15:59:42+00:00", "title": "Recompute all benchmark statistics", "command": ["uv", "run", "pytest", "tests/test_benchmark_evidence.py", "-q"], "exit_code": 0, "duration_s": 0.66}
-->
````bash
$ uv run pytest tests/test_benchmark_evidence.py -q
````

exit 0 · 0.7s


````python title=test_benchmark_evidence.py
import json
from pathlib import Path
from typing import cast

import pytest

import icml_2026_repro
import icml_2026_repro.benchmark_evidence as benchmark_evidence
from icml_2026_repro.benchmark_evidence import audit_benchmark_release, build_benchmark_evidence

NAPE_PATH = Path(__file__).resolve().parents[1] / "external" / "NAPE"
REQUIRED_RAW_FILES = [
    "operations.txt",
    "predictable_state.json",
    "sheet_image.png",
    "spreadsheet.xlsx",
]


@pytest.fixture
def pinned_clean_nape_checkout(monkeypatch):
    monkeypatch.setattr(
        benchmark_evidence,
        "read_git_head",
        lambda _: benchmark_evidence.GITHUB_REVISION,
    )
    monkeypatch.setattr(benchmark_evidence, "read_git_worktree_status", lambda _: "")


def test_audit_benchmark_release_reproduces_paper_statistics():
    evidence = audit_benchmark_release(NAPE_PATH)
    artifact_audit = cast(dict[str, object], evidence["artifact_audit"])
    construction_pipeline = cast(dict[str, dict[str, object]], evidence["construction_pipeline"])

    assert evidence["observed"] == {
        "trajectories": 52,
        "operations": 11907,
        "minimum_sequence_length": 35,
        "maximum_sequence_length": 821,
        "mean_sequence_length": pytest.approx(228.98076923076923),
        "paper_rounded_mean": 229,
        "median_sequence_length": 164,
    }
    assert artifact_audit["matched_trajectory_ids"] == 52
    assert artifact_audit["required_files_per_trajectory"] == REQUIRED_RAW_FILES
    assert construction_pipeline["human_annotation"]["evidence_scope"] == (
        "release_provenance_only"
    )
    assert evidence["source_revision"] == benchmark_evidence.GITHUB_REVISION
    assert evidence["evidence_scope"] == "released_artifact_recomputation_and_source_audit"
    assert evidence["verdict"] == "reproduced"


@pytest.fixture
def minimal_nape_release(tmp_path, pinned_clean_nape_checkout):
    del pinned_clean_nape_checkout
    nape_path = tmp_path / "NAPE"
    trajectories_path = nape_path / "data" / "trajectories"
    raw_path = nape_path / "data" / "raw"
    trajectories_path.mkdir(parents=True)
    for name, operations in {"first": ["one", "two"], "second": ["three"]}.items():
        (trajectories_path / f"{name}.json").write_text(
            json.dumps({"name": name, "operations": operations}), encoding="utf-8"
        )
        trajectory_raw_path = raw_path / name
        trajectory_raw_path.mkdir(parents=True)
        for required_file in REQUIRED_RAW_FILES:
            (trajectory_raw_path / required_file).write_text("fixture\n", encoding="utf-8")
    for source_path in benchmark_evidence.PIPELINE_SOURCES.values():
        path = nape_path / source_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("fixture\n", encoding="utf-8")
    return nape_path


def test_audit_benchmark_release_rejects_duplicate_trajectory_names(minimal_nape_release):
    (minimal_nape_release / "data" / "trajectories" / "third.json").write_text(
        json.dumps({"name": "first", "operations": ["four"]}), encoding="utf-8"
    )

    with pytest.raises(ValueError, match="duplicate trajectory name"):
        audit_benchmark_release(minimal_nape_release)


def test_audit_benchmark_release_rejects_filename_name_mismatch(minimal_nape_release):
    trajectory_path = minimal_nape_release / "data" / "trajectories" / "first.json"
    trajectory_path.write_text(
        json.dumps({"name": "other", "operations": ["one"]}), encoding="utf-8"
    )

    with pytest.raises(ValueError, match="filename/name mismatch"):
        audit_benchmark_release(minimal_nape_release)


def test_audit_benchmark_release_rejects_unequal_trajectory_and_raw_ids(minimal_nape_release):
    (minimal_nape_release / "data" / "raw" / "second").rename(
        minimal_nape_release / "data" / "raw" / "other"
    )

    with pytest.raises(ValueError, match="trajectory/raw ID sets differ"):
        audit_benchmark_release(minimal_nape_release)


def test_audit_benchmark_release_rejects_missing_required_raw_files(minimal_nape_release):
    (minimal_nape_release / "data" / "raw" / "first" / "sheet_image.png").unlink()

    with pytest.raises(ValueError, match="missing required raw files"):
        audit_benchmark_release(minimal_nape_release)


def test_audit_benchmark_release_rejects_non_string_operations(minimal_nape_release):
    trajectory_path = minimal_nape_release / "data" / "trajectories" / "first.json"
    trajectory_path.write_text(
        json.dumps({"name": "first", "operations": ["one", 2]}), encoding="utf-8"
    )

    with pytest.raises(ValueError, match="operations must be a non-empty list of strings"):
        audit_benchmark_release(minimal_nape_release)


def test_audit_benchmark_release_rejects_empty_release(minimal_nape_release):
    for trajectory_path in (minimal_nape_release / "data" / "trajectories").glob("*.json"):
        trajectory_path.unlink()

    with pytest.raises(ValueError, match="no trajectory JSON files"):
        audit_benchmark_release(minimal_nape_release)


def test_audit_benchmark_release_rejects_statistics_inconsistent_with_paper(
    minimal_nape_release,
):
    with pytest.raises(ValueError, match="statistics do not match paper values"):
        audit_benchmark_release(minimal_nape_release)


@pytest.mark.parametrize("trajectory", [[], {"name": [], "operations": ["one"]}])
def test_audit_benchmark_release_rejects_invalid_trajectory_json_roots(
    minimal_nape_release, trajectory
):
    trajectory_path = minimal_nape_release / "data" / "trajectories" / "first.json"
    trajectory_path.write_text(json.dumps(trajectory), encoding="utf-8")

    with pytest.raises(ValueError, match="trajectory JSON must be an object|name must be a string"):
        audit_benchmark_release(minimal_nape_release)


def test_audit_benchmark_release_rejects_missing_raw_directory(minimal_nape_release):
    raw_path = minimal_nape_release / "data" / "raw"
    raw_path.rename(minimal_nape_release / "data" / "raw-missing")

    with pytest.raises(ValueError, match="missing data/raw directory"):
        audit_benchmark_release(minimal_nape_release)


def test_audit_benchmark_release_rejects_mismatched_revision(monkeypatch):
    monkeypatch.setattr(benchmark_evidence, "read_git_head", lambda _: "wrong-revision")

    with pytest.raises(ValueError, match="wrong-revision.*expected"):
        audit_benchmark_release(NAPE_PATH)


def test_audit_benchmark_release_rejects_dirty_checkout(monkeypatch):
    monkeypatch.setattr(benchmark_evidence, "read_git_worktree_status", lambda _: " M data/raw")

    with pytest.raises(ValueError, match="checkout is dirty"):
        audit_benchmark_release(NAPE_PATH)


def test_build_benchmark_evidence_rejects_mismatched_revision(monkeypatch):
    monkeypatch.setattr(benchmark_evidence, "read_git_head", lambda _: "wrong-revision")

    with pytest.raises(ValueError, match="wrong-revision.*expected"):
        build_benchmark_evidence()


def test_build_benchmark_evidence_rejects_dirty_checkout(monkeypatch):
    monkeypatch.setattr(
        benchmark_evidence,
        "read_git_head",
        lambda _: benchmark_evidence.GITHUB_REVISION,
    )
    monkeypatch.setattr(benchmark_evidence, "read_git_worktree_status", lambda _: " M data/raw")

    with pytest.raises(ValueError, match="checkout is dirty"):
        build_benchmark_evidence()


def test_build_benchmark_evidence_records_verified_pinned_revision():
    evidence = build_benchmark_evidence()

    assert evidence["source_revision"] == benchmark_evidence.GITHUB_REVISION
    assert evidence["evidence_scope"] == "released_artifact_recomputation_and_source_audit"


def test_package_exports_benchmark_builder_as_canonical_claim_1_interface():
    assert icml_2026_repro.build_benchmark_evidence is build_benchmark_evidence
    assert not hasattr(icml_2026_repro, "build_claim_1_evidence")

````


````output
.................                                                        [100%]
17 passed in 0.66s

````


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_03aa9c4eb09a", "created_at": "2026-07-21T16:05:13+00:00", "title": "Outcome and evidence scope"}
-->
**REPRODUCED.** The pinned release contains **52 trajectories** and **11,907 operations**. Sequence lengths span **35-821**, with arithmetic mean **228.98** (paper-rounded to **229**) and median **164**. All 52 trajectory IDs match raw-artifact directories containing the workbook, sheet image, operation sequence, and predictability output.

The source audit locates executable symbolic sequencing, region annotation, and LLM refinement stages in the pinned NAPE checkout. Human annotation is supported only as **release provenance**; the repository does not provide an independently executable human-annotation procedure. Counting definitions, paths, and revision are in `claim_1_benchmark.json` in the attached bundle.
