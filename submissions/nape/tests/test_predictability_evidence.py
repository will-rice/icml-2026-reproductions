import json
from pathlib import Path
from typing import cast

import pytest

import icml_2026_repro.predictability_evidence as predictability_evidence
from icml_2026_repro.predictability_evidence import (
    aggregate_predictability,
    build_predictability_evidence,
)

RAW_PATH = Path(__file__).resolve().parents[1] / "external" / "NAPE" / "data" / "raw"


@pytest.fixture
def pinned_clean_nape_checkout(monkeypatch):
    monkeypatch.setattr(
        predictability_evidence,
        "read_git_head",
        lambda _: predictability_evidence.GITHUB_REVISION,
    )
    monkeypatch.setattr(predictability_evidence, "read_git_worktree_status", lambda _: "")


@pytest.fixture
def raw_fixture(tmp_path, pinned_clean_nape_checkout):
    del pinned_clean_nape_checkout
    raw_path = tmp_path / "NAPE" / "data" / "raw"
    trajectory_path = raw_path / "trajectory"
    trajectory_path.mkdir(parents=True)
    predictable_state = {
        "trajectory_name": "trajectory",
        "predictable_count": 2,
        "final_state_size": 4,
        "coverage_pct": 50.0,
        "predictable_properties": {"Sheet1": {"A1": ["value", "number_format"]}},
    }
    (trajectory_path / "predictable_state.json").write_text(
        json.dumps(predictable_state), encoding="utf-8"
    )
    return raw_path, trajectory_path, predictable_state


def test_aggregate_predictability_reproduces_released_ceiling():
    evidence = aggregate_predictability(RAW_PATH)
    observed = cast(dict[str, object], evidence["observed"])

    assert observed["trajectories"] == 52
    assert observed["predictable_properties"] == 126940
    assert observed["final_state_properties"] == 186574
    assert observed["weighted_coverage_pct"] == pytest.approx(68.03734711160227)
    assert observed["mean_coverage_pct"] == pytest.approx(65.99230769230769)
    assert observed["median_coverage_pct"] == pytest.approx(66.34)
    assert observed["trajectories_above_50_pct"] == 44
    assert evidence["evidence_scope"] == "released_oracle_output_recomputation"


def test_aggregate_predictability_rejects_missing_predictable_state(raw_fixture):
    raw_path, trajectory_path, _ = raw_fixture
    (trajectory_path / "predictable_state.json").unlink()

    with pytest.raises(ValueError, match="missing predictable_state.json"):
        aggregate_predictability(raw_path)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda _: [], "JSON must be an object"),
        (
            lambda row: {key: value for key, value in row.items() if key != "trajectory_name"},
            "trajectory_name must be a string",
        ),
        (lambda row: {**row, "trajectory_name": "other"}, "trajectory_name does not match"),
        (lambda row: {**row, "predictable_count": True}, "predictable_count must be an integer"),
        (lambda row: {**row, "predictable_count": 2.0}, "predictable_count must be an integer"),
        (lambda row: {**row, "predictable_count": -1}, "predictable_count must be within"),
        (lambda row: {**row, "predictable_count": 5}, "predictable_count must be within"),
        (lambda row: {**row, "final_state_size": True}, "final_state_size must be an integer"),
        (lambda row: {**row, "final_state_size": 4.0}, "final_state_size must be an integer"),
        (lambda row: {**row, "final_state_size": 0}, "final_state_size must be positive"),
        (lambda row: {**row, "final_state_size": -1}, "final_state_size must be positive"),
        (lambda row: {**row, "coverage_pct": "50"}, "coverage_pct must be a finite number"),
        (lambda row: {**row, "coverage_pct": float("nan")}, "coverage_pct must be a finite number"),
        (lambda row: {**row, "coverage_pct": 101}, "coverage_pct must be within"),
        (lambda row: {**row, "coverage_pct": -1}, "coverage_pct must be within"),
        (lambda row: {**row, "coverage_pct": 51}, "coverage_pct does not match"),
        (
            lambda row: {**row, "predictable_count": 1, "coverage_pct": 25},
            "does not match predictable_properties",
        ),
        (
            lambda row: {**row, "predictable_properties": []},
            "predictable_properties must be an object",
        ),
        (lambda row: {**row, "predictable_properties": {"Sheet1": []}}, "must be an object"),
        (
            lambda row: {**row, "predictable_properties": {"Sheet1": {"A1": "value"}}},
            "must be a list of strings",
        ),
        (
            lambda row: {**row, "predictable_properties": {"Sheet1": {"A1": [1]}}},
            "must be a list of strings",
        ),
    ],
)
def test_aggregate_predictability_rejects_invalid_release_rows(raw_fixture, change, message):
    raw_path, trajectory_path, predictable_state = raw_fixture
    (trajectory_path / "predictable_state.json").write_text(
        json.dumps(change(predictable_state)), encoding="utf-8"
    )

    with pytest.raises(ValueError, match=message):
        aggregate_predictability(raw_path)


def test_aggregate_predictability_rejects_duplicate_trajectory_ids(raw_fixture):
    raw_path, _, predictable_state = raw_fixture
    duplicate_path = raw_path / "z-other"
    duplicate_path.mkdir()
    (duplicate_path / "predictable_state.json").write_text(
        json.dumps(predictable_state), encoding="utf-8"
    )

    with pytest.raises(ValueError, match="duplicate trajectory_name"):
        aggregate_predictability(raw_path)


def test_aggregate_predictability_rejects_dirty_checkout(monkeypatch):
    monkeypatch.setattr(
        predictability_evidence, "read_git_worktree_status", lambda _: " M data/raw"
    )

    with pytest.raises(ValueError, match="checkout is dirty"):
        aggregate_predictability(RAW_PATH)


def test_aggregate_predictability_rejects_mismatched_revision(monkeypatch):
    monkeypatch.setattr(predictability_evidence, "read_git_head", lambda _: "wrong-revision")

    with pytest.raises(ValueError, match="wrong-revision.*expected"):
        aggregate_predictability(RAW_PATH)


def test_build_predictability_evidence_rejects_mismatched_revision(monkeypatch):
    monkeypatch.setattr(predictability_evidence, "read_git_head", lambda _: "wrong-revision")

    with pytest.raises(ValueError, match="wrong-revision.*expected"):
        build_predictability_evidence()


def test_build_predictability_evidence_rejects_dirty_checkout(monkeypatch):
    monkeypatch.setattr(
        predictability_evidence, "read_git_worktree_status", lambda _: " M data/raw"
    )

    with pytest.raises(ValueError, match="checkout is dirty"):
        build_predictability_evidence()


def test_build_predictability_evidence_records_verified_pinned_revision():
    evidence = build_predictability_evidence()

    assert evidence["source_revision"] == predictability_evidence.GITHUB_REVISION
