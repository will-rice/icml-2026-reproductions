import json
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest

import icml_2026_repro.future_adaptation_evidence as future_adaptation_evidence
from icml_2026_repro.audit import GITHUB_REVISION
from icml_2026_repro.future_adaptation_evidence import (
    audit_future_adaptation,
    audit_residual_patch_fixture,
    build_future_adaptation_evidence,
)

TRAJECTORY_PATH = (
    Path(__file__).resolve().parents[1] / "external" / "NAPE" / "data" / "trajectories"
)
NAPE_PATH = TRAJECTORY_PATH.parents[1]
EXPECTED_COUNTING_DEFINITIONS = {
    "eligible_case": (
        "One deterministic adaptation case from each structurally valid trajectory JSON file "
        "containing at least two symbolic operations."
    ),
    "executed_case": (
        "An eligible case for which parsing and pre-evaluation application are supported "
        "and the evaluator, simulator, rebuild, and final comparison complete."
    ),
    "skipped_case": (
        "An eligible case is skipped only when symbolic parsing or pre-evaluation state "
        "application explicitly raises NotImplementedError for an unsupported operation; "
        "skipped cases are excluded from every mechanism denominator."
    ),
    "deterministic_mutation": (
        "After operation 0, predict the exact operation 1 followed by VALUE on the first "
        "operation's sheet at ZZ999 with value 987654321."
    ),
    "removal_case": (
        "An executed case with at least one operation in changes['operations_removed']."
    ),
    "inverse_insertion_case": (
        "An executed case with at least one operation in changes['inverse_ops_added']."
    ),
    "residual_patch_case": (
        "An executed case with metadata['missing_ops_count'] greater than zero."
    ),
    "target_preserved_case": (
        "An executed case whose rebuilt-versus-target comparison has zero false positives, "
        "zero false negatives, and zero mismatches."
    ),
}
EXPECTED_RESIDUAL_FIXTURE_DEFINITIONS = {
    "eligible_case": (
        "The exact ground-truth and prediction pair in this deterministic fixture is one "
        "eligible case."
    ),
    "executed_case": (
        "The eligible fixture is executed when the official evaluator, simulator, rebuild, "
        "and final comparison complete."
    ),
    "residual_patch_case": (
        "The executed fixture is a residual patch case when metadata['missing_ops_count'] "
        "is greater than zero."
    ),
    "target_preserved_case": (
        "The executed fixture preserves the target when the rebuilt-versus-target "
        "comparison has zero false positives, zero false negatives, and zero mismatches."
    ),
    "skip_treatment": (
        "The deterministic fixture does not permit skips; any failure aborts evidence generation."
    ),
}


def test_audit_future_adaptation_preserves_all_release_targets():
    evidence = audit_future_adaptation(NAPE_PATH)

    assert evidence["summary"] == {
        "trajectories": 52,
        "removal_cases": 50,
        "inverse_insertion_cases": 52,
        "residual_patch_cases": 0,
        "target_preserved_cases": 52,
        "skipped_cases": 0,
    }
    assert evidence["counting_definitions"] == EXPECTED_COUNTING_DEFINITIONS
    assert evidence["case_counts"] == {
        "release_trajectories": 52,
        "eligible_cases": 52,
        "executed_cases": 52,
        "skipped_cases": 0,
    }
    assert evidence["denominators"] == {
        "removal_cases": 52,
        "inverse_insertion_cases": 52,
        "residual_patch_cases": 52,
        "target_preserved_cases": 52,
    }
    summary = cast(dict[str, int], evidence["summary"])
    denominators = cast(dict[str, int], evidence["denominators"])
    assert summary["residual_patch_cases"] == 0
    assert denominators["residual_patch_cases"] == 52
    trajectories = cast(list[dict[str, object]], evidence["trajectories"])
    assert all(row["target_preserved"] for row in trajectories)


def test_residual_patch_fixture_synthesizes_missing_operation():
    evidence = audit_residual_patch_fixture()

    assert evidence["mechanism"] == "residual_correction"
    assert evidence["counting_definitions"] == EXPECTED_RESIDUAL_FIXTURE_DEFINITIONS
    assert evidence["case_counts"] == {
        "eligible_cases": 1,
        "executed_cases": 1,
        "skipped_cases": 0,
        "residual_patch_cases": 1,
        "target_preserved_cases": 1,
    }
    assert evidence["denominators"] == {
        "executed_cases": 1,
        "residual_patch_cases": 1,
        "target_preserved_cases": 1,
    }
    assert evidence["denominator_definitions"] == {
        "executed_cases": "eligible_cases",
        "residual_patch_cases": "executed_cases",
        "target_preserved_cases": "executed_cases",
    }
    assert evidence["evidence_scope"] == "deterministic_fixture_mechanism_proof"
    assert evidence["ground_truth"] == [
        "VALUE | Sheet1!A1 | 1",
        "VALUE | Sheet1!A2 | 1",
        "VALUE | Sheet1!A2 | 1",
        "VALUE | Sheet1!A2 | 1",
    ]
    assert evidence["prediction"] == [
        "VALUE | Sheet1!A1 | 1",
        "VALUE | Sheet1!A1 | 2",
    ]
    assert evidence["missing_ops_count"] == 1
    assert evidence["target_preserved"] is True


def test_audit_future_adaptation_rejects_malformed_operation_with_filename(tmp_path):
    trajectory_path = tmp_path / "malformed.json"
    trajectory_path.write_text(
        json.dumps(
            {
                "operations": [
                    "VALUE | Sheet1!A1 | 1",
                    "VALUE | Sheet1!not-a-range | 2",
                ]
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="malformed.json"):
        future_adaptation_evidence._audit_fixture_trajectory_directory(tmp_path)


def test_audit_future_adaptation_rejects_empty_release(tmp_path):
    with pytest.raises(ValueError, match="no trajectory files"):
        future_adaptation_evidence._audit_fixture_trajectory_directory(tmp_path)


def test_audit_future_adaptation_skips_only_unsupported_pre_evaluation_application(
    tmp_path, monkeypatch
):
    trajectory_path = tmp_path / "unsupported.json"
    trajectory_path.write_text(
        json.dumps(
            {
                "operations": [
                    "VALUE | Sheet1!A1 | 1",
                    "VALUE | Sheet1!A2 | 2",
                ]
            }
        ),
        encoding="utf-8",
    )

    class UnsupportedStateBuilder:
        def apply_operations(self, operations):
            del operations
            raise NotImplementedError("unsupported operation application")

    monkeypatch.setattr(future_adaptation_evidence, "StateBuilder", UnsupportedStateBuilder)

    evidence = future_adaptation_evidence._audit_fixture_trajectory_directory(tmp_path)

    assert evidence["case_counts"] == {
        "release_trajectories": 1,
        "eligible_cases": 1,
        "executed_cases": 0,
        "skipped_cases": 1,
    }
    assert evidence["denominators"] == {
        "removal_cases": 0,
        "inverse_insertion_cases": 0,
        "residual_patch_cases": 0,
        "target_preserved_cases": 0,
    }
    trajectories = cast(list[dict[str, object]], evidence["trajectories"])
    assert trajectories == [
        {
            "name": "unsupported",
            "operations_removed": 0,
            "inverse_ops_added": 0,
            "missing_ops_count": 0,
            "target_preserved": False,
            "skipped": True,
            "reason": (
                "unsupported.json: unsupported operation during pre-evaluation state "
                "application: unsupported operation application"
            ),
        }
    ]


@pytest.mark.parametrize("stage", ["evaluator", "simulator", "comparator"])
def test_downstream_not_implemented_error_propagates_with_filename(tmp_path, monkeypatch, stage):
    trajectory_path = tmp_path / "downstream.json"
    trajectory_path.write_text(
        json.dumps(
            {
                "operations": [
                    "VALUE | Sheet1!A1 | 1",
                    "VALUE | Sheet1!A2 | 2",
                ]
            }
        ),
        encoding="utf-8",
    )

    if stage == "evaluator":

        class FailingEvaluator:
            def evaluate(self, **kwargs):
                del kwargs
                raise NotImplementedError("evaluator failure")

        monkeypatch.setattr(future_adaptation_evidence, "StepEvaluator", FailingEvaluator)
    elif stage == "simulator":

        class FailingManager:
            def simulate_future_edits(self, **kwargs):
                del kwargs
                raise NotImplementedError("simulator failure")

        monkeypatch.setattr(future_adaptation_evidence, "FutureEditsManager", FailingManager)
    else:

        class FailingComparator:
            def __init__(self, ignore_defaults):
                assert ignore_defaults is True

            def compare(self, predicted_state, true_state, skip_ops_diff):
                del predicted_state, true_state, skip_ops_diff
                raise NotImplementedError("comparator failure")

        monkeypatch.setattr(future_adaptation_evidence, "StateComparator", FailingComparator)

    with pytest.raises(NotImplementedError, match=rf"downstream\.json: {stage} failed") as error:
        future_adaptation_evidence._audit_fixture_trajectory_directory(tmp_path)

    assert isinstance(error.value.__cause__, NotImplementedError)


def test_unexpected_pre_evaluation_state_failure_propagates_with_filename(tmp_path, monkeypatch):
    trajectory_path = tmp_path / "state-failure.json"
    trajectory_path.write_text(
        json.dumps(
            {
                "operations": [
                    "VALUE | Sheet1!A1 | 1",
                    "VALUE | Sheet1!A2 | 2",
                ]
            }
        ),
        encoding="utf-8",
    )

    class FailingStateBuilder:
        def apply_operations(self, operations):
            del operations
            raise RuntimeError("state builder failure")

    monkeypatch.setattr(future_adaptation_evidence, "StateBuilder", FailingStateBuilder)

    with pytest.raises(
        RuntimeError,
        match=r"state-failure\.json: pre-evaluation state application failed",
    ) as error:
        future_adaptation_evidence._audit_fixture_trajectory_directory(tmp_path)

    assert isinstance(error.value.__cause__, RuntimeError)


def test_audit_future_adaptation_raises_when_final_state_is_unequal(monkeypatch):
    class UnequalComparator:
        def __init__(self, ignore_defaults):
            assert ignore_defaults is True

        def compare(self, predicted_state, true_state, skip_ops_diff):
            del predicted_state, true_state
            assert skip_ops_diff is True
            return SimpleNamespace(false_positives=1, false_negatives=0, mismatches=0)

    monkeypatch.setattr(future_adaptation_evidence, "StateComparator", UnequalComparator)

    with pytest.raises(RuntimeError, match="target state"):
        audit_future_adaptation(NAPE_PATH)


def test_audit_future_adaptation_is_deterministic():
    assert audit_future_adaptation(NAPE_PATH) == audit_future_adaptation(NAPE_PATH)


def test_release_audit_rejects_copied_trajectory_input(tmp_path):
    copied_nape_path = tmp_path / "NAPE"
    copied_trajectory_path = copied_nape_path / "data" / "trajectories"
    copied_trajectory_path.mkdir(parents=True)
    source_path = next(TRAJECTORY_PATH.glob("*.json"))
    (copied_trajectory_path / source_path.name).write_bytes(source_path.read_bytes())
    subprocess.run(["git", "init", str(copied_nape_path)], check=True, capture_output=True)
    subprocess.run(
        ["git", "-C", str(copied_nape_path), "add", "data/trajectories"],
        check=True,
    )
    subprocess.run(
        [
            "git",
            "-C",
            str(copied_nape_path),
            "-c",
            "user.name=Test User",
            "-c",
            "user.email=test@example.com",
            "commit",
            "-m",
            "copied fixture",
        ],
        check=True,
        capture_output=True,
    )

    with pytest.raises(RuntimeError, match="NAPE checkout is at .* expected"):
        audit_future_adaptation(copied_nape_path)


def test_build_future_adaptation_evidence_composes_official_evidence(tmp_path, monkeypatch):
    case_counts = {
        "release_trajectories": 52,
        "eligible_cases": 52,
        "executed_cases": 52,
        "skipped_cases": 0,
    }
    denominators = {
        "removal_cases": 52,
        "inverse_insertion_cases": 52,
        "residual_patch_cases": 52,
        "target_preserved_cases": 52,
    }
    release_evidence = {
        "summary": {"trajectories": 52, "residual_patch_cases": 0},
        "counting_definitions": EXPECTED_COUNTING_DEFINITIONS,
        "case_counts": case_counts,
        "denominators": denominators,
    }
    residual_evidence = {
        "mechanism": "residual_correction",
        "counting_definitions": EXPECTED_RESIDUAL_FIXTURE_DEFINITIONS,
        "case_counts": {
            "eligible_cases": 1,
            "executed_cases": 1,
            "skipped_cases": 0,
            "residual_patch_cases": 1,
            "target_preserved_cases": 1,
        },
        "denominators": {
            "executed_cases": 1,
            "residual_patch_cases": 1,
            "target_preserved_cases": 1,
        },
        "denominator_definitions": {
            "executed_cases": "eligible_cases",
            "residual_patch_cases": "executed_cases",
            "target_preserved_cases": "executed_cases",
        },
        "missing_ops_count": 1,
        "target_preserved": True,
        "evidence_scope": "deterministic_fixture_mechanism_proof",
    }
    trace_evidence = {"summary": {"accepted": 1, "rejected": 1}}
    monkeypatch.setattr(
        future_adaptation_evidence,
        "audit_future_adaptation",
        lambda _: release_evidence,
    )
    monkeypatch.setattr(
        future_adaptation_evidence,
        "audit_residual_patch_fixture",
        lambda: residual_evidence,
    )
    monkeypatch.setattr(
        future_adaptation_evidence,
        "build_online_trace_evidence",
        lambda _: trace_evidence,
    )

    assert build_future_adaptation_evidence(tmp_path) == {
        "claim": (
            "NAPE's future-edit implementation removes satisfied future operations, prepends "
            "inverses for false positives, and patches residual differences to preserve the "
            "target state after accepted predictions."
        ),
        "source_revision": GITHUB_REVISION,
        "verified_input_root": "external/NAPE",
        "input_path": "external/NAPE/data/trajectories/*.json",
        "release_case_scope": (
            "One deterministic adaptation case per each of the 52 released trajectories; "
            "this is not a full per-action model rollout."
        ),
        "counting_definitions": EXPECTED_COUNTING_DEFINITIONS,
        "case_counts": case_counts,
        "denominators": denominators,
        "release_sweep": release_evidence,
        "residual_patch_fixture": residual_evidence,
        "orchestrator_trace": trace_evidence,
        "evidence_scope": "official_evaluator_one_case_per_release_trajectory_and_fixture",
        "verdict": "reproduced",
    }


@pytest.mark.parametrize(
    "produce_evidence",
    [
        lambda output_dir: audit_future_adaptation(NAPE_PATH),
        lambda output_dir: audit_residual_patch_fixture(),
        build_future_adaptation_evidence,
    ],
)
def test_public_evidence_rejects_mismatched_nape_revision(tmp_path, monkeypatch, produce_evidence):
    monkeypatch.setattr(
        future_adaptation_evidence,
        "read_git_head",
        lambda _: "wrong-revision",
    )

    with pytest.raises(RuntimeError, match="wrong-revision.*expected"):
        produce_evidence(tmp_path)


@pytest.mark.parametrize(
    "produce_evidence",
    [
        lambda output_dir: audit_future_adaptation(NAPE_PATH),
        lambda output_dir: audit_residual_patch_fixture(),
        build_future_adaptation_evidence,
    ],
)
def test_public_evidence_rejects_dirty_nape_checkout(tmp_path, monkeypatch, produce_evidence):
    monkeypatch.setattr(future_adaptation_evidence, "read_git_worktree_status", lambda _: " M x")

    with pytest.raises(RuntimeError, match="dirty"):
        produce_evidence(tmp_path)
