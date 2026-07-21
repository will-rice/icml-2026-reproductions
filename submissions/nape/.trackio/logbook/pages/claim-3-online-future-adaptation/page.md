# Claim 3: Online future adaptation


---
<!-- trackio-cell
{"type": "code", "id": "cell_0cef6563e8b6", "created_at": "2026-07-21T16:00:06+00:00", "title": "Exercise one official adaptation case per trajectory", "command": ["uv", "run", "pytest", "tests/test_future_adaptation_evidence.py", "tests/test_online_trace.py", "-q"], "exit_code": 0, "duration_s": 16.85}
-->
````bash
$ uv run pytest tests/test_future_adaptation_evidence.py tests/test_online_trace.py -q
````

exit 0 · 16.9s


````python title=test_future_adaptation_evidence.py
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

````


````python title=test_online_trace.py
import logging
import subprocess
from typing import cast

import pytest
from next_action_pred_eval.evaluation.experiment_recorder import TIMELINE_CONTEXT_SIZE

from icml_2026_repro import online_trace
from icml_2026_repro.online_trace import GROUND_TRUTH, build_online_trace_evidence

EXPECTED_GROUND_TRUTH = [
    "VALUE | Sheet1!A1 | 1",
    "VALUE | Sheet1!A2 | 2",
    "VALUE | Sheet1!A3 | 3",
    "VALUE | Sheet1!A4 | 4",
    "VALUE | Sheet1!A5 | 5",
]


def test_online_trace_covers_prediction_decisions_and_future_update(tmp_path):
    evidence = build_online_trace_evidence(tmp_path)
    summary = cast(dict[str, object], evidence["summary"])
    decisions = cast(list[dict[str, object]], evidence["decisions"])

    assert evidence["verdict"] == "verified"
    assert summary["accepted"] == 1
    assert summary["rejected"] == 1
    assert summary["predictions_after_user_actions"] is True
    assert summary["future_was_updated"] is True
    assert {decision["accepted"] for decision in decisions} == {True, False}


def test_online_trace_uses_exact_fixture_and_multi_action_acceptance(tmp_path):
    evidence = build_online_trace_evidence(tmp_path)
    decisions = cast(list[dict[str, object]], evidence["decisions"])
    accepted = next(decision for decision in decisions if decision["accepted"] is True)

    assert GROUND_TRUTH == EXPECTED_GROUND_TRUTH
    assert evidence["fixture"] == EXPECTED_GROUND_TRUTH
    assert len(cast(list[str], accepted["predicted_ops"])) >= 2


def test_online_trace_is_idempotent_in_existing_output_directory(tmp_path):
    sentinel = tmp_path / "caller-file.txt"
    sentinel.write_text("preserve me", encoding="utf-8")

    first = build_online_trace_evidence(tmp_path)
    second = build_online_trace_evidence(tmp_path)
    decisions = cast(list[dict[str, object]], second["decisions"])

    assert second == first
    assert len(decisions) == 2
    assert sentinel.read_text(encoding="utf-8") == "preserve me"


def test_online_trace_future_counts_use_uncapped_timeline_lengths(tmp_path):
    evidence = build_online_trace_evidence(tmp_path)
    decisions = cast(list[dict[str, object]], evidence["decisions"])
    accepted = next(decision for decision in decisions if decision["accepted"] is True)

    user_t = cast(int, accepted["user_t"])
    prediction_t = cast(int, accepted["t"])
    gt_len_before = cast(int, accepted["gt_len_before"])
    gt_len_after = cast(int, accepted["gt_len_after"])
    assert accepted["future_before_count"] == gt_len_before - user_t
    assert accepted["future_after_count"] == gt_len_after - prediction_t
    assert accepted["future_before_count"] != accepted["future_after_count"]


def test_online_trace_rejects_symlinked_run_directory_without_touching_target(tmp_path):
    target = tmp_path / "target"
    target.mkdir()
    target_artifact = target / "timeline.jsonl"
    target_artifact.write_text("caller data\n", encoding="utf-8")
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    (output_dir / online_trace.RUN_DIRECTORY_NAME).symlink_to(target, target_is_directory=True)

    with pytest.raises(RuntimeError, match="symlink"):
        build_online_trace_evidence(output_dir)

    assert target_artifact.read_text(encoding="utf-8") == "caller data\n"


def test_online_trace_rejects_symlinked_output_directory_without_touching_target(tmp_path):
    target = tmp_path / "target"
    run_directory = target / online_trace.RUN_DIRECTORY_NAME
    run_directory.mkdir(parents=True)
    target_artifact = run_directory / "timeline.jsonl"
    target_artifact.write_text("caller data\n", encoding="utf-8")
    output_dir = tmp_path / "output"
    output_dir.symlink_to(target, target_is_directory=True)

    with pytest.raises(RuntimeError, match="output directory.*symlink"):
        build_online_trace_evidence(output_dir)

    assert target_artifact.read_text(encoding="utf-8") == "caller data\n"


def test_online_trace_writes_isolated_logs_for_distinct_output_directories(tmp_path):
    first_output = tmp_path / "first"
    second_output = tmp_path / "second"

    build_online_trace_evidence(first_output)
    first_log = first_output / online_trace.RUN_DIRECTORY_NAME / "trajectory.log"
    first_log_before_second_run = first_log.read_text(encoding="utf-8")
    build_online_trace_evidence(second_output)
    second_log = second_output / online_trace.RUN_DIRECTORY_NAME / "trajectory.log"

    assert first_log.exists()
    assert second_log.exists()
    assert first_log.read_text(encoding="utf-8") == first_log_before_second_run
    second_log_text = second_log.read_text(encoding="utf-8")
    assert str(first_output) in first_log_before_second_run
    assert str(second_output) not in first_log_before_second_run
    assert str(second_output) in second_log_text
    assert str(first_output) not in second_log_text


def test_online_trace_closes_trace_logger_handlers_after_exception(tmp_path, monkeypatch):
    monkeypatch.setattr(online_trace, "SCRIPTED_RESPONSES", [])

    with pytest.raises(RuntimeError, match="responses exhausted"):
        build_online_trace_evidence(tmp_path)

    assert logging.getLogger("trajectory.claim-2-fixture").handlers == []


def test_online_trace_rejects_mismatched_nape_revision(tmp_path, monkeypatch):
    monkeypatch.setattr(online_trace, "read_git_head", lambda _: "wrong-revision", raising=False)

    with pytest.raises(RuntimeError, match="wrong-revision.*expected"):
        build_online_trace_evidence(tmp_path)


def test_read_git_worktree_status_reports_tracked_staged_and_untracked_changes(tmp_path):
    repository = tmp_path / "repository"
    subprocess.run(["git", "init", str(repository)], check=True, capture_output=True)
    tracked = repository / "tracked.py"
    tracked.write_text("original\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(repository), "add", "tracked.py"], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(repository),
            "-c",
            "user.name=Test User",
            "-c",
            "user.email=test@example.com",
            "commit",
            "-m",
            "test fixture",
        ],
        check=True,
        capture_output=True,
    )
    tracked.write_text("modified\n", encoding="utf-8")
    (repository / "staged.py").write_text("staged\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(repository), "add", "staged.py"], check=True)
    (repository / "untracked.py").write_text("untracked\n", encoding="utf-8")

    status = online_trace.read_git_worktree_status(repository)

    assert " M tracked.py" in status
    assert "A  staged.py" in status
    assert "?? untracked.py" in status


def test_online_trace_rejects_dirty_nape_checkout(tmp_path, monkeypatch):
    repository_root = tmp_path / "repository-root"
    nape_path = repository_root / "external" / "NAPE"
    nape_path.mkdir(parents=True)
    subprocess.run(["git", "init", str(nape_path)], check=True, capture_output=True)
    source_path = nape_path / "src" / "dirty.py"
    source_path.parent.mkdir()
    source_path.write_text("dirty = True\n", encoding="utf-8")
    monkeypatch.setattr(online_trace, "REPOSITORY_ROOT", repository_root, raising=False)
    monkeypatch.setattr(
        online_trace,
        "read_git_head",
        lambda _: online_trace.GITHUB_REVISION,
        raising=False,
    )

    with pytest.raises(RuntimeError, match=r"dirty.*src/dirty\.py"):
        build_online_trace_evidence(tmp_path / "output")


def test_prediction_decision_uses_uncapped_counts_and_validates_recorder_pointers():
    predicted_ops = ["VALUE | Sheet1!A11 | 11", "VALUE | Sheet1!A12 | 12"]
    user_t = 10
    future_before_count = TIMELINE_CONTEXT_SIZE + 15
    prediction_t = user_t + len(predicted_ops)
    user_event: dict[str, object] = {
        "event": "user_step",
        "t": user_t,
        "user_step": 10,
        "op": "VALUE | Sheet1!A10 | 10",
        "history_len": user_t,
        "future_len": future_before_count,
    }
    prediction_event: dict[str, object] = {
        "event": "prediction",
        "t": prediction_t,
        "t_after": prediction_t + len(predicted_ops),
        "user_step": 10,
        "prediction_index": 1,
        "history_tail": [],
        "future_head": [f"capped-{index}" for index in range(TIMELINE_CONTEXT_SIZE)],
        "predicted_ops": predicted_ops,
        "accepted": True,
        "gt_len_before": user_t + future_before_count,
        "gt_len_after": user_t + future_before_count,
    }

    decision = online_trace._prediction_decision(prediction_event, user_event)

    assert decision["future_before_count"] == future_before_count
    assert decision["future_after_count"] == future_before_count - len(predicted_ops)
    assert decision["future_before_count"] != len(prediction_event["future_head"])
    for field in ("t", "t_after"):
        invalid_event = dict(prediction_event)
        invalid_event[field] = cast(int, invalid_event[field]) + 1
        with pytest.raises(RuntimeError, match="pointer"):
            online_trace._prediction_decision(invalid_event, user_event)

````


````output
...............................                                          [100%]
31 passed in 16.85s

````


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_446537eb63a6", "created_at": "2026-07-21T16:05:26+00:00", "title": "Outcome and evidence scope"}
-->
**REPRODUCED.** The official evaluator completed **one deterministic adaptation case per each of the 52 released trajectories**, with no skips. This is not a full per-action model rollout. Satisfied future operations were removed in **50/52** cases, inverse operations were inserted in **52/52**, and the rebuilt state preserved the original target in **52/52**. The release sweep did not trigger residual correction, so **one fixed residual-patch fixture** separately executes that mechanism and preserves its target.

The small orchestrator trace is supporting evidence only: it establishes prediction-after-action ordering plus one accepted and one rejected decision. It is not used as release-wide mechanism evidence. Definitions, per-case rows, denominators, and the fixture are in `claim_3_future_adaptation.json` in the attached bundle.
