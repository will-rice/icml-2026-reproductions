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
