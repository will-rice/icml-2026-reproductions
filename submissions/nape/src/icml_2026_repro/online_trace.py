"""Run a deterministic online evaluation trace with the official NAPE evaluator."""

import json
import logging
from pathlib import Path
from typing import Optional, Union, cast

from next_action_pred_eval.core.operation import Operation
from next_action_pred_eval.core.symbolic import symbolic_to_operations
from next_action_pred_eval.evaluation import (
    HEURISTIC_STEPS_SAVED,
    ISolver,
    Orchestrator,
    PredictionResult,
)
from next_action_pred_eval.evaluation.stride import StrideConfig, StrideMode

from icml_2026_repro.audit import (
    GITHUB_REVISION,
    REPOSITORY_ROOT,
    read_git_head,
    read_git_worktree_status,
)

GROUND_TRUTH = [
    "VALUE | Sheet1!A1 | 1",
    "VALUE | Sheet1!A2 | 2",
    "VALUE | Sheet1!A3 | 3",
    "VALUE | Sheet1!A4 | 4",
    "VALUE | Sheet1!A5 | 5",
]

SCRIPTED_RESPONSES = [
    GROUND_TRUTH[1:3],
    ["VALUE | Sheet1!A5 | 999"],
    [],
]

RUN_DIRECTORY_NAME = "claim-2-online-trace"
EXPERIMENT_NAME = "claim-2-fixture"
TRACE_LOGGER_NAME = f"trajectory.{EXPERIMENT_NAME}"
GENERATED_RUN_ARTIFACTS = (
    "divergence_log.jsonl",
    "experiment_summary.json",
    "final_trajectory.jsonl",
    "predictions.jsonl",
    "target_state.json",
    "timeline.jsonl",
    "trajectory.log",
)


class ScriptedSolver(ISolver):
    """Return a fixed symbolic prediction on each call."""

    def __init__(self, responses: list[list[str]]) -> None:
        self.responses = responses
        self.response_index = 0

    def predict(
        self,
        previous_actions: list[Union[Operation, str]],
        workbook_state: Optional[dict[str, object]] = None,
        context: Optional[dict[str, object]] = None,
    ) -> PredictionResult:
        """Return the next scripted response as parsed NAPE operations."""
        del previous_actions, workbook_state, context
        if self.response_index >= len(self.responses):
            raise RuntimeError("scripted solver responses exhausted")
        predicted_symbolic = self.responses[self.response_index]
        self.response_index += 1
        return PredictionResult(
            predicted_operations=symbolic_to_operations(predicted_symbolic),
            predicted_symbolic=predicted_symbolic,
            input_tokens=0,
            output_tokens=0,
            total_tokens=0,
            metadata={"solver": "scripted"},
        )

    def reset(self) -> None:
        """Restart the scripted response sequence."""
        self.response_index = 0


def _read_timeline(path: Path) -> list[dict[str, object]]:
    """Parse the official recorder timeline JSONL."""
    events: list[dict[str, object]] = []
    with path.open(encoding="utf-8") as timeline_file:
        for line_number, line in enumerate(timeline_file, start=1):
            try:
                event = json.loads(line)
            except json.JSONDecodeError as error:
                raise RuntimeError(f"invalid timeline event at line {line_number}") from error
            if not isinstance(event, dict):
                raise RuntimeError(f"timeline event at line {line_number} is not an object")
            events.append(event)
    return events


def _require_timeline_int(event: dict[str, object], field: str) -> int:
    """Return a required integer recorder field."""
    value = event.get(field)
    if isinstance(value, bool) or not isinstance(value, int):
        raise RuntimeError(f"timeline event has invalid {field}")
    return value


def _verify_nape_checkout() -> None:
    """Require the exact clean NAPE checkout used by the editable dependency."""
    nape_path = REPOSITORY_ROOT / "external" / "NAPE"
    revision = read_git_head(nape_path)
    if revision != GITHUB_REVISION:
        raise RuntimeError(f"NAPE checkout is at {revision}, expected {GITHUB_REVISION}")
    status = read_git_worktree_status(nape_path)
    if status:
        changed_paths = ", ".join(status.splitlines())
        raise RuntimeError(f"NAPE checkout is dirty: {changed_paths}")


def _prepare_run_directory(output_dir: Path) -> Path:
    """Create the dedicated run directory and reset known recorder artifacts."""
    if output_dir.is_symlink():
        raise RuntimeError("claim-2 output directory must not be a symlink")
    output_dir.mkdir(parents=True, exist_ok=True)
    run_directory = output_dir / RUN_DIRECTORY_NAME
    if run_directory.is_symlink():
        raise RuntimeError("claim-2 run directory must not be a symlink")
    run_directory.mkdir(parents=True, exist_ok=True)
    for artifact_name in GENERATED_RUN_ARTIFACTS:
        artifact_path = run_directory / artifact_name
        if artifact_path.is_symlink():
            artifact_path.unlink()
        elif artifact_path.exists():
            if not artifact_path.is_file():
                raise RuntimeError(f"generated artifact path is not a file: {artifact_name}")
            artifact_path.write_text("", encoding="utf-8")
    return run_directory


def _close_trace_logger_handlers() -> None:
    """Close handlers owned by the fixed NAPE trajectory logger."""
    trajectory_logger = logging.getLogger(TRACE_LOGGER_NAME)
    for handler in tuple(trajectory_logger.handlers):
        trajectory_logger.removeHandler(handler)
        handler.close()


def _future_counts(
    event: dict[str, object],
    user_event: dict[str, object],
    accepted: bool,
    predicted_count: int,
) -> tuple[int, int, Optional[int], Optional[int]]:
    """Derive uncapped future counts from official timeline length fields."""
    user_t = _require_timeline_int(user_event, "t")
    prediction_t = _require_timeline_int(event, "t")
    user_future_count = _require_timeline_int(user_event, "future_len")
    if not accepted:
        if prediction_t != user_t:
            raise RuntimeError("rejected prediction has inconsistent recorder pointer")
        return user_future_count, user_future_count, None, None

    gt_len_before = _require_timeline_int(event, "gt_len_before")
    gt_len_after = _require_timeline_int(event, "gt_len_after")
    recorder_t_after = _require_timeline_int(event, "t_after")
    if prediction_t != user_t + predicted_count:
        raise RuntimeError("accepted prediction has inconsistent recorder pointer")
    if recorder_t_after != prediction_t + predicted_count:
        raise RuntimeError("accepted prediction has inconsistent recorder t_after pointer")
    future_before_count = gt_len_before - user_t
    future_after_count = gt_len_after - prediction_t
    if future_before_count != user_future_count:
        raise RuntimeError("accepted prediction has inconsistent future-before fields")
    if future_before_count < 0 or future_after_count < 0:
        raise RuntimeError("accepted prediction has negative future count")
    return future_before_count, future_after_count, gt_len_before, gt_len_after


def _prediction_decision(
    event: dict[str, object], user_event: dict[str, object]
) -> dict[str, object]:
    """Validate one official prediction event and build its evidence record."""
    predicted_ops = event.get("predicted_ops")
    accepted = event.get("accepted")
    if (
        not isinstance(predicted_ops, list)
        or not predicted_ops
        or not all(isinstance(operation, str) for operation in predicted_ops)
    ):
        raise RuntimeError("online trace has an empty non-empty prediction transition")
    if not isinstance(accepted, bool):
        raise RuntimeError("online trace prediction event has an invalid schema")

    prediction_index = _require_timeline_int(event, "prediction_index")
    user_step = _require_timeline_int(event, "user_step")
    if user_step != _require_timeline_int(user_event, "user_step"):
        raise RuntimeError("prediction was not recorded after its user action")
    user_t = _require_timeline_int(user_event, "t")
    prediction_t = _require_timeline_int(event, "t")
    future_before_count, future_after_count, gt_len_before, gt_len_after = _future_counts(
        event, user_event, accepted, len(predicted_ops)
    )
    return {
        "prediction_index": prediction_index,
        "user_step": user_step,
        "user_t": user_t,
        "t": prediction_t,
        "accepted": accepted,
        "predicted_ops": cast(list[str], predicted_ops),
        "future_before_count": future_before_count,
        "future_after_count": future_after_count,
        "gt_len_before": gt_len_before,
        "gt_len_after": gt_len_after,
    }


def build_online_trace_evidence(output_dir: Path) -> dict[str, object]:
    """Build evidence for the online evaluation transitions."""
    _verify_nape_checkout()
    _close_trace_logger_handlers()
    run_directory = _prepare_run_directory(output_dir)
    orchestrator = Orchestrator(
        solver=ScriptedSolver(SCRIPTED_RESPONSES),
        stride_config=StrideConfig(mode=StrideMode.FIXED_INTERVAL, interval=1),
        acceptance_heuristics=[HEURISTIC_STEPS_SAVED],
        output_dir=run_directory,
    )
    try:
        orchestrator.run_experiment(
            cast(list[Union[Operation, str]], GROUND_TRUTH),
            experiment_name=EXPERIMENT_NAME,
            online_mode=True,
        )
    finally:
        _close_trace_logger_handlers()

    timeline_path = run_directory / "timeline.jsonl"
    if not timeline_path.exists():
        raise RuntimeError("official evaluator did not write timeline.jsonl")
    events = _read_timeline(timeline_path)

    decisions: list[dict[str, object]] = []
    last_user_event: Optional[dict[str, object]] = None
    for event in events:
        event_type = event.get("event")
        if event_type == "user_step":
            last_user_event = event
            continue
        if event_type != "prediction":
            continue
        if last_user_event is None:
            raise RuntimeError("prediction was not recorded after its user action")
        decisions.append(_prediction_decision(event, last_user_event))

    accepted_decisions = [decision for decision in decisions if decision["accepted"] is True]
    rejected_decisions = [decision for decision in decisions if decision["accepted"] is False]
    future_updates = [
        decision
        for decision in accepted_decisions
        if decision["future_before_count"] != decision["future_after_count"]
    ]
    if not accepted_decisions or not rejected_decisions:
        raise RuntimeError("online trace did not record both accepted and rejected predictions")
    has_multi_action_acceptance = any(
        len(cast(list[str], decision["predicted_ops"])) >= 2 for decision in accepted_decisions
    )
    if not has_multi_action_acceptance:
        raise RuntimeError("online trace did not record a multi-action accepted prediction")
    if not future_updates:
        raise RuntimeError("online trace did not record an accepted future update")

    return {
        "source_revision": GITHUB_REVISION,
        "fixture": list(GROUND_TRUTH),
        "decisions": decisions,
        "summary": {
            "predictions": len(decisions),
            "accepted": len(accepted_decisions),
            "rejected": len(rejected_decisions),
            "predictions_after_user_actions": True,
            "future_was_updated": True,
        },
        "verdict": "verified",
    }
