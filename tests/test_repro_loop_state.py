"""Tests for persistent ICML reproduction loop state."""

import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest


STATE_MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "skills"
    / "icml-repro-loop"
    / "scripts"
    / "state.py"
)


def state_module():
    """Load the state script without requiring package scaffolding."""
    spec = importlib.util.spec_from_file_location("repro_loop_state", STATE_MODULE_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_new_state_starts_idle():
    state = state_module().new_state()

    assert state == {
        "version": 1,
        "phase": "idle",
        "current": None,
        "history": [],
        "rejections": [],
        "total_api_cost_usd": 0.0,
    }


def test_save_and_load_round_trip_without_temporary_file(tmp_path: Path):
    module = state_module()
    path = tmp_path / "repro-loop.json"
    state = module.new_state()

    module.save_state(path, state)

    assert module.load_state(path) == state
    assert list(tmp_path.glob("*.tmp")) == []


@pytest.mark.parametrize(
    ("state", "field"),
    [
        ({"version": 1}, "keys"),
        (
            {
                "version": 2,
                "phase": "idle",
                "current": None,
                "history": [],
                "rejections": [],
                "total_api_cost_usd": 0.0,
            },
            "version",
        ),
        (
            {
                "version": 1,
                "phase": "unknown",
                "current": None,
                "history": [],
                "rejections": [],
                "total_api_cost_usd": 0.0,
            },
            "phase",
        ),
        (
            {
                "version": 1,
                "phase": "idle",
                "current": None,
                "history": [],
                "rejections": [],
                "total_api_cost_usd": -0.01,
            },
            "total_api_cost_usd",
        ),
        (
            {
                "version": 1,
                "phase": "idle",
                "current": None,
                "history": {},
                "rejections": [],
                "total_api_cost_usd": 0.0,
            },
            "history",
        ),
        (
            {
                "version": 1,
                "phase": "selected",
                "current": None,
                "history": [],
                "rejections": [],
                "total_api_cost_usd": 0.0,
            },
            "current",
        ),
    ],
)
def test_save_rejects_invalid_state_with_field_name(
    tmp_path: Path, state: dict, field: str
):
    with pytest.raises(ValueError, match=field):
        state_module().save_state(tmp_path / "repro-loop.json", state)


def test_select_paper_records_estimated_cost_without_mutating_state():
    module = state_module()
    initial = module.new_state()
    paper = {
        "paper_id": "icml-2026-001",
        "title": "Reliable Reproductions",
        "slug": "reliable-reproductions",
        "estimated_api_cost_usd": 4.25,
    }

    selected = module.select_paper(initial, paper)

    assert initial == module.new_state()
    assert selected["phase"] == "selected"
    assert selected["current"] == {
        **paper,
        "project_path": "submissions/reliable-reproductions",
        "polls": [],
    }


@pytest.mark.parametrize("field", ["paper_id", "title", "slug"])
def test_select_paper_requires_identity_fields(field: str):
    paper = {
        "paper_id": "icml-2026-001",
        "title": "Reliable Reproductions",
        "slug": "reliable-reproductions",
    }
    paper.pop(field)

    with pytest.raises(ValueError, match=field):
        state_module().select_paper(state_module().new_state(), paper)


@pytest.mark.parametrize("cost", [10.01, 11.0])
def test_select_paper_rejects_estimated_cost_above_limit(cost: float):
    paper = {
        "paper_id": "icml-2026-001",
        "title": "Reliable Reproductions",
        "slug": "reliable-reproductions",
        "estimated_api_cost_usd": cost,
    }

    with pytest.raises(ValueError, match="estimated_api_cost_usd"):
        state_module().select_paper(state_module().new_state(), paper)


def test_select_paper_accepts_estimated_cost_at_limit():
    paper = {
        "paper_id": "icml-2026-001",
        "title": "Reliable Reproductions",
        "slug": "reliable-reproductions",
        "estimated_api_cost_usd": 10.0,
    }

    selected = state_module().select_paper(state_module().new_state(), paper)

    assert selected["current"]["estimated_api_cost_usd"] == 10.0


def test_select_paper_rejects_completed_paper_id():
    module = state_module()
    state = module.new_state()
    paper = {
        "paper_id": "icml-2026-001",
        "title": "Reliable Reproductions",
        "slug": "reliable-reproductions",
    }
    state["history"].append(
        {
            **paper,
            "project_path": "submissions/reliable-reproductions",
        }
    )

    with pytest.raises(ValueError, match="paper_id"):
        module.select_paper(state, paper)


def test_reject_candidate_persists_immutable_record(tmp_path: Path):
    module = state_module()
    initial = module.new_state()
    candidate = rejection()

    rejected = module.reject_candidate(initial, candidate)
    candidate["reason"] = "changed after rejection"
    path = tmp_path / "repro-loop.json"
    module.save_state(path, rejected)

    assert initial == module.new_state()
    assert rejected["phase"] == "idle"
    assert rejected["rejections"] == [rejection()]
    assert module.load_state(path) == rejected


@pytest.mark.parametrize(
    "candidate",
    [
        {},
        {
            "paper_id": "icml-2026-002",
            "title": "Rejected Candidate",
            "reason": "No released artifacts",
            "checked_at": "",
        },
        {
            "paper_id": "icml-2026-002",
            "title": "Rejected Candidate",
            "reason": "No released artifacts",
            "checked_at": "2026-07-21T19:00:00Z",
            "extra": "unexpected",
        },
    ],
)
@pytest.mark.parametrize("operation", ["save", "load"])
def test_persisted_state_rejects_malformed_rejection_records(
    candidate: dict, operation: str, tmp_path: Path
):
    module = state_module()
    state = module.new_state()
    state["rejections"] = [candidate]
    path = tmp_path / "repro-loop.json"

    if operation == "save":
        with pytest.raises(ValueError):
            module.save_state(path, state)
    else:
        path.write_text(json.dumps(state), encoding="utf-8")
        with pytest.raises(ValueError):
            module.load_state(path)


@pytest.mark.parametrize("source", ["rejections", "history", "current"])
def test_validate_state_rejects_duplicate_rejected_paper_ids(source: str):
    module = state_module()
    state = module.new_state()
    if source == "rejections":
        state["rejections"] = [rejection(), rejection()]
    elif source == "history":
        state["rejections"] = [rejection()]
        state["history"] = [
            {
                **paper(),
                "paper_id": rejection()["paper_id"],
                "project_path": "submissions/reliable-reproductions",
            }
        ]
    else:
        state = module.select_paper(module.new_state(), paper())
        state["rejections"] = [
            {**rejection(), "paper_id": state["current"]["paper_id"]}
        ]

    with pytest.raises(ValueError, match="paper_id"):
        module.validate_state(state)


def test_reject_candidate_requires_idle_state():
    module = state_module()
    selected = module.select_paper(module.new_state(), paper())

    with pytest.raises(ValueError, match="phase"):
        module.reject_candidate(selected, rejection())


def test_select_paper_rejects_previously_rejected_candidate():
    module = state_module()
    state = module.reject_candidate(module.new_state(), rejection())
    candidate = {**paper(), "paper_id": rejection()["paper_id"]}

    with pytest.raises(ValueError, match="paper_id"):
        module.select_paper(state, candidate)


def test_transition_requires_design_approval_to_start_implementation():
    module = state_module()
    selected = module.select_paper(module.new_state(), paper())
    design_pending = module.transition(selected, "design-pending")

    with pytest.raises(ValueError, match="design_approved"):
        module.transition(design_pending, "implementing")


@pytest.mark.parametrize(
    ("target", "updates"),
    [
        ("deployed", {}),
        ("submitted", {}),
        ("complete", {}),
    ],
)
def test_transition_requires_phase_artifacts(target: str, updates: dict):
    module = state_module()
    state = state_in_phase(module, {"deployed": "validated", "submitted": "deployed", "complete": "judging"}[target])

    with pytest.raises(ValueError):
        module.transition(state, target, **updates)


def test_transition_rejects_actual_cost_above_limit():
    module = state_module()
    state = state_in_phase(module, "implementing")

    with pytest.raises(ValueError, match="actual_api_cost_usd"):
        module.transition(state, "validated", actual_api_cost_usd=10.01)


def test_transition_accepts_actual_cost_at_limit():
    module = state_module()
    state = state_in_phase(module, "implementing")

    validated = module.transition(state, "validated", actual_api_cost_usd=10.0)

    assert validated["current"]["actual_api_cost_usd"] == 10.0


def test_completion_is_recorded_and_cost_is_totaled_when_returning_to_idle():
    module = state_module()
    complete = state_in_phase(module, "judging")
    complete = module.transition(
        complete,
        "complete",
        verdict="accepted",
        actual_api_cost_usd=3.5,
    )

    idle = module.transition(complete, "idle")

    assert complete["phase"] == "complete"
    assert idle["phase"] == "idle"
    assert idle["current"] is None
    assert idle["total_api_cost_usd"] == 3.5
    assert idle["history"] == [complete["current"]]


ALLOWED = {
    "idle": {"selected"},
    "selected": {"design-pending", "blocked"},
    "design-pending": {"implementing", "blocked"},
    "implementing": {"validated", "blocked"},
    "validated": {"deployed", "blocked"},
    "deployed": {"submitted", "blocked"},
    "submitted": {"judging", "blocked"},
    "judging": {"improving", "complete", "blocked"},
    "improving": {"validated", "blocked"},
    "complete": {"idle"},
    "blocked": {"idle"},
}


@pytest.mark.parametrize(
    ("source", "target"),
    [
        (source, target)
        for source, targets in ALLOWED.items()
        for target in sorted(targets)
    ],
)
def test_transition_permits_only_the_declared_transitions(source: str, target: str):
    module = state_module()
    state = state_in_phase(module, source)

    if source == "idle" and target == "selected":
        transitioned = module.select_paper(state, paper())
    else:
        transitioned = module.transition(state, target, **updates_for(target))

    assert transitioned["phase"] == target


@pytest.mark.parametrize(
    ("source", "target"),
    [
        (source, target)
        for source, targets in ALLOWED.items()
        for target in ALLOWED
        if target not in targets
    ],
)
def test_transition_rejects_undeclared_transitions(source: str, target: str):
    module = state_module()

    with pytest.raises(ValueError, match="phase"):
        module.transition(state_in_phase(module, source), target, **updates_for(target))


def test_cli_initializes_shows_selects_and_transitions_state(tmp_path: Path):
    path = tmp_path / "repro-loop.json"
    paper_json = json.dumps(paper())

    run_cli("init", str(path))
    assert json.loads(run_cli("show", str(path)).stdout)["phase"] == "idle"
    run_cli("select", str(path), paper_json)
    run_cli("transition", str(path), "design-pending", "{}")

    assert json.loads(run_cli("show", str(path)).stdout)["phase"] == "design-pending"


def test_cli_rejects_candidate_without_changing_idle_phase(tmp_path: Path):
    path = tmp_path / "repro-loop.json"

    run_cli("init", str(path))
    rejected = json.loads(run_cli("reject", str(path), json.dumps(rejection())).stdout)

    assert rejected["phase"] == "idle"
    assert rejected["rejections"] == [rejection()]
    assert json.loads(run_cli("show", str(path)).stdout) == rejected


def test_cli_updates_current_without_changing_phase(tmp_path: Path):
    path = tmp_path / "repro-loop.json"
    run_cli("init", str(path))
    run_cli("select", str(path), json.dumps(paper()))

    updated = json.loads(
        run_cli(
            "update",
            str(path),
            json.dumps(
                {
                    "last_poll_at": "2026-07-21T18:00:00Z",
                    "last_poll_status": "pending",
                    "external_ids": {"submission": "submission-123"},
                }
            ),
        ).stdout
    )

    assert updated["phase"] == "selected"
    assert updated["current"]["last_poll_status"] == "pending"
    assert state_module().load_state(path) == updated


def test_cli_init_creates_parent_directory(tmp_path: Path):
    path = tmp_path / "state" / "repro-loop.json"

    run_cli("init", str(path))

    assert json.loads(run_cli("show", str(path)).stdout) == state_module().new_state()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("total_api_cost_usd", float("nan")),
        ("estimated_api_cost_usd", float("inf")),
        ("actual_api_cost_usd", float("-inf")),
    ],
)
def test_save_rejects_non_finite_costs_without_overwriting_existing_file(
    tmp_path: Path, field: str, value: float
):
    module = state_module()
    path = tmp_path / "repro-loop.json"
    state = module.select_paper(module.new_state(), paper())
    module.save_state(path, state)
    persisted = path.read_text(encoding="utf-8")

    if field == "total_api_cost_usd":
        state[field] = value
    else:
        state["current"][field] = value

    with pytest.raises(ValueError, match=field):
        module.save_state(path, state)

    assert path.read_text(encoding="utf-8") == persisted


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("total_api_cost_usd", float("nan")),
        ("estimated_api_cost_usd", float("inf")),
        ("actual_api_cost_usd", float("-inf")),
    ],
)
def test_load_rejects_non_finite_costs(field: str, value: float, tmp_path: Path):
    module = state_module()
    state = module.select_paper(module.new_state(), paper())
    if field == "total_api_cost_usd":
        state[field] = value
    else:
        state["current"][field] = value
    path = tmp_path / "repro-loop.json"
    path.write_text(json.dumps(state), encoding="utf-8")

    with pytest.raises(ValueError, match=field):
        module.load_state(path)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_transition_rejects_non_finite_actual_cost(value: float):
    module = state_module()

    with pytest.raises(ValueError, match="actual_api_cost_usd"):
        module.transition(
            state_in_phase(module, "implementing"),
            "validated",
            actual_api_cost_usd=value,
        )


def test_transition_rejects_estimated_cost_change_after_selection():
    module = state_module()

    with pytest.raises(ValueError, match="estimated_api_cost_usd"):
        module.transition(
            state_in_phase(module, "implementing"),
            "validated",
            estimated_api_cost_usd=4.0,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("polls", []),
        ("last_poll_at", "2026-07-21T18:00:00Z"),
        ("last_poll_status", "pending"),
        ("external_ids", {"submission": "submission-123"}),
    ],
)
def test_transition_rejects_operational_field_updates(field: str, value: object):
    module = state_module()

    with pytest.raises(ValueError, match=field):
        module.transition(
            state_in_phase(module, "selected"),
            "design-pending",
            **{field: value},
        )


def test_update_current_persists_poll_cost_and_external_ids():
    module = state_module()
    state = state_in_phase(module, "judging")
    external_ids = {"submission": "submission-123", "judge": "judge-456"}
    identity = {
        field: state["current"][field]
        for field in ("paper_id", "title", "slug", "project_path")
    }

    updated = module.update_current(
        state,
        actual_api_cost_usd=2.5,
        last_poll_at="2026-07-21T18:00:00Z",
        last_poll_status="pending",
        external_ids=external_ids,
    )
    external_ids["judge"] = "changed"

    assert updated["phase"] == "judging"
    assert {field: updated["current"][field] for field in identity} == identity
    assert updated["current"]["actual_api_cost_usd"] == 2.5
    assert updated["current"]["last_poll_at"] == "2026-07-21T18:00:00Z"
    assert updated["current"]["last_poll_status"] == "pending"
    assert updated["current"]["polls"] == [
        {"at": "2026-07-21T18:00:00Z", "status": "pending"}
    ]
    assert updated["current"]["external_ids"] == {
        "submission": "submission-123",
        "judge": "judge-456",
    }
    assert state["current"].get("actual_api_cost_usd") is None


def test_update_current_retains_two_polls_in_order():
    module = state_module()
    state = state_in_phase(module, "judging")

    first = module.update_current(
        state,
        last_poll_at="2026-07-21T18:00:00Z",
        last_poll_status="pending",
    )
    second = module.update_current(
        first,
        last_poll_at="2026-07-21T18:05:00Z",
        last_poll_status="accepted",
    )

    assert second["current"]["polls"] == [
        {"at": "2026-07-21T18:00:00Z", "status": "pending"},
        {"at": "2026-07-21T18:05:00Z", "status": "accepted"},
    ]
    assert second["current"]["last_poll_at"] == "2026-07-21T18:05:00Z"
    assert second["current"]["last_poll_status"] == "accepted"


def test_update_current_merges_external_ids_across_updates():
    module = state_module()
    state = state_in_phase(module, "judging")

    first = module.update_current(
        state, external_ids={"submission": "submission-123"}
    )
    second = module.update_current(first, external_ids={"judge": "judge-456"})

    assert second["current"]["external_ids"] == {
        "submission": "submission-123",
        "judge": "judge-456",
    }


def test_update_current_rejects_conflicting_external_id():
    module = state_module()
    state = module.update_current(
        state_in_phase(module, "judging"),
        external_ids={"submission": "submission-123"},
    )

    with pytest.raises(ValueError, match="external_ids"):
        module.update_current(
            state, external_ids={"submission": "different-submission"}
        )


@pytest.mark.parametrize("field", ["last_poll_at", "last_poll_status"])
def test_update_current_requires_both_poll_fields(field: str):
    module = state_module()
    updates = {
        "last_poll_at": "2026-07-21T18:00:00Z",
        "last_poll_status": "pending",
    }
    updates.pop(field)

    with pytest.raises(ValueError, match=field):
        module.update_current(state_in_phase(module, "judging"), **updates)


@pytest.mark.parametrize("value", [3.0, 10.01, float("nan"), "4.0"])
def test_update_current_rejects_invalid_actual_cost(value: object):
    module = state_module()
    state = module.update_current(
        state_in_phase(module, "judging"), actual_api_cost_usd=4.0
    )

    with pytest.raises(ValueError, match="actual_api_cost_usd"):
        module.update_current(state, actual_api_cost_usd=value)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("phase", "complete"),
        ("paper_id", "icml-2026-999"),
        ("title", "Changed Title"),
        ("slug", "changed-title"),
        ("project_path", "submissions/changed-title"),
        ("estimated_api_cost_usd", 1.0),
        ("design_approved", True),
        ("deployed_sha", "def456"),
        ("space_id", "org/other-reproduction"),
        ("verdict", "accepted"),
    ],
)
def test_update_current_rejects_non_persistence_fields(field: str, value: object):
    module = state_module()

    with pytest.raises(ValueError, match=field):
        module.update_current(state_in_phase(module, "judging"), **{field: value})


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("last_poll_at", 123),
        ("last_poll_status", None),
        ("external_ids", []),
        ("external_ids", {}),
        ("external_ids", {"": "submission-123"}),
        ("external_ids", {"submission": ""}),
        ("external_ids", {"submission": 123}),
    ],
)
def test_update_current_rejects_invalid_persistence_values(
    field: str, value: object
):
    module = state_module()
    updates = {field: value}
    if field == "last_poll_at":
        updates["last_poll_status"] = "pending"
    elif field == "last_poll_status":
        updates["last_poll_at"] = "2026-07-21T18:00:00Z"

    with pytest.raises(ValueError, match=field):
        module.update_current(state_in_phase(module, "judging"), **updates)


def test_update_current_rejects_idle_state():
    module = state_module()

    with pytest.raises(ValueError, match="current"):
        module.update_current(module.new_state(), last_poll_status="pending")


@pytest.mark.parametrize(
    "polls",
    [
        {},
        [{}],
        [{"at": "", "status": "pending"}],
        [{"at": "2026-07-21T18:00:00Z", "status": ""}],
        [{"at": "2026-07-21T18:00:00Z", "status": "pending", "extra": "x"}],
    ],
)
@pytest.mark.parametrize("operation", ["save", "load"])
def test_persisted_state_rejects_malformed_polls(
    polls: object, operation: str, tmp_path: Path
):
    module = state_module()
    state = module.select_paper(module.new_state(), paper())
    state["current"]["polls"] = polls
    path = tmp_path / "repro-loop.json"

    if operation == "save":
        with pytest.raises(ValueError, match="polls"):
            module.save_state(path, state)
    else:
        path.write_text(json.dumps(state), encoding="utf-8")
        with pytest.raises(ValueError, match="polls"):
            module.load_state(path)


@pytest.mark.parametrize(
    ("polls", "last_values", "field"),
    [
        (
            [],
            {
                "last_poll_at": "2026-07-21T18:00:00Z",
                "last_poll_status": "pending",
            },
            "last_poll_at",
        ),
        (
            [{"at": "2026-07-21T18:00:00Z", "status": "pending"}],
            {
                "last_poll_at": "2026-07-21T18:05:00Z",
                "last_poll_status": "pending",
            },
            "last_poll_at",
        ),
        (
            [{"at": "2026-07-21T18:00:00Z", "status": "pending"}],
            {
                "last_poll_at": "2026-07-21T18:00:00Z",
                "last_poll_status": "accepted",
            },
            "last_poll_status",
        ),
        (
            [{"at": "2026-07-21T18:00:00Z", "status": "pending"}],
            {"last_poll_at": "2026-07-21T18:00:00Z"},
            "last_poll_status",
        ),
    ],
)
@pytest.mark.parametrize("operation", ["save", "load"])
def test_persisted_state_rejects_last_poll_mismatch(
    polls: list[dict[str, str]],
    last_values: dict[str, str],
    field: str,
    operation: str,
    tmp_path: Path,
):
    module = state_module()
    state = module.select_paper(module.new_state(), paper())
    state["current"]["polls"] = polls
    state["current"].update(last_values)
    path = tmp_path / "repro-loop.json"

    if operation == "save":
        with pytest.raises(ValueError, match=field):
            module.save_state(path, state)
    else:
        path.write_text(json.dumps(state), encoding="utf-8")
        with pytest.raises(ValueError, match=field):
            module.load_state(path)


def test_persisted_state_accepts_polls_without_last_values(tmp_path: Path):
    module = state_module()
    state = module.select_paper(module.new_state(), paper())
    state["current"]["polls"] = [
        {"at": "2026-07-21T18:00:00Z", "status": "pending"}
    ]
    path = tmp_path / "repro-loop.json"

    module.save_state(path, state)

    assert module.load_state(path) == state


@pytest.mark.parametrize(
    "external_ids",
    [
        [],
        {},
        {"": "submission-123"},
        {"submission": ""},
        {"submission": 123},
    ],
)
@pytest.mark.parametrize("operation", ["save", "load"])
def test_persisted_state_rejects_malformed_external_ids(
    external_ids: object, operation: str, tmp_path: Path
):
    module = state_module()
    state = module.select_paper(module.new_state(), paper())
    state["current"]["external_ids"] = external_ids
    path = tmp_path / "repro-loop.json"

    if operation == "save":
        with pytest.raises(ValueError, match="external_ids"):
            module.save_state(path, state)
    else:
        path.write_text(json.dumps(state), encoding="utf-8")
        with pytest.raises(ValueError, match="external_ids"):
            module.load_state(path)


@pytest.mark.parametrize(
    ("source", "target", "field", "value"),
    [
        ("design-pending", "implementing", "design_approved", True),
        ("validated", "deployed", "deployed_sha", "abc123"),
        ("deployed", "submitted", "space_id", "org/reproduction"),
        ("judging", "complete", "verdict", "accepted"),
    ],
)
def test_transition_requires_fresh_phase_artifact_update(
    source: str, target: str, field: str, value: object
):
    module = state_module()
    state = state_in_phase(module, source)
    state["current"][field] = value

    with pytest.raises(ValueError, match=field):
        module.transition(state, target)


def test_blocked_attempt_is_archived_and_cannot_be_reselected():
    module = state_module()
    selected = module.select_paper(module.new_state(), paper())
    blocked = module.transition(selected, "blocked", actual_api_cost_usd=2.5)

    idle = module.transition(blocked, "idle")

    assert idle["current"] is None
    assert idle["history"] == [blocked["current"]]
    assert idle["total_api_cost_usd"] == 2.5
    with pytest.raises(ValueError, match="paper_id"):
        module.select_paper(idle, paper())


def test_cli_init_refuses_to_overwrite_existing_state(tmp_path: Path):
    path = tmp_path / "repro-loop.json"
    run_cli("init", str(path))
    persisted = path.read_text(encoding="utf-8")

    with pytest.raises(subprocess.CalledProcessError):
        run_cli("init", str(path))

    assert path.read_text(encoding="utf-8") == persisted


def test_select_derives_project_path_and_rejects_historical_project_path():
    module = state_module()
    state = module.new_state()
    state["history"].append(
        {
            "paper_id": "icml-2026-000",
            "title": "Prior Attempt",
            "slug": "reliable-reproductions",
            "project_path": "submissions/reliable-reproductions",
        }
    )

    with pytest.raises(ValueError, match="project_path"):
        module.select_paper(state, paper())


def test_transition_rejects_historical_submission_space_id():
    module = state_module()
    state = state_in_phase(module, "deployed")
    state["history"].append(
        {
            "paper_id": "icml-2026-000",
            "title": "Prior Attempt",
            "slug": "prior-attempt",
            "project_path": "submissions/prior-attempt",
            "space_id": "org/reproduction",
        }
    )

    with pytest.raises(ValueError, match="space_id"):
        module.transition(state, "submitted", space_id="org/reproduction")


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("paper_id", "icml-2026-999"),
        ("title", "Changed Title"),
        ("slug", "changed-title"),
        ("project_path", "submissions/changed-title"),
    ],
)
def test_transition_rejects_immutable_paper_field_changes(field: str, value: str):
    module = state_module()

    with pytest.raises(ValueError, match=field):
        module.transition(
            state_in_phase(module, "selected"),
            "design-pending",
            **{field: value},
        )


def test_transition_rejects_decreasing_actual_api_cost():
    module = state_module()
    validated = module.transition(
        state_in_phase(module, "implementing"),
        "validated",
        actual_api_cost_usd=4.0,
    )

    with pytest.raises(ValueError, match="actual_api_cost_usd"):
        module.transition(
            validated,
            "deployed",
            deployed_sha="abc123",
            actual_api_cost_usd=3.0,
        )


def test_transition_rejects_space_id_change_after_submission():
    module = state_module()

    with pytest.raises(ValueError, match="space_id"):
        module.transition(
            state_in_phase(module, "submitted"),
            "judging",
            space_id="org/other-reproduction",
        )


@pytest.mark.parametrize("field", ["project_path", "space_id"])
@pytest.mark.parametrize("operation", ["save", "load"])
def test_persisted_state_rejects_duplicate_project_paths_and_space_ids(
    field: str, operation: str, tmp_path: Path
):
    module = state_module()
    state = module.select_paper(module.new_state(), paper())
    state["current"]["space_id"] = "org/reproduction"
    state["history"].append(
        {
            "paper_id": "icml-2026-000",
            "title": "Prior Attempt",
            "slug": "prior-attempt",
            "project_path": "submissions/prior-attempt",
            "space_id": "org/prior-reproduction",
        }
    )
    if field == "project_path":
        state["history"][0][field] = state["current"][field]
        state["history"][0]["slug"] = state["current"]["slug"]
    else:
        state["history"][0][field] = state["current"][field]
    path = tmp_path / "repro-loop.json"

    if operation == "save":
        with pytest.raises(ValueError, match=field):
            module.save_state(path, state)
    else:
        path.write_text(json.dumps(state), encoding="utf-8")
        with pytest.raises(ValueError, match=field):
            module.load_state(path)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("slug", "../outside"),
        ("slug", "Uppercase-Slug"),
        ("project_path", "submissions/../outside"),
        ("project_path", "submissions/reliable-reproductions/"),
    ],
)
@pytest.mark.parametrize("operation", ["save", "load"])
def test_persisted_state_rejects_invalid_slug_and_project_path(
    field: str, value: str, operation: str, tmp_path: Path
):
    module = state_module()
    state = module.select_paper(module.new_state(), paper())
    state["current"][field] = value
    path = tmp_path / "repro-loop.json"

    if operation == "save":
        with pytest.raises(ValueError, match=field):
            module.save_state(path, state)
    else:
        path.write_text(json.dumps(state), encoding="utf-8")
        with pytest.raises(ValueError, match=field):
            module.load_state(path)


def test_transition_rejects_project_path_alias():
    module = state_module()

    with pytest.raises(ValueError, match="project_path"):
        module.transition(
            state_in_phase(module, "selected"),
            "design-pending",
            project_path="submissions/../outside",
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("version", True),
        ("phase", ["selected"]),
    ],
)
def test_save_rejects_invalid_top_level_types_with_field_name(
    field: str, value: object, tmp_path: Path
):
    module = state_module()
    state = module.new_state()
    state[field] = value

    with pytest.raises(ValueError, match=field):
        module.save_state(tmp_path / "repro-loop.json", state)


def paper() -> dict:
    return {
        "paper_id": "icml-2026-001",
        "title": "Reliable Reproductions",
        "slug": "reliable-reproductions",
    }


def rejection() -> dict:
    return {
        "paper_id": "icml-2026-002",
        "title": "Rejected Candidate",
        "reason": "No released artifacts",
        "checked_at": "2026-07-21T19:00:00Z",
    }


def state_in_phase(module, phase: str) -> dict:
    if phase == "idle":
        return module.new_state()
    if phase == "blocked":
        return module.transition(
            module.select_paper(module.new_state(), paper()), "blocked"
        )

    state = module.select_paper(module.new_state(), paper())
    if phase == "complete":
        state = module.transition(state, "design-pending")
        state = module.transition(state, "implementing", design_approved=True)
        state = module.transition(state, "validated")
        state = module.transition(state, "deployed", deployed_sha="abc123")
        state = module.transition(state, "submitted", space_id="org/reproduction")
        state = module.transition(state, "judging")
        return module.transition(state, "complete", verdict="accepted")
    if phase == "selected":
        return state
    for target in (
        "design-pending",
        "implementing",
        "validated",
        "deployed",
        "submitted",
        "judging",
        "improving",
        "complete",
    ):
        state = module.transition(state, target, **updates_for(target))
        if target == phase:
            return state
    raise AssertionError(f"Unsupported phase: {phase}")


def updates_for(phase: str) -> dict:
    return {
        "implementing": {"design_approved": True},
        "deployed": {"deployed_sha": "abc123"},
        "submitted": {"space_id": "org/reproduction"},
        "complete": {"verdict": "accepted"},
    }.get(phase, {})


def run_cli(*arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(STATE_MODULE_PATH), *arguments],
        check=True,
        capture_output=True,
        text=True,
    )
