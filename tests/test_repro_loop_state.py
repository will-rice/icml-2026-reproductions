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
        "version": 3,
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
        ({"version": 3}, "keys"),
        (
            {
                "version": 1,
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
                "version": 3,
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
                "version": 3,
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
                "version": 3,
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
                "version": 3,
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
        "upstream_revision": "abc123",
        "target_claims": ["claim-1", "claim-2"],
    }

    selected = module.select_paper(initial, paper)

    assert initial == module.new_state()
    assert selected["phase"] == "selected"
    assert selected["current"] == {
        **paper,
        "project_path": "submissions/reliable-reproductions",
        "polls": [],
        "improvement_attempts": 0,
        "verdicts": [],
    }


@pytest.mark.parametrize("field", ["paper_id", "title", "slug"])
def test_select_paper_requires_identity_fields(field: str):
    paper = {
        "paper_id": "icml-2026-001",
        "title": "Reliable Reproductions",
        "slug": "reliable-reproductions",
        "estimated_api_cost_usd": 4.25,
        "upstream_revision": "abc123",
        "target_claims": ["claim-1", "claim-2"],
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
        "upstream_revision": "abc123",
        "target_claims": ["claim-1", "claim-2"],
    }

    with pytest.raises(ValueError, match="estimated_api_cost_usd"):
        state_module().select_paper(state_module().new_state(), paper)


def test_select_paper_accepts_estimated_cost_at_limit():
    paper = {
        "paper_id": "icml-2026-001",
        "title": "Reliable Reproductions",
        "slug": "reliable-reproductions",
        "estimated_api_cost_usd": 10.0,
        "upstream_revision": "abc123",
        "target_claims": ["claim-1", "claim-2"],
    }

    selected = state_module().select_paper(state_module().new_state(), paper)

    assert selected["current"]["estimated_api_cost_usd"] == 10.0


def test_select_paper_rejects_completed_paper_id():
    module = state_module()
    state = module.new_state()
    candidate = paper()
    state["history"].append(persisted_paper(**candidate))

    with pytest.raises(ValueError, match="paper_id"):
        module.select_paper(state, candidate)


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
        verdict=verdict(),
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
    "blocked": {"idle", "selected"},
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
    elif source == "blocked" and target == "idle":
        transitioned = module.transition(state, target, abandon=True)
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
                    "external_ids": {"submission": "submission-123"},
                }
            ),
        ).stdout
    )

    assert updated["phase"] == "selected"
    assert updated["current"]["external_ids"] == {
        "submission": "submission-123"
    }
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


def test_abandoned_blocked_attempt_is_archived_and_cannot_be_reselected():
    module = state_module()
    selected = module.select_paper(module.new_state(), paper())
    blocked = module.transition(
        selected,
        "blocked",
        blocker="upstream unavailable",
        actual_api_cost_usd=2.5,
    )

    idle = module.transition(blocked, "idle", abandon=True)

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
        persisted_paper(
            paper_id="icml-2026-000",
            title="Prior Attempt",
        )
    )

    with pytest.raises(ValueError, match="project_path"):
        module.select_paper(state, paper())


def test_transition_rejects_historical_submission_space_id():
    module = state_module()
    state = state_in_phase(module, "deployed")
    state["history"].append(
        persisted_paper(
            paper_id="icml-2026-000",
            title="Prior Attempt",
            slug="prior-attempt",
            project_path="submissions/prior-attempt",
            space_id="org/reproduction",
        )
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
        persisted_paper(
            paper_id="icml-2026-000",
            title="Prior Attempt",
            slug="prior-attempt",
            project_path="submissions/prior-attempt",
            space_id="org/prior-reproduction",
        )
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


@pytest.mark.parametrize("field", ["estimated_api_cost_usd", "upstream_revision"])
def test_select_paper_requires_cost_and_upstream_revision(field: str):
    candidate = {
        **paper(),
        "estimated_api_cost_usd": 0.0,
        "upstream_revision": "abc123",
    }
    candidate.pop(field)

    with pytest.raises(ValueError, match=field):
        state_module().select_paper(state_module().new_state(), candidate)


@pytest.mark.parametrize("field", ["estimated_api_cost_usd", "upstream_revision"])
@pytest.mark.parametrize("operation", ["save", "load"])
def test_persisted_selection_requires_cost_and_upstream_revision(
    field: str, operation: str, tmp_path: Path
):
    module = state_module()
    state = module.select_paper(module.new_state(), paper())
    state["current"].pop(field)
    path = tmp_path / "repro-loop.json"

    if operation == "save":
        with pytest.raises(ValueError, match=field):
            module.save_state(path, state)
    else:
        path.write_text(json.dumps(state), encoding="utf-8")
        with pytest.raises(ValueError, match=field):
            module.load_state(path)


def test_select_paper_initializes_improvement_attempts():
    selected = state_module().select_paper(
        state_module().new_state(),
        {
            **paper(),
            "estimated_api_cost_usd": 0.0,
            "upstream_revision": "abc123",
        },
    )

    assert selected["current"]["improvement_attempts"] == 0


def test_blocked_transition_records_origin_and_resumes_without_archiving():
    module = state_module()
    selected = module.select_paper(module.new_state(), paper())

    blocked = module.transition(selected, "blocked", blocker="credentials missing")
    resumed = module.transition(blocked, "selected")

    assert blocked["current"]["blocked_from"] == "selected"
    assert blocked["current"]["blocker"] == "credentials missing"
    assert resumed["phase"] == "selected"
    assert resumed["history"] == []
    assert resumed["total_api_cost_usd"] == 0.0
    assert "blocked_from" not in resumed["current"]
    assert "blocker" not in resumed["current"]


@pytest.mark.parametrize(
    "source_phase",
    [
        "selected",
        "design-pending",
        "implementing",
        "validated",
        "deployed",
        "submitted",
        "judging",
        "improving",
    ],
)
def test_blocked_transition_resumes_every_origin_without_fresh_artifacts(
    source_phase: str,
):
    module = state_module()
    source = state_in_phase(module, source_phase)
    blocked = module.transition(source, "blocked", blocker="external outage")

    resumed = module.transition(blocked, source_phase)

    assert resumed["phase"] == source_phase
    assert resumed["history"] == []
    assert "blocked_from" not in resumed["current"]
    assert "blocker" not in resumed["current"]


@pytest.mark.parametrize("blocker", [None, "", 1])
def test_blocked_transition_requires_nonempty_blocker(blocker: object):
    module = state_module()

    with pytest.raises(ValueError, match="blocker"):
        module.transition(
            module.select_paper(module.new_state(), paper()),
            "blocked",
            blocker=blocker,
        )


def test_blocked_transition_requires_explicit_abandon_before_archiving():
    module = state_module()
    blocked = module.transition(
        module.select_paper(module.new_state(), paper()),
        "blocked",
        blocker="upstream unavailable",
        actual_api_cost_usd=2.5,
    )

    with pytest.raises(ValueError, match="abandon"):
        module.transition(blocked, "idle")

    idle = module.transition(blocked, "idle", abandon=True)
    assert idle["current"] is None
    assert idle["history"] == [blocked["current"]]
    assert idle["total_api_cost_usd"] == 2.5


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("blocker", ""),
        ("blocked_from", "idle"),
        ("blocked_from", "unknown"),
    ],
)
@pytest.mark.parametrize("operation", ["save", "load"])
def test_persisted_blocked_state_validates_recovery_fields(
    field: str, value: object, operation: str, tmp_path: Path
):
    module = state_module()
    state = module.transition(
        module.select_paper(module.new_state(), paper()),
        "blocked",
        blocker="credentials missing",
    )
    state["current"][field] = value
    path = tmp_path / "repro-loop.json"

    if operation == "save":
        with pytest.raises(ValueError, match=field):
            module.save_state(path, state)
    else:
        path.write_text(json.dumps(state), encoding="utf-8")
        with pytest.raises(ValueError, match=field):
            module.load_state(path)


@pytest.mark.parametrize(
    ("poll_limit", "poll_deadline", "field"),
    [
        (0, "2026-07-22T18:00:00Z", "poll_limit"),
        (-1, "2026-07-22T18:00:00Z", "poll_limit"),
        (1.5, "2026-07-22T18:00:00Z", "poll_limit"),
        (True, "2026-07-22T18:00:00Z", "poll_limit"),
        (2, "2026-07-22T18:00:00", "poll_deadline"),
        (2, "not-a-date", "poll_deadline"),
    ],
)
def test_entering_judging_requires_bounded_poll_configuration(
    poll_limit: object, poll_deadline: object, field: str
):
    module = state_module()

    with pytest.raises(ValueError, match=field):
        module.transition(
            state_in_phase(module, "submitted"),
            "judging",
            poll_limit=poll_limit,
            poll_deadline=poll_deadline,
        )


@pytest.mark.parametrize("field", ["poll_limit", "poll_deadline"])
def test_entering_judging_requires_both_poll_fields(field: str):
    module = state_module()
    updates = {
        "poll_limit": 2,
        "poll_deadline": "2026-07-22T18:00:00Z",
    }
    updates.pop(field)

    with pytest.raises(ValueError, match=field):
        module.transition(state_in_phase(module, "submitted"), "judging", **updates)


@pytest.mark.parametrize("field", ["poll_limit", "poll_deadline"])
@pytest.mark.parametrize("operation", ["save", "load"])
def test_persisted_judging_state_requires_both_poll_fields(
    field: str, operation: str, tmp_path: Path
):
    module = state_module()
    state = valid_judging_state(module)
    state["current"].pop(field)
    path = tmp_path / "repro-loop.json"

    if operation == "save":
        with pytest.raises(ValueError, match=field):
            module.save_state(path, state)
    else:
        path.write_text(json.dumps(state), encoding="utf-8")
        with pytest.raises(ValueError, match=field):
            module.load_state(path)


def test_judging_polls_append_through_limit_and_deadline():
    module = state_module()
    judging = valid_judging_state(module)

    first = module.update_current(
        judging,
        last_poll_at="2026-07-22T17:00:00Z",
        last_poll_status="pending",
    )
    second = module.update_current(
        first,
        last_poll_at="2026-07-22T18:00:00Z",
        last_poll_status="pending",
    )

    assert len(second["current"]["polls"]) == 2
    with pytest.raises(ValueError, match="poll_limit"):
        module.update_current(
            second,
            last_poll_at="2026-07-22T18:00:00Z",
            last_poll_status="pending",
        )
    with pytest.raises(ValueError, match="poll_deadline"):
        module.update_current(
            first,
            last_poll_at="2026-07-22T18:00:01Z",
            last_poll_status="pending",
        )


def test_judging_poll_timestamp_must_be_timezone_aware():
    module = state_module()

    with pytest.raises(ValueError, match="last_poll_at"):
        module.update_current(
            valid_judging_state(module),
            last_poll_at="2026-07-22T17:00:00",
            last_poll_status="pending",
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("poll_limit", 0),
        ("poll_limit", 1.5),
        ("poll_deadline", "2026-07-22T18:00:00"),
        ("poll_deadline", "invalid"),
    ],
)
@pytest.mark.parametrize("operation", ["save", "load"])
def test_persisted_judging_state_validates_poll_configuration(
    field: str, value: object, operation: str, tmp_path: Path
):
    module = state_module()
    state = valid_judging_state(module)
    state["current"][field] = value
    path = tmp_path / "repro-loop.json"

    if operation == "save":
        with pytest.raises(ValueError, match=field):
            module.save_state(path, state)
    else:
        path.write_text(json.dumps(state), encoding="utf-8")
        with pytest.raises(ValueError, match=field):
            module.load_state(path)


@pytest.mark.parametrize(
    "invalid_verdict",
    [
        "accepted",
        {},
        {"claims": []},
        {"claims": [{"claim": "claim-1", "status": "verified", "extra": True}]},
        {"claims": [{"claim": "", "status": "verified"}]},
        {"claims": [{"claim": "claim-1", "status": "accepted"}]},
        {"claims": [{"claim": "claim-1", "status": []}]},
    ],
)
def test_completion_requires_exact_claim_level_verdict(invalid_verdict: object):
    module = state_module()

    with pytest.raises(ValueError, match="verdict"):
        module.transition(
            valid_judging_state(module),
            "complete",
            verdict=invalid_verdict,
        )


@pytest.mark.parametrize(
    "status",
    ["verified", "partial", "inconclusive", "contradicted", "unavailable"],
)
def test_completion_accepts_supported_claim_statuses(status: str):
    module = state_module()

    complete = module.transition(
        valid_judging_state(module),
        "complete",
        verdict=verdict(status),
    )

    assert complete["current"]["verdict"]["claims"][0]["status"] == status


@pytest.mark.parametrize("operation", ["save", "load"])
def test_persisted_complete_state_validates_verdict(
    operation: str, tmp_path: Path
):
    module = state_module()
    state = module.transition(
        valid_judging_state(module), "complete", verdict=verdict()
    )
    state["current"]["verdict"] = "accepted"
    path = tmp_path / "repro-loop.json"

    if operation == "save":
        with pytest.raises(ValueError, match="verdict"):
            module.save_state(path, state)
    else:
        path.write_text(json.dumps(state), encoding="utf-8")
        with pytest.raises(ValueError, match="verdict"):
            module.load_state(path)


@pytest.mark.parametrize("reason", [None, "", 1])
def test_improvement_requires_nonempty_reason(reason: object):
    module = state_module()

    with pytest.raises(ValueError, match="improvement_reason"):
        module.transition(
            valid_judging_state(module),
            "improving",
            improvement_reason=reason,
        )


def test_improvement_counts_one_attempt_and_rejects_a_second():
    module = state_module()
    improving = module.transition(
        valid_judging_state(module),
        "improving",
        verdict=verdict("partial"),
        improvement_reason="Add missing claim provenance",
    )

    assert improving["current"]["improvement_attempts"] == 1
    validated = module.transition(improving, "validated")
    deployed = module.transition(validated, "deployed", deployed_sha="def456")
    assert deployed["current"]["deployed_sha"] == "def456"
    submitted = module.transition(
        deployed, "submitted", space_id="org/reproduction"
    )
    judging = module.transition(
        submitted,
        "judging",
        poll_limit=2,
        poll_deadline="2026-07-23T18:00:00Z",
    )

    with pytest.raises(ValueError, match="improvement_attempts"):
        module.transition(
            judging,
            "improving",
            improvement_reason="Try again",
        )


def test_improvement_attempt_count_cannot_be_spoofed():
    module = state_module()
    improving = module.transition(
        valid_judging_state(module),
        "improving",
        verdict=verdict("partial"),
        improvement_reason="Add missing claim provenance",
    )
    judging = module.transition(
        module.transition(
            module.transition(
                module.transition(improving, "validated"),
                "deployed",
                deployed_sha="abc123",
            ),
            "submitted",
            space_id="org/reproduction",
        ),
        "judging",
        poll_limit=2,
        poll_deadline="2026-07-23T18:00:00Z",
    )

    with pytest.raises(ValueError, match="improvement_attempts"):
        module.transition(
            judging,
            "improving",
            improvement_reason="Try again",
            improvement_attempts=0,
        )


@pytest.mark.parametrize("operation", ["save", "load"])
def test_persisted_improving_state_requires_counted_attempt(
    operation: str, tmp_path: Path
):
    module = state_module()
    state = module.transition(
        valid_judging_state(module),
        "improving",
        verdict=verdict("partial"),
        improvement_reason="Add missing claim provenance",
    )
    state["current"]["improvement_attempts"] = 0
    state["current"].pop("improvement_reason")
    path = tmp_path / "repro-loop.json"

    if operation == "save":
        with pytest.raises(ValueError, match="improvement_attempts"):
            module.save_state(path, state)
    else:
        path.write_text(json.dumps(state), encoding="utf-8")
        with pytest.raises(ValueError, match="improvement_attempts"):
            module.load_state(path)


@pytest.mark.parametrize("value", [-1, 2, 1.5, True])
@pytest.mark.parametrize("operation", ["save", "load"])
def test_persisted_state_validates_improvement_attempts(
    value: object, operation: str, tmp_path: Path
):
    module = state_module()
    state = module.select_paper(module.new_state(), paper())
    state["current"]["improvement_attempts"] = value
    path = tmp_path / "repro-loop.json"

    if operation == "save":
        with pytest.raises(ValueError, match="improvement_attempts"):
            module.save_state(path, state)
    else:
        path.write_text(json.dumps(state), encoding="utf-8")
        with pytest.raises(ValueError, match="improvement_attempts"):
            module.load_state(path)


def test_cli_help_names_required_transition_metadata():
    help_text = run_cli("--help").stdout

    assert "upstream_revision" in help_text
    assert "target_claims" in help_text
    assert "poll_limit" in help_text
    assert "blocker" in help_text
    assert "abandon" in help_text
    assert "improvement_reason" in help_text
    assert "claim-level verdicts" in help_text


@pytest.mark.parametrize(
    "target_claims",
    [None, [], ["claim-1"], ["claim-1", "claim-1"], ["claim-1", ""], "claims"],
)
def test_selection_requires_two_unique_nonempty_target_claims(
    target_claims: object,
):
    module = state_module()
    candidate = paper()
    if target_claims is None:
        candidate.pop("target_claims")
    else:
        candidate["target_claims"] = target_claims

    with pytest.raises(ValueError, match="target_claims"):
        module.select_paper(module.new_state(), candidate)


def test_selection_initializes_verdict_history():
    selected = state_module().select_paper(state_module().new_state(), paper())

    assert selected["current"]["verdicts"] == []


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("upstream_revision", "def456"),
        ("target_claims", ["claim-1", "claim-3"]),
    ],
)
def test_transition_rejects_provenance_and_claim_mutation(
    field: str, value: object
):
    module = state_module()

    with pytest.raises(ValueError, match=field):
        module.transition(
            state_in_phase(module, "selected"),
            "design-pending",
            **{field: value},
        )


def test_transition_rejects_design_approval_mutation_once_set():
    module = state_module()

    with pytest.raises(ValueError, match="design_approved"):
        module.transition(
            state_in_phase(module, "implementing"),
            "validated",
            design_approved=False,
        )


def test_transition_rejects_deployed_sha_mutation_once_set():
    module = state_module()

    with pytest.raises(ValueError, match="deployed_sha"):
        module.transition(
            state_in_phase(module, "deployed"),
            "submitted",
            deployed_sha="def456",
            space_id="org/reproduction",
        )


@pytest.mark.parametrize(
    ("phase", "field", "value"),
    [
        ("implementing", "design_approved", False),
        ("deployed", "deployed_sha", None),
        ("submitted", "space_id", None),
        ("judging", "poll_round_start", None),
    ],
)
@pytest.mark.parametrize("operation", ["save", "load"])
def test_persisted_phase_prerequisites_are_enforced(
    phase: str, field: str, value: object, operation: str, tmp_path: Path
):
    module = state_module()
    state = state_in_phase(module, phase)
    if value is None:
        state["current"].pop(field, None)
    else:
        state["current"][field] = value
    path = tmp_path / "repro-loop.json"

    if operation == "save":
        with pytest.raises(ValueError, match=field):
            module.save_state(path, state)
    else:
        path.write_text(json.dumps(state), encoding="utf-8")
        with pytest.raises(ValueError, match=field):
            module.load_state(path)


@pytest.mark.parametrize(
    ("source_phase", "field"),
    [
        ("implementing", "design_approved"),
        ("deployed", "deployed_sha"),
        ("submitted", "space_id"),
        ("judging", "poll_round_start"),
    ],
)
def test_persisted_blocked_state_validates_origin_prerequisites(
    source_phase: str, field: str, tmp_path: Path
):
    module = state_module()
    state = module.transition(
        state_in_phase(module, source_phase), "blocked", blocker="external outage"
    )
    state["current"].pop(field, None)

    with pytest.raises(ValueError, match=field):
        module.save_state(tmp_path / "repro-loop.json", state)


@pytest.mark.parametrize(
    "claims",
    [
        [{"claim": "claim-1", "status": "verified"}],
        [
            {"claim": "claim-1", "status": "verified"},
            {"claim": "claim-1", "status": "partial"},
        ],
        [
            {"claim": "claim-1", "status": "verified"},
            {"claim": "claim-3", "status": "partial"},
        ],
    ],
)
def test_verdict_claim_names_must_exactly_match_targets(claims: list[dict]):
    module = state_module()

    with pytest.raises(ValueError, match="verdict"):
        module.transition(
            valid_judging_state(module),
            "complete",
            verdict={"claims": claims},
        )


def test_improvement_requires_verdict_and_appends_authoritative_history():
    module = state_module()

    with pytest.raises(ValueError, match="verdict"):
        module.transition(
            valid_judging_state(module),
            "improving",
            improvement_reason="Add missing provenance",
        )

    improving = module.transition(
        valid_judging_state(module),
        "improving",
        verdict=verdict("partial"),
        improvement_reason="Add missing provenance",
    )

    assert improving["current"]["verdicts"] == [
        {
            **verdict("partial"),
            "improvement_attempt": 1,
            "improvement_reason": "Add missing provenance",
        }
    ]
    assert "verdict" not in improving["current"]


def test_completion_appends_final_verdict_and_keeps_consistent_api_value():
    module = state_module()
    complete = module.transition(
        valid_judging_state(module), "complete", verdict=verdict()
    )

    assert complete["current"]["verdicts"] == [
        {**verdict(), "improvement_attempt": 0}
    ]
    assert complete["current"]["verdict"] == verdict()


def test_improved_completion_preserves_both_authoritative_verdicts():
    module = state_module()
    improving = module.transition(
        valid_judging_state(module),
        "improving",
        verdict=verdict("partial"),
        improvement_reason="Add missing provenance",
    )
    validated = module.transition(improving, "validated")
    deployed = module.transition(validated, "deployed", deployed_sha="abc123")
    submitted = module.transition(
        deployed, "submitted", space_id="org/reproduction"
    )
    judging = module.transition(
        submitted,
        "judging",
        poll_limit=1,
        poll_deadline="2026-07-23T18:00:00Z",
    )
    complete = module.transition(judging, "complete", verdict=verdict())

    assert complete["current"]["verdicts"] == [
        {
            **verdict("partial"),
            "improvement_attempt": 1,
            "improvement_reason": "Add missing provenance",
        },
        {**verdict(), "improvement_attempt": 1},
    ]
    assert complete["current"]["verdict"] == verdict()


@pytest.mark.parametrize("operation", ["save", "load"])
def test_persisted_final_verdict_must_match_authoritative_history(
    operation: str, tmp_path: Path
):
    module = state_module()
    state = module.transition(
        valid_judging_state(module), "complete", verdict=verdict()
    )
    state["current"]["verdict"] = verdict("partial")
    path = tmp_path / "repro-loop.json"

    if operation == "save":
        with pytest.raises(ValueError, match="verdict"):
            module.save_state(path, state)
    else:
        path.write_text(json.dumps(state), encoding="utf-8")
        with pytest.raises(ValueError, match="verdict"):
            module.load_state(path)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("improvement_attempt", 0),
        ("improvement_reason", None),
        ("improvement_reason", "Different reason"),
    ],
)
def test_persisted_improvement_verdict_validates_attempt_metadata(
    field: str, value: object, tmp_path: Path
):
    module = state_module()
    state = module.transition(
        valid_judging_state(module),
        "improving",
        verdict=verdict("partial"),
        improvement_reason="Add missing provenance",
    )
    if value is None:
        state["current"]["verdicts"][0].pop(field)
    else:
        state["current"]["verdicts"][0][field] = value

    with pytest.raises(ValueError, match="verdicts"):
        module.save_state(tmp_path / "repro-loop.json", state)


def test_persisted_final_verdict_rejects_improvement_reason_metadata(
    tmp_path: Path,
):
    module = state_module()
    improving = module.transition(
        valid_judging_state(module),
        "improving",
        verdict=verdict("partial"),
        improvement_reason="Add missing provenance",
    )
    validated = module.transition(improving, "validated")
    deployed = module.transition(validated, "deployed", deployed_sha="abc123")
    submitted = module.transition(
        deployed, "submitted", space_id="org/reproduction"
    )
    judging = module.transition(
        submitted,
        "judging",
        poll_limit=1,
        poll_deadline="2026-07-23T18:00:00Z",
    )
    state = module.transition(judging, "complete", verdict=verdict())
    state["current"]["verdicts"][-1]["improvement_reason"] = (
        "Add missing provenance"
    )

    with pytest.raises(ValueError, match="verdicts"):
        module.save_state(tmp_path / "repro-loop.json", state)


def test_poll_limits_are_scoped_to_each_judging_round():
    module = state_module()
    judging = valid_judging_state(module)
    first = module.update_current(
        judging,
        last_poll_at="2026-07-22T17:00:00Z",
        last_poll_status="pending",
    )
    second = module.update_current(
        first,
        last_poll_at="2026-07-22T18:00:00Z",
        last_poll_status="partial",
    )
    improving = module.transition(
        second,
        "improving",
        verdict=verdict("partial"),
        improvement_reason="Add missing provenance",
    )
    validated = module.transition(improving, "validated")
    deployed = module.transition(validated, "deployed", deployed_sha="abc123")
    submitted = module.transition(
        deployed, "submitted", space_id="org/reproduction"
    )
    second_round = module.transition(
        submitted,
        "judging",
        poll_limit=1,
        poll_deadline="2026-07-23T18:00:00Z",
    )

    assert second_round["current"]["poll_round_start"] == 2
    final_poll = module.update_current(
        second_round,
        last_poll_at="2026-07-23T18:00:00Z",
        last_poll_status="verified",
    )
    assert len(final_poll["current"]["polls"]) == 3
    with pytest.raises(ValueError, match="poll_limit"):
        module.update_current(
            final_poll,
            last_poll_at="2026-07-23T18:00:00Z",
            last_poll_status="verified",
        )


@pytest.mark.parametrize("poll_round_start", [-1, 3, 1.5, True])
def test_persisted_judging_state_validates_poll_round_start(
    poll_round_start: object, tmp_path: Path
):
    module = state_module()
    state = valid_judging_state(module)
    state["current"]["poll_round_start"] = poll_round_start

    with pytest.raises(ValueError, match="poll_round_start"):
        module.save_state(tmp_path / "repro-loop.json", state)


def valid_judging_state(module) -> dict:
    return module.transition(
        state_in_phase(module, "submitted"),
        "judging",
        poll_limit=2,
        poll_deadline="2026-07-22T18:00:00Z",
    )


def paper() -> dict:
    return {
        "paper_id": "icml-2026-001",
        "title": "Reliable Reproductions",
        "slug": "reliable-reproductions",
        "estimated_api_cost_usd": 4.25,
        "upstream_revision": "abc123",
        "target_claims": ["claim-1", "claim-2"],
    }


def rejection() -> dict:
    return {
        "paper_id": "icml-2026-002",
        "title": "Rejected Candidate",
        "reason": "No released artifacts",
        "checked_at": "2026-07-21T19:00:00Z",
    }


def verdict(status: str = "verified") -> dict:
    return {
        "claims": [
            {"claim": "claim-1", "status": status},
            {"claim": "claim-2", "status": status},
        ]
    }


def persisted_paper(**overrides: object) -> dict:
    record = {
        **paper(),
        "project_path": "submissions/reliable-reproductions",
        "polls": [],
        "design_approved": True,
        "deployed_sha": "prior123",
        "space_id": "org/prior-reproduction",
        "improvement_attempts": 0,
        "verdicts": [{**verdict(), "improvement_attempt": 0}],
        "verdict": verdict(),
    }
    record.update(overrides)
    return record


def state_in_phase(module, phase: str) -> dict:
    if phase == "idle":
        return module.new_state()
    if phase == "blocked":
        return module.transition(
            module.select_paper(module.new_state(), paper()),
            "blocked",
            blocker="credentials missing",
        )

    state = module.select_paper(module.new_state(), paper())
    if phase == "complete":
        state = module.transition(state, "design-pending")
        state = module.transition(state, "implementing", design_approved=True)
        state = module.transition(state, "validated")
        state = module.transition(state, "deployed", deployed_sha="abc123")
        state = module.transition(state, "submitted", space_id="org/reproduction")
        state = module.transition(
            state,
            "judging",
            poll_limit=2,
            poll_deadline="2026-07-22T18:00:00Z",
        )
        return module.transition(state, "complete", verdict=verdict())
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
        "judging": {
            "poll_limit": 2,
            "poll_deadline": "2026-07-22T18:00:00Z",
        },
        "improving": {
            "verdict": verdict("partial"),
            "improvement_reason": "Address verdict evidence gap",
        },
        "complete": {"verdict": verdict()},
        "blocked": {"blocker": "credentials missing"},
    }.get(phase, {})


def run_cli(*arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(STATE_MODULE_PATH), *arguments],
        check=True,
        capture_output=True,
        text=True,
    )
