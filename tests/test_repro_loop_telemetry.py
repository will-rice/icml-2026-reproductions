"""Tests for the append-only reproduction-loop telemetry store."""

import importlib.util
import json
from pathlib import Path
import sys

import pytest


SCRIPTS_PATH = (
    Path(__file__).resolve().parents[1]
    / "skills"
    / "icml-repro-loop"
    / "scripts"
)
STORE_MODULE_PATH = SCRIPTS_PATH / "store.py"
TELEMETRY_MODULE_PATH = SCRIPTS_PATH / "telemetry.py"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def store():
    return load_module("repro_loop_telemetry_store", STORE_MODULE_PATH)


@pytest.fixture
def telemetry():
    sys.modules.pop("telemetry", None)
    return load_module("telemetry", TELEMETRY_MODULE_PATH)


@pytest.fixture
def paths(tmp_path, store):
    return store.StatePaths(tmp_path / "repro-loop.json")


def test_events_are_append_only_and_ordered(paths, telemetry):
    queued = telemetry.append_event(
        paths,
        "session-a",
        0,
        "worker-queued",
        {
            "attempt_id": "attempt-a",
            "paper_id": "paper-a",
            "observed_at": "2026-07-27T00:00:00+00:00",
        },
    )
    assert queued["sequence"] == 0
    with pytest.raises(FileExistsError):
        telemetry.append_event(
            paths,
            "session-a",
            0,
            "worker-queued",
            {
                "attempt_id": "attempt-a",
                "paper_id": "paper-a",
                "observed_at": "2026-07-27T00:00:00+00:00",
            },
        )
    assert telemetry.read_session(paths, "session-a") == [queued]
    with pytest.raises(ValueError, match="reserved"):
        telemetry.append_event(
            paths,
            "session-a",
            1,
            "worker-launched",
            {"session_id": "spoofed"},
        )


def test_missing_exit_is_open_and_has_no_guessed_duration(paths, telemetry):
    telemetry.append_event(
        paths,
        "session-a",
        0,
        "worker-launched",
        {
            "attempt_id": "attempt-a",
            "paper_id": "paper-a",
            "observed_at": "2026-07-27T00:00:00+00:00",
            "monotonic_ns": 100,
        },
    )
    summary = telemetry.summarize_worker_session(
        telemetry.read_session(paths, "session-a")
    )
    assert summary["status"] == "open"
    assert summary["elapsed_seconds"] is None


def test_exited_worker_uses_monotonic_duration(paths, telemetry):
    telemetry.append_event(
        paths,
        "session-a",
        0,
        "worker-launched",
        {"monotonic_ns": 1_000_000_000},
    )
    telemetry.append_event(
        paths,
        "session-a",
        1,
        "worker-exited",
        {"monotonic_ns": 3_500_000_000},
    )

    summary = telemetry.summarize_worker_session(
        telemetry.read_session(paths, "session-a")
    )

    assert summary == {
        "session_id": "session-a",
        "status": "exited",
        "elapsed_seconds": 2.5,
    }


def test_exit_before_launch_counter_has_no_guessed_duration(paths, telemetry):
    telemetry.append_event(
        paths,
        "session-a",
        0,
        "worker-launched",
        {"monotonic_ns": 200},
    )
    telemetry.append_event(
        paths,
        "session-a",
        1,
        "worker-exited",
        {"monotonic_ns": 100},
    )

    summary = telemetry.summarize_worker_session(
        telemetry.read_session(paths, "session-a")
    )

    assert summary["status"] == "exited"
    assert summary["elapsed_seconds"] is None


@pytest.mark.parametrize(
    ("records", "message"),
    [
        (
            [
                (0, "worker-queued", "session-a"),
                (0, "worker-launched", "session-a"),
            ],
            "sequence",
        ),
        ([(0, "worker-queued", "session-b")], "session_id"),
        ([(1, "worker-queued", "session-a")], "sequence"),
    ],
)
def test_read_session_rejects_corrupt_history(paths, telemetry, records, message):
    for sequence, event, record_session_id in records:
        path = paths.telemetry_event("session-a", sequence, event)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "version": 1,
                    "session_id": record_session_id,
                    "sequence": sequence,
                    "event": event,
                }
            ),
            encoding="utf-8",
        )

    with pytest.raises(ValueError, match=message):
        telemetry.read_session(paths, "session-a")


def test_read_session_rejects_boolean_version(paths, telemetry):
    path = paths.telemetry_event("session-a", 0, "worker-queued")
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps(
            {
                "version": True,
                "session_id": "session-a",
                "sequence": 0,
                "event": "worker-queued",
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="version"):
        telemetry.read_session(paths, "session-a")


def test_iter_sessions_sorts_validated_session_histories(paths, telemetry):
    telemetry.append_event(
        paths, "session-b", 0, "worker-queued", {"paper_id": "paper-b"}
    )
    telemetry.append_event(
        paths, "session-a", 0, "worker-queued", {"paper_id": "paper-a"}
    )

    histories = list(telemetry.iter_sessions(paths))

    assert [[event["session_id"] for event in history] for history in histories] == [
        ["session-a"],
        ["session-b"],
    ]
