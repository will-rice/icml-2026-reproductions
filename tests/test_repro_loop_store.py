"""Tests for the schema-v6 sharded reproduction-loop store."""

import importlib.util
import threading
from pathlib import Path

import pytest


STORE_MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "skills"
    / "icml-repro-loop"
    / "scripts"
    / "store.py"
)


def store_module():
    """Load the store script without requiring package scaffolding."""
    spec = importlib.util.spec_from_file_location("repro_loop_store", STORE_MODULE_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    import sys

    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def new_attempt(attempt_id: str, paper_id: str) -> dict:
    return {
        "attempt_id": attempt_id,
        "paper_id": paper_id,
        "phase": "implementing",
        "updated_at": "2026-07-24T12:00:00+00:00",
    }


def run_barrier_writers(paths, *attempts: dict) -> None:
    barrier = threading.Barrier(len(attempts))
    errors = []

    def write(attempt: dict) -> None:
        try:
            barrier.wait()
            store_module().atomic_json_write(
                paths.attempt(attempt["attempt_id"]),
                attempt,
                store_module().validate_attempt,
            )
        except BaseException as error:  # pragma: no cover - asserted below
            errors.append(error)

    threads = [threading.Thread(target=write, args=(attempt,)) for attempt in attempts]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert errors == []


def test_state_paths_create_independent_shards(tmp_path):
    store = store_module()
    paths = store.StatePaths(tmp_path / "repro-loop.json")

    assert paths.attempt("attempt-1") == tmp_path / "repro-loop/attempts/attempt-1.json"
    assert paths.judgment("attempt-1") == (
        tmp_path / "repro-loop/judgments/attempt-1.json"
    )
    assert paths.lease("attempt--attempt-1") == (
        tmp_path / "repro-loop/leases/attempt--attempt-1.json"
    )


@pytest.mark.parametrize("identifier", ["", "../escape", "has/slash", "white space"])
def test_state_paths_reject_unsafe_identifiers(tmp_path, identifier):
    paths = store_module().StatePaths(tmp_path / "repro-loop.json")

    with pytest.raises(ValueError, match="identifier"):
        paths.attempt(identifier)


def test_new_index_has_twenty_paper_capacity():
    index = store_module().new_index()
    assert index["max_runnable_attempts"] == 20
    assert index["resource_limits"]["publication_per_provider"] == 20


def test_independent_attempt_writes_do_not_lose_updates(tmp_path):
    store = store_module()
    paths = store.StatePaths(tmp_path / "repro-loop.json")

    run_barrier_writers(paths, new_attempt("a1", "p1"), new_attempt("a2", "p2"))

    assert store.read_json(paths.attempt("a1"))["paper_id"] == "p1"
    assert store.read_json(paths.attempt("a2"))["paper_id"] == "p2"


def test_atomic_json_write_replaces_value_and_removes_temporary_file(tmp_path):
    store = store_module()
    path = tmp_path / "attempt.json"

    store.atomic_json_write(path, new_attempt("a1", "p1"), store.validate_attempt)
    replacement = {**new_attempt("a1", "p1"), "phase": "validated"}
    store.atomic_json_write(path, replacement, store.validate_attempt)

    assert store.read_json(path) == replacement
    assert list(tmp_path.glob("*.tmp")) == []


def test_atomic_json_write_requires_validator(tmp_path):
    store = store_module()

    with pytest.raises(TypeError, match="validator"):
        store.atomic_json_write(tmp_path / "attempt.json", new_attempt("a1", "p1"))


def test_locked_json_serializes_updates(tmp_path):
    store = store_module()
    path = tmp_path / "index.json"
    store.atomic_json_write(path, store.new_index(), store.validate_index)

    with store.locked_json(path, store.validate_index) as index:
        index["rejections"].append({"paper_id": "p1"})

    assert store.read_json(path)["rejections"] == [{"paper_id": "p1"}]


def test_locked_json_requires_validator(tmp_path):
    store = store_module()

    with pytest.raises(TypeError, match="validator"):
        store.locked_json(tmp_path / "index.json")


def test_validate_index_requires_exact_attempt_reference_fields():
    store = store_module()
    index = store.new_index()
    index["attempts"]["a1"] = {
        "path": "repro-loop/attempts/a1.json",
        "paper_id": "p1",
        "phase": "implementing",
        "updated_at": "2026-07-24T12:00:00+00:00",
    }
    store.validate_index(index)
    index["attempts"]["a1"]["extra"] = True

    with pytest.raises(ValueError, match="attempts"):
        store.validate_index(index)


@pytest.mark.parametrize(
    ("validator_name", "value"),
    [
        ("validate_attempt", {"attempt_id": "a1"}),
        ("validate_judgment", {"attempt_id": "a1"}),
        ("validate_snapshot", {"snapshot_id": "s1"}),
    ],
)
def test_shard_validators_require_identity_and_timestamp(validator_name, value):
    validator = getattr(store_module(), validator_name)

    with pytest.raises(ValueError):
        validator(value)
