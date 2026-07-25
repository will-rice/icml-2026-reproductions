"""Tests for immutable external-lifecycle attestations."""

import hashlib
import importlib
import json
from pathlib import Path
import sys

import pytest


SCRIPTS = (
    Path(__file__).resolve().parents[1] / "skills" / "icml-repro-loop" / "scripts"
)
KINDS = ("validation", "deployment", "submission", "verdict", "authority-audit")


def load_module(name: str):
    sys.path.insert(0, str(SCRIPTS))
    sys.modules.pop(name, None)
    return importlib.import_module(name)


@pytest.fixture
def store():
    return load_module("store")


@pytest.fixture
def attestations():
    load_module("store")
    return load_module("attestations")


@pytest.fixture
def paths(tmp_path, store):
    return store.StatePaths(tmp_path / "repro-loop.json")


def record(kind: str = "validation", **updates: object) -> dict:
    value = {
        "kind": kind,
        "attempt_id": "attempt-1",
        "attempt_number": 1,
        "observed_at": "2026-07-25T12:00:00+00:00",
        "source_commit": "abc123",
        "payload_sha256": "1" * 64,
    }
    if kind == "validation":
        value.update(validation_fields())
    value.update(updates)
    return value


def validation_fields() -> dict:
    return {
        "worktree": "/tmp/test-worktree",
        "branch": "test-branch",
        "base_sha": "2" * 40,
        "project_path": "submissions/paper-1",
        "design_path": "docs/designs/paper-1.md",
        "commands": [],
        "checks": [],
        "environment": [],
        "source_tree": "3" * 40,
        "environment_sha256": "4" * 64,
    }


def canonical_id(value: dict) -> str:
    payload = {key: item for key, item in value.items() if key != "attestation_id"}
    content = json.dumps(
        payload, allow_nan=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")
    return hashlib.sha256(content).hexdigest()


def object_path(paths, attestation_id: str) -> Path:
    return paths.root / "attestation-objects" / f"{attestation_id}.json"


def test_attestation_path_is_scoped_by_kind_attempt_and_number(paths):
    assert paths.attestation("validation", "attempt-1") == (
        paths.root / "attestations" / "validation" / "attempt-1--1.json"
    )
    assert paths.attestation("verdict", "attempt-1", 2) == (
        paths.root / "attestations" / "verdict" / "attempt-1--2.json"
    )


@pytest.mark.parametrize("kind", KINDS)
def test_persist_and_read_verify_content_addressed_attestation(
    paths, attestations, kind
):
    payload = record(kind)

    attestation_id = attestations.persist(paths, payload)

    assert attestation_id == canonical_id(payload)
    assert object_path(paths, attestation_id).exists()
    assert not paths.attestation(kind, "attempt-1").exists()
    assert attestations.read(paths, attestation_id) == {
        "attestation_id": attestation_id,
        **payload,
    }


def test_persist_is_byte_idempotent_and_allows_corrected_inert_same_slot_object(
    paths, attestations
):
    payload = record()
    attestation_id = attestations.persist(paths, payload)
    path = object_path(paths, attestation_id)
    original = path.read_bytes()

    assert attestations.persist(paths, payload) == attestation_id
    assert path.read_bytes() == original
    corrected = record(source_commit="different")
    corrected_id = attestations.persist(paths, corrected)
    assert corrected_id != attestation_id
    assert attestations.read(paths, corrected_id) == {
        "attestation_id": corrected_id,
        **corrected,
    }
    assert not paths.attestation("validation", "attempt-1").exists()
    assert path.read_bytes() == original


def test_read_rejects_content_that_no_longer_matches_attestation_id(
    paths, attestations, store
):
    attestation_id = attestations.persist(paths, record())
    path = object_path(paths, attestation_id)
    corrupted = store.read_json(path)
    corrupted["source_commit"] = "tampered"
    store._atomic_json_write(path, corrupted)

    with pytest.raises(ValueError, match="attestation_id"):
        attestations.read(paths, attestation_id)


@pytest.mark.parametrize(
    "invalid",
    [
        record(extra="not-kind-specific"),
        record(kind="unknown"),
        record(attempt_number=0),
        record(observed_at="2026-07-25T12:00:00"),
        record(source_commit=""),
        record(payload_sha256="not-a-sha"),
    ],
)
def test_persist_rejects_non_exact_or_invalid_common_envelope(
    paths, attestations, invalid
):
    with pytest.raises(ValueError):
        attestations.persist(paths, invalid)


def test_persist_rejects_conflicting_caller_supplied_attestation_id(
    paths, attestations
):
    with pytest.raises(ValueError, match="attestation_id"):
        attestations.persist(paths, record(attestation_id="0" * 64))


def test_read_rejects_unknown_attestation_id(paths, attestations):
    with pytest.raises(ValueError, match="attestation_id"):
        attestations.read(paths, "0" * 64)
