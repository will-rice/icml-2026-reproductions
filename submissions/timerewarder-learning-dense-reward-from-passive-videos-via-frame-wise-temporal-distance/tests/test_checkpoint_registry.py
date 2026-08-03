import hashlib
import json
from pathlib import Path

import pytest

from timerewarder_repro.checkpoint import (
    TASK_CHECKPOINTS,
    checkpoint_entry,
    load_checkpoint_registry,
)


MODEL_REPOSITORY = "CowAndSheep/timerewarder"
MODEL_REVISION = "23eded140eb8c8d9f194243a115d218b5072d800"
SCHEMA_SHA256 = "b85388515bb8e5eef2735b4a0a3c62889682a2d4e0958f492631b3c1fbc5bab3"


def _registry() -> dict[str, object]:
    return {
        "format": "timerewarder-checkpoint-registry-v1",
        "model": {
            "repository": MODEL_REPOSITORY,
            "revision": MODEL_REVISION,
        },
        "schema": {
            "path": "artifacts/model-schema.json",
            "sha256": SCHEMA_SHA256,
        },
        "checkpoints": [
            {
                "task": task,
                "file": filename,
                "repository": MODEL_REPOSITORY,
                "model_revision": MODEL_REVISION,
                "lfs_sha256": hashlib.sha256(task.encode()).hexdigest(),
                "size_bytes": 1000 + index,
                "schema_sha256": SCHEMA_SHA256,
                "receipt": f"artifacts/conversion-receipts/{Path(filename).stem}.json",
                "approval": f"artifacts/conversion-approvals/{Path(filename).stem}.json",
            }
            for index, (task, filename) in enumerate(TASK_CHECKPOINTS)
        ],
    }


def _write_registry(tmp_path: Path, registry: dict[str, object]) -> Path:
    path = tmp_path / "checkpoints.json"
    path.write_text(json.dumps(registry), encoding="utf-8")
    return path


def test_committed_registry_pins_all_ten_checkpoints() -> None:
    path = Path(__file__).parents[1] / "artifacts" / "checkpoints.json"

    registry = load_checkpoint_registry(path)

    assert registry["model"] == {
        "repository": MODEL_REPOSITORY,
        "revision": MODEL_REVISION,
    }
    assert registry["schema"] == {
        "path": "artifacts/model-schema.json",
        "sha256": SCHEMA_SHA256,
    }
    entries = registry["checkpoints"]
    assert [(entry["task"], entry["file"]) for entry in entries] == list(
        TASK_CHECKPOINTS
    )
    assert len({entry["lfs_sha256"] for entry in entries}) == 10
    assert all(
        len(entry["lfs_sha256"]) == 64
        and set(entry["lfs_sha256"]) <= set("0123456789abcdef")
        and type(entry["size_bytes"]) is int
        and entry["size_bytes"] > 0
        and entry["schema_sha256"] == SCHEMA_SHA256
        for entry in entries
    )


def test_checkpoint_entry_returns_only_named_task(tmp_path: Path) -> None:
    registry = load_checkpoint_registry(_write_registry(tmp_path, _registry()))

    entry = checkpoint_entry(registry, "door-open-v2")

    assert entry["file"] == "door_open_20bins.pth"
    assert entry["task"] == "door-open-v2"


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda value: value["checkpoints"].pop(), "task set"),
        (
            lambda value: value["checkpoints"].append(value["checkpoints"][0].copy()),
            "task set",
        ),
        (
            lambda value: value["checkpoints"][0].update(
                {"task": "unknown-task", "file": "unknown.pth"}
            ),
            "task set",
        ),
        (
            lambda value: value["checkpoints"][0].update({"file": "wrong.pth"}),
            "filename",
        ),
        (
            lambda value: value["checkpoints"][0].update({"model_revision": "main"}),
            "revision",
        ),
        (
            lambda value: value["checkpoints"][0].update({"lfs_sha256": "bad"}),
            "SHA-256",
        ),
        (
            lambda value: value["checkpoints"][0].update({"size_bytes": 0}),
            "size",
        ),
        (
            lambda value: value["checkpoints"][1].update(
                {"lfs_sha256": value["checkpoints"][0]["lfs_sha256"]}
            ),
            "duplicate",
        ),
        (
            lambda value: value["checkpoints"][0].update({"schema_sha256": "0" * 64}),
            "schema",
        ),
    ],
)
def test_registry_rejects_identity_drift(
    tmp_path: Path, mutation, message: str
) -> None:
    registry = _registry()
    mutation(registry)

    with pytest.raises(ValueError, match=message):
        load_checkpoint_registry(_write_registry(tmp_path, registry))


def test_checkpoint_entry_rejects_unknown_task(tmp_path: Path) -> None:
    registry = load_checkpoint_registry(_write_registry(tmp_path, _registry()))

    with pytest.raises(ValueError, match="unknown checkpoint task"):
        checkpoint_entry(registry, "not-a-task")
