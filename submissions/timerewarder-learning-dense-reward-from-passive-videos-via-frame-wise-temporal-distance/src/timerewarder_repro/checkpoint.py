"""Frozen representative-evaluation protocol for approved TimeRewarder inputs."""

import json
import math
import re
from collections.abc import Mapping
from pathlib import Path

TASK_CHECKPOINTS = (
    ("basketball-v3", "basketball_20bins.pth"),
    ("button-press-topdown-v2", "button_press_topdown_20bins.pth"),
    ("disassemble-v2", "disassemble_20bins.pth"),
    ("door-open-v2", "door_open_20bins.pth"),
    ("drawer-open-v2", "drawer_open_20bins.pth"),
    ("lever-pull-v2", "lever_pull_20bins.pth"),
    ("plate-slide-v2", "plate_slide_20bins.pth"),
    ("stick-push-v2", "stick_push_20bins.pth"),
    ("window-close-v2", "window_close_20bins.pth"),
    ("window-open-v2", "window_open_20bins.pth"),
)
HELD_OUT_ORDINALS = (1, 26, 51, 76, 100)
MODEL_REPOSITORY = "CowAndSheep/timerewarder"
MODEL_REVISION = "23eded140eb8c8d9f194243a115d218b5072d800"
MODEL_SCHEMA_SHA256 = "b85388515bb8e5eef2735b4a0a3c62889682a2d4e0958f492631b3c1fbc5bab3"
_SHA256 = re.compile(r"[0-9a-f]{64}")
_REGISTRY_KEYS = {"format", "model", "schema", "checkpoints"}
_ENTRY_KEYS = {
    "task",
    "file",
    "repository",
    "model_revision",
    "lfs_sha256",
    "size_bytes",
    "schema_sha256",
    "receipt",
    "approval",
}


def load_checkpoint_registry(path: Path) -> dict[str, object]:
    """Load the exact ten-checkpoint registry and reject mutable identities."""
    try:
        registry = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("checkpoint registry is unreadable") from error
    if not isinstance(registry, dict) or set(registry) != _REGISTRY_KEYS:
        raise ValueError("checkpoint registry schema")
    if registry["format"] != "timerewarder-checkpoint-registry-v1":
        raise ValueError("checkpoint registry format")
    if registry["model"] != {
        "repository": MODEL_REPOSITORY,
        "revision": MODEL_REVISION,
    }:
        raise ValueError("checkpoint registry revision")
    if registry["schema"] != {
        "path": "artifacts/model-schema.json",
        "sha256": MODEL_SCHEMA_SHA256,
    }:
        raise ValueError("checkpoint registry schema")

    entries = registry["checkpoints"]
    if not isinstance(entries, list) or len(entries) != len(TASK_CHECKPOINTS):
        raise ValueError("checkpoint registry task set")
    if not all(isinstance(entry, dict) for entry in entries):
        raise ValueError("checkpoint registry entry")
    tasks = [entry.get("task") for entry in entries]
    if tasks != [task for task, _ in TASK_CHECKPOINTS]:
        raise ValueError("checkpoint registry task set")

    observed_hashes: set[str] = set()
    for entry, (task, filename) in zip(entries, TASK_CHECKPOINTS, strict=True):
        if set(entry) != _ENTRY_KEYS:
            raise ValueError("checkpoint registry entry")
        if entry["file"] != filename:
            raise ValueError(f"checkpoint registry filename: {task}")
        if (
            entry["repository"] != MODEL_REPOSITORY
            or entry["model_revision"] != MODEL_REVISION
        ):
            raise ValueError(f"checkpoint registry revision: {task}")
        lfs_sha256 = entry["lfs_sha256"]
        if not isinstance(lfs_sha256, str) or _SHA256.fullmatch(lfs_sha256) is None:
            raise ValueError(f"checkpoint registry SHA-256: {task}")
        if lfs_sha256 in observed_hashes:
            raise ValueError("checkpoint registry duplicate SHA-256")
        observed_hashes.add(lfs_sha256)
        if type(entry["size_bytes"]) is not int or entry["size_bytes"] <= 0:
            raise ValueError(f"checkpoint registry size: {task}")
        if entry["schema_sha256"] != MODEL_SCHEMA_SHA256:
            raise ValueError(f"checkpoint registry schema: {task}")
        stem = Path(filename).stem
        if entry["receipt"] != f"artifacts/conversion-receipts/{stem}.json":
            raise ValueError(f"checkpoint registry receipt path: {task}")
        if entry["approval"] != f"artifacts/conversion-approvals/{stem}.json":
            raise ValueError(f"checkpoint registry approval path: {task}")
    return registry


def checkpoint_entry(registry: Mapping[str, object], task: str) -> Mapping[str, object]:
    """Return one exact task entry from an already validated registry."""
    entries = registry.get("checkpoints")
    if not isinstance(entries, list):
        raise ValueError("checkpoint registry entries")
    for entry in entries:
        if isinstance(entry, dict) and entry.get("task") == task:
            return entry
    raise ValueError(f"unknown checkpoint task: {task}")


def anchor_indices(frame_count: int) -> list[int]:
    """Select the five preregistered anchors from one decoded video."""
    if (
        not isinstance(frame_count, int)
        or isinstance(frame_count, bool)
        or frame_count < 5
    ):
        raise ValueError("selected video has fewer than five frames")
    return [math.floor(index * (frame_count - 1) / 4) for index in range(5)]


def ordered_anchor_pairs(
    anchors: list[int], denominator: int
) -> list[dict[str, float | int]]:
    """Enumerate every non-self anchor pairing in deterministic order."""
    if (
        not isinstance(denominator, int)
        or isinstance(denominator, bool)
        or denominator < 1
    ):
        raise ValueError("task population denominator must be positive")
    return [
        {"start": start, "end": end, "target": (end - start) / denominator}
        for start in anchors
        for end in anchors
        if start != end
    ]


def build_protocol(
    dataset_manifest: dict[str, object], frame_counts: dict[str, int]
) -> tuple[dict[str, object], ...]:
    """Build the fixed 50-video, 1,000-pair protocol before inference."""
    tasks = []
    for task, checkpoint in TASK_CHECKPOINTS:
        annotations = _validated_annotations(dataset_manifest, task)
        population_paths = annotations["population_paths"]
        maximum = max(_frame_count(frame_counts, path) for path in population_paths)
        videos = []
        for ordinal in HELD_OUT_ORDINALS:
            path = annotations["held_out"][ordinal - 1]
            frame_count = _frame_count(frame_counts, path)
            anchors = anchor_indices(frame_count)
            videos.append(
                {
                    "path": path,
                    "frame_count": frame_count,
                    "anchors": anchors,
                    "ordered_pairs": ordered_anchor_pairs(anchors, maximum),
                }
            )
        tasks.append(
            {
                "task": task,
                "checkpoint": checkpoint,
                "max_frames": maximum,
                "held_out_ordinals": list(HELD_OUT_ORDINALS),
                "videos": videos,
            }
        )
    if (
        sum(len(video["ordered_pairs"]) for task in tasks for video in task["videos"])
        != 1000
    ):
        raise ValueError("protocol must contain exactly 1,000 ordered pairs")
    return tuple(tasks)


def _validated_annotations(
    dataset_manifest: dict[str, object], task: str
) -> dict[str, list[str]]:
    if not isinstance(dataset_manifest, dict):
        raise ValueError("dataset manifest must be a mapping")
    task_records = dataset_manifest.get("tasks")
    if not isinstance(task_records, dict) or task not in task_records:
        raise ValueError(f"missing annotations for task: {task}")
    record = task_records[task]
    if not isinstance(record, dict):
        raise ValueError(f"invalid annotations for task: {task}")
    held_out = record.get("held_out")
    population_paths = record.get("population_paths")
    if not isinstance(held_out, list) or len(held_out) != 100:
        raise ValueError(f"task must have exactly 100 held-out annotations: {task}")
    if not isinstance(population_paths, list) or not population_paths:
        raise ValueError(f"task must have population annotations: {task}")
    if not all(isinstance(path, str) and path for path in held_out + population_paths):
        raise ValueError(f"task annotations must be nonempty paths: {task}")
    if len(set(held_out)) != len(held_out) or len(set(population_paths)) != len(
        population_paths
    ):
        raise ValueError(f"task annotations must not contain duplicates: {task}")
    return {"held_out": held_out, "population_paths": population_paths}


def _frame_count(frame_counts: dict[str, int], path: str) -> int:
    if path not in frame_counts:
        raise ValueError(f"missing decoded frame count: {path}")
    frame_count = frame_counts[path]
    if (
        not isinstance(frame_count, int)
        or isinstance(frame_count, bool)
        or frame_count < 1
    ):
        raise ValueError(f"invalid decoded frame count: {path}")
    return frame_count
