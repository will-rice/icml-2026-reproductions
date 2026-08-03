"""Deterministic representative TimeRewarder evaluation and metrics."""

import gc
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import torch

from timerewarder_repro.checkpoint import (
    HELD_OUT_ORDINALS,
    TASK_CHECKPOINTS,
    anchor_indices,
)
from timerewarder_repro.media import decode_anchor_frames
from timerewarder_repro.model import (
    load_approved_model,
    predict_distances,
    preprocess_rgb,
)


def tie_aware_spearman(values: np.ndarray, order: np.ndarray) -> float:
    """Compute Spearman correlation using independent average ranks."""
    values = _finite_vector(values, "values")
    order = _finite_vector(order, "order")
    if values.shape != order.shape:
        raise ValueError("rank vectors must have equal shape")
    left = _average_ranks(values)
    right = _average_ranks(order)
    left -= left.mean(dtype=np.float64)
    right -= right.mean(dtype=np.float64)
    denominator = np.sqrt(
        np.sum(left * left, dtype=np.float64)
        * np.sum(right * right, dtype=np.float64)
    )
    if denominator == 0.0:
        return 0.0
    return float(np.sum(left * right, dtype=np.float64) / denominator)


def cumulative_anchor_values(
    forward: np.ndarray, reverse: np.ndarray
) -> np.ndarray:
    """Accumulate forward-minus-reverse adjacent-anchor rewards."""
    forward = _finite_vector(forward, "forward")
    reverse = _finite_vector(reverse, "reverse")
    if forward.shape != reverse.shape or forward.size != 4:
        raise ValueError("four forward and reverse adjacent scores are required")
    values = np.zeros(5, dtype=np.float64)
    values[1:] = np.cumsum(forward - reverse, dtype=np.float64)
    return values


def compute_distance_metrics(
    prediction: np.ndarray, target: np.ndarray
) -> dict[str, float]:
    """Compute preregistered float64 distance and direction measurements."""
    prediction = _finite_vector(prediction, "prediction")
    target = _finite_vector(target, "target")
    if prediction.shape != target.shape or prediction.size == 0:
        raise ValueError("prediction and target must have equal nonempty shape")
    error = np.abs(prediction - target)
    baseline = float(np.mean(np.abs(target), dtype=np.float64))
    prediction_mae = float(np.mean(error, dtype=np.float64))
    improvement = 0.0 if baseline == 0.0 else (baseline - prediction_mae) / baseline
    sign_accuracy = float(
        np.mean(np.sign(prediction) == np.sign(target), dtype=np.float64)
    )
    reverse_index = _reverse_pair_indices(target)
    antisymmetry = float(
        np.mean(np.abs(prediction + prediction[reverse_index]), dtype=np.float64)
    )
    return {
        "prediction_mae": prediction_mae,
        "zero_baseline_mae": baseline,
        "relative_improvement": float(improvement),
        "sign_accuracy": sign_accuracy,
        "mean_antisymmetry_error": antisymmetry,
    }


def task_passes(
    metrics: Mapping[str, float], *, tolerance: float = 1e-6
) -> bool:
    """Apply the frozen per-task acceptance thresholds."""
    required = (
        "prediction_mae",
        "relative_improvement",
        "sign_accuracy",
        "mean_antisymmetry_error",
    )
    if any(
        key not in metrics or not np.isfinite(float(metrics[key])) for key in required
    ):
        return False
    return bool(
        metrics["prediction_mae"] <= 0.20 + tolerance
        and metrics["relative_improvement"] >= 0.10 - tolerance
        and metrics["sign_accuracy"] >= 0.80 - tolerance
        and metrics["mean_antisymmetry_error"] <= 0.15 + tolerance
    )


def evaluate_representative(
    registry_path: Path,
    dataset_manifest_path: Path,
    schema_path: Path,
    cache_dir: Path,
) -> dict[str, object]:
    """Evaluate five pinned released videos per task, failing closed by stratum."""
    from timerewarder_repro.checkpoint import load_checkpoint_registry

    registry = load_checkpoint_registry(registry_path)
    dataset = _read_mapping(dataset_manifest_path, "dataset manifest")
    _validate_dataset_manifest(dataset)
    project_root = registry_path.resolve().parent.parent
    results: list[dict[str, object]] = []
    pooled_prediction: list[float] = []
    pooled_target: list[float] = []
    all_voc: list[float] = []

    registry_entries = {
        str(entry["task"]): entry for entry in registry["checkpoints"]
    }
    for task, checkpoint_file in TASK_CHECKPOINTS:
        entry = registry_entries[task]
        try:
            task_record = dataset["tasks"][task]
            maximum = max(
                int(dataset["files"][path]["frame_count"])
                for path in task_record["population_paths"]
            )
            output = (
                cache_dir
                / "converted"
                / Path(checkpoint_file).stem
                / "model.safetensors"
            )
            model = load_approved_model(
                output,
                project_root / str(entry["approval"]),
                project_root / str(entry["receipt"]),
                schema_path,
            )
            task_prediction: list[float] = []
            task_target: list[float] = []
            videos: list[dict[str, object]] = []
            for ordinal in HELD_OUT_ORDINALS:
                relative = task_record["held_out"][ordinal - 1]
                file_record = dataset["files"][relative]
                anchors = anchor_indices(int(file_record["frame_count"]))
                decoded = decode_anchor_frames(
                    cache_dir / "demos" / relative,
                    anchors,
                    expected_sha256=str(file_record["sha256"]),
                )
                frame_tensor = torch.stack([preprocess_rgb(frame) for frame in decoded])
                pairs = [
                    (start, end)
                    for start in range(5)
                    for end in range(5)
                    if start != end
                ]
                target = np.asarray(
                    [
                        (anchors[end] - anchors[start]) / maximum
                        for start, end in pairs
                    ],
                    dtype=np.float64,
                )
                prediction = predict_distances(model, frame_tensor, pairs)
                pair_to_value = {
                    pair: float(value)
                    for pair, value in zip(pairs, prediction, strict=True)
                }
                forward = np.asarray(
                    [pair_to_value[(index, index + 1)] for index in range(4)],
                    dtype=np.float64,
                )
                reverse = np.asarray(
                    [pair_to_value[(index + 1, index)] for index in range(4)],
                    dtype=np.float64,
                )
                values = cumulative_anchor_values(forward, reverse)
                voc = tie_aware_spearman(
                    values, np.arange(5, dtype=np.float64)
                )
                video_metrics = compute_distance_metrics(prediction, target)
                videos.append(
                    {
                        "ordinal": ordinal,
                        "path": relative,
                        "sha256": file_record["sha256"],
                        "size_bytes": file_record["size_bytes"],
                        "frame_count": file_record["frame_count"],
                        "anchors": anchors,
                        "metrics": video_metrics,
                        "voc": voc,
                    }
                )
                task_prediction.extend(prediction.tolist())
                task_target.extend(target.tolist())
                all_voc.append(voc)
            prediction_array = np.asarray(task_prediction, dtype=np.float64)
            target_array = np.asarray(task_target, dtype=np.float64)
            metrics = compute_distance_metrics(prediction_array, target_array)
            mean_voc = float(
                np.mean([video["voc"] for video in videos], dtype=np.float64)
            )
            results.append(
                {
                    "task": task,
                    "status": "available",
                    "checkpoint": checkpoint_file,
                    "checkpoint_sha256": entry["lfs_sha256"],
                    "approval_sha256": _read_mapping(
                        project_root / str(entry["approval"]), "approval"
                    )["approval_sha256"],
                    "max_frames": maximum,
                    "ordered_pairs": len(task_prediction),
                    "metrics": metrics,
                    "passes": task_passes(metrics),
                    "mean_voc": mean_voc,
                    "videos": videos,
                }
            )
            pooled_prediction.extend(task_prediction)
            pooled_target.extend(task_target)
            del model
            gc.collect()
        except Exception as error:
            results.append(
                {
                    "task": task,
                    "status": "unavailable",
                    "checkpoint": checkpoint_file,
                    "failure_category": type(error).__name__,
                    "failure": str(error).splitlines()[-1][:200],
                }
            )

    available = [record for record in results if record["status"] == "available"]
    pooled = (
        compute_distance_metrics(
            np.asarray(pooled_prediction, dtype=np.float64),
            np.asarray(pooled_target, dtype=np.float64),
        )
        if pooled_prediction
        else None
    )
    result: dict[str, object] = {
        "format": "timerewarder-representative-v1",
        "dataset": dataset["dataset"],
        "dataset_manifest_sha256": _sha256_file(dataset_manifest_path),
        "registry_sha256": _sha256_file(registry_path),
        "schema_sha256": _sha256_file(schema_path),
        "protocol": {
            "held_out_ordinals": list(HELD_OUT_ORDINALS),
            "anchors_per_video": 5,
            "ordered_pairs_per_video": 20,
            "videos_per_task": 5,
            "task_count": 10,
            "paper_figure_3_protocol": False,
        },
        "tasks": results,
        "successful_task_count": len(available),
        "passing_task_count": sum(
            bool(record.get("passes")) for record in available
        ),
        "pooled_metrics": pooled,
        "voc_values": all_voc,
        "mean_voc": (
            float(np.mean(all_voc, dtype=np.float64)) if all_voc else None
        ),
    }
    _reject_nonfinite(result)
    return result


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    position = 0
    while position < values.size:
        end = position + 1
        while end < values.size and values[order[end]] == values[order[position]]:
            end += 1
        ranks[order[position:end]] = (position + end - 1) / 2.0
        position = end
    return ranks


def _reverse_pair_indices(target: np.ndarray) -> np.ndarray:
    if target.size % 20 == 0:
        pairs = [
            (start, end)
            for start in range(5)
            for end in range(5)
            if start != end
        ]
        reverse = {pair: index for index, pair in enumerate(pairs)}
        indices = np.concatenate(
            [
                np.asarray(
                    [
                        offset + reverse[(end, start)]
                        for start, end in pairs
                    ],
                    dtype=np.int64,
                )
                for offset in range(0, target.size, 20)
            ]
        )
        if not np.allclose(target[indices], -target, atol=1e-15, rtol=0.0):
            raise ValueError("targets do not match ordered five-anchor pairs")
        return indices
    used: set[int] = set()
    result = np.empty(target.size, dtype=np.int64)
    for index, value in enumerate(target):
        candidates = np.flatnonzero(np.isclose(target, -value, atol=1e-15, rtol=0.0))
        candidate = next(
            (int(item) for item in candidates if int(item) not in used), None
        )
        if candidate is None:
            raise ValueError("targets do not form ordered reverse pairs")
        result[index] = candidate
        used.add(candidate)
    return result


def _finite_vector(value: np.ndarray, name: str) -> np.ndarray:
    if (
        not isinstance(value, np.ndarray)
        or value.dtype != np.float64
        or value.ndim != 1
        or value.size == 0
        or not np.isfinite(value).all()
    ):
        raise ValueError(f"{name} must be a finite float64 vector")
    return value.copy()


def _validate_dataset_manifest(dataset: dict[str, object]) -> None:
    expected = {
        "repository": "CowAndSheep/timerewarder-demos",
        "revision": "b966abcebc110dd97dd96018e395180e069756c4",
    }
    if (
        dataset.get("format") != "timerewarder-dataset-manifest-v1"
        or dataset.get("dataset") != expected
        or not isinstance(dataset.get("files"), dict)
        or not isinstance(dataset.get("tasks"), dict)
        or set(dataset["tasks"]) != {task for task, _ in TASK_CHECKPOINTS}
    ):
        raise ValueError("dataset manifest identity")


def _read_mapping(path: Path, label: str) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is unreadable") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a mapping")
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _reject_nonfinite(value: object) -> None:
    if isinstance(value, float) and not np.isfinite(value):
        raise ValueError("non-finite representative measurement")
    if isinstance(value, dict):
        for child in value.values():
            _reject_nonfinite(child)
    elif isinstance(value, list):
        for child in value:
            _reject_nonfinite(child)
