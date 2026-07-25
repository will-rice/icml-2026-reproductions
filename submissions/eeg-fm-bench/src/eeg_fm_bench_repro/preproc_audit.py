"""Execute released preprocessing methods on deterministic synthetic EEG."""

from __future__ import annotations

import ast
import hashlib
import math
import types
import warnings
from pathlib import Path
from typing import Any

import numpy as np

CLAIM_ID = "standardized-preprocessing-reproducibility"
_BUILDER = "data/processor/builder.py"
_DATASETS = {
    "adftd": ("data/dataset/adftd.py", "AdftdConfig", "AdftdBuilder"),
    "workload": ("data/dataset/workload.py", "WorkloadConfig", "WorkloadBuilder"),
}


def _tree(path: Path) -> tuple[str, ast.Module]:
    source = path.read_text(encoding="utf-8")
    return source, ast.parse(source, filename=str(path))


def _class(tree: ast.Module, name: str) -> ast.ClassDef:
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == name:
            return node
    raise ValueError(f"class {name} not found")


def _method(path: Path, class_name: str, method_name: str) -> tuple[Any, str]:
    source, tree = _tree(path)
    owner = _class(tree, class_name)
    node = next(
        (
            item
            for item in owner.body
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
            and item.name == method_name
        ),
        None,
    )
    if not isinstance(node, ast.FunctionDef):
        raise ValueError(f"{class_name}.{method_name} not found in {path}")
    segment = ast.get_source_segment(source, node)
    if segment is None:
        raise ValueError(f"cannot recover source for {class_name}.{method_name}")
    standalone = ast.fix_missing_locations(
        ast.Module(
            body=[
                ast.ImportFrom(
                    module="__future__",
                    names=[ast.alias(name="annotations")],
                    level=0,
                ),
                node,
            ],
            type_ignores=[],
        )
    )
    namespace = {
        "math": math,
        "np": np,
        "ndarray": np.ndarray,
        "warnings": warnings,
    }
    exec(compile(standalone, str(path), "exec"), namespace)
    return namespace[method_name], hashlib.sha256(segment.encode()).hexdigest()


def _field_literal(path: Path, class_name: str, field_name: str) -> Any:
    _, tree = _tree(path)
    for node in _class(tree, class_name).body:
        if not isinstance(node, ast.AnnAssign):
            continue
        if not isinstance(node.target, ast.Name) or node.target.id != field_name:
            continue
        value = node.value
        if isinstance(value, ast.Call):
            default_factory = next(
                (
                    keyword.value
                    for keyword in value.keywords
                    if keyword.arg == "default_factory"
                ),
                None,
            )
            if isinstance(default_factory, ast.Lambda):
                value = default_factory.body
        return ast.literal_eval(value)
    raise ValueError(f"{class_name}.{field_name} not found in {path}")


class _SyntheticRaw:
    def __init__(self, data: np.ndarray, sfreq: float):
        self.data = data
        self.info = {"sfreq": sfreq}
        self.duration = data.shape[1] / sfreq

    def filter(self, **_: Any) -> _SyntheticRaw:
        return self

    def notch_filter(self, **_: Any) -> _SyntheticRaw:
        return self

    def resample(self, sfreq: float, **_: Any) -> _SyntheticRaw:
        target = round(self.data.shape[1] * sfreq / self.info["sfreq"])
        old_axis = np.linspace(0.0, 1.0, self.data.shape[1])
        new_axis = np.linspace(0.0, 1.0, target)
        self.data = np.stack(
            [np.interp(new_axis, old_axis, channel) for channel in self.data]
        ).astype(np.float32)
        self.info["sfreq"] = sfreq
        self.duration = target / sfreq
        return self


def _one_dataset(
    snapshot: Path,
    dataset_name: str,
    dataset_path: str,
    config_name: str,
    builder_name: str,
    methods: dict[str, tuple[Any, str]],
) -> dict[str, Any]:
    montage = _field_literal(snapshot / dataset_path, config_name, "montage")
    montage_name = next(iter(montage))
    config = types.SimpleNamespace(
        montage=montage,
        fs=256.0,
        filter_high=40.0,
        filter_low=0.5,
        filter_notch=50.0,
        is_notched=True,
        wnd_len=2560,
        is_finetune=False,
        category=[],
        category_query_dict={},
        dataset_name=dataset_name,
        task_type=types.SimpleNamespace(value="synthetic_audit"),
    )
    builder = types.SimpleNamespace(
        config=config,
        _std_chs_cache={},
        montage_10_20_replace_dict={"T3": "T7", "T4": "T8", "T5": "P7", "T6": "P8"},
        _milli_sec_to_pts=lambda value: math.floor(value * config.fs / 1000),
    )
    standardize, _ = methods[f"{builder_name}.standardize_chs_names"]
    channels = standardize(builder, montage_name)

    rng = np.random.default_rng(20260724)
    original = rng.standard_normal((len(channels), 11_000), dtype=np.float32)

    def execute() -> tuple[list[dict[str, Any]], float]:
        raw = _SyntheticRaw(original.copy(), 500.0)
        processed = methods["EEGDatasetBuilder._resample_and_filter"][0](builder, raw)
        windows = methods["EEGDatasetBuilder._generate_window_sample"][0](
            builder,
            processed.data,
            montage_name,
            np.arange(len(channels), dtype=np.int64),
            [("default", 0, -1)],
            True,
        )
        return windows, float(processed.info["sfreq"])

    first, first_sfreq = execute()
    second, second_sfreq = execute()
    first_arrays = [item["data"] for item in first]
    second_arrays = [item["data"] for item in second]
    identical = first_sfreq == second_sfreq and len(first_arrays) == len(second_arrays)
    identical = identical and all(
        np.array_equal(left, right)
        for left, right in zip(first_arrays, second_arrays, strict=True)
    )
    digest = hashlib.sha256(b"".join(item.tobytes() for item in first_arrays)).hexdigest()
    return {
        "channel_count": len(channels),
        "channels": channels,
        "resampled_sfreq": first_sfreq,
        "window_shape": [len(first_arrays), len(channels), config.wnd_len],
        "repeat_identical": bool(identical),
        "output_sha256": digest,
    }


def run_preproc_audit(snapshot: Path) -> dict[str, Any]:
    """Run exact released method bodies in a controlled, raw-data-free fixture."""

    snapshot = Path(snapshot)
    builder_path = snapshot / _BUILDER
    methods = {
        "EEGDatasetBuilder._generate_window_sample": _method(
            builder_path, "EEGDatasetBuilder", "_generate_window_sample"
        ),
        "EEGDatasetBuilder._resample_and_filter": _method(
            builder_path, "EEGDatasetBuilder", "_resample_and_filter"
        ),
    }
    for _, (relative, _, builder_name) in _DATASETS.items():
        methods[f"{builder_name}.standardize_chs_names"] = _method(
            snapshot / relative, builder_name, "standardize_chs_names"
        )

    datasets = {
        name: _one_dataset(snapshot, name, relative, config_name, builder_name, methods)
        for name, (relative, config_name, builder_name) in _DATASETS.items()
    }
    deterministic = all(item["repeat_identical"] for item in datasets.values())
    return {
        "claim_id": CLAIM_ID,
        "kind": "numerical_audit",
        "status": "verified" if deterministic else "contradicted",
        "deterministic": deterministic,
        "datasets": datasets,
        "source_execution": {
            "methods": [
                "EEGDatasetBuilder._generate_window_sample",
                "EEGDatasetBuilder._resample_and_filter",
                "AdftdBuilder.standardize_chs_names",
                "WorkloadBuilder.standardize_chs_names",
            ],
            "sha256": {name: digest for name, (_, digest) in sorted(methods.items())},
            "note": "Exact pinned method ASTs executed with synthetic raw-data stand-ins.",
        },
    }
