import hashlib
import inspect
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from safetensors.torch import save_file

from timerewarder_repro.conversion import approve_conversion
from timerewarder_repro.model import (
    load_approved_model,
    predict_distances,
    preprocess_rgb,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tiny_approved_fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    output = tmp_path / "model.safetensors"
    save_file({"weight": torch.arange(4, dtype=torch.float32).reshape(2, 2)}, output)
    schema = {
        "format": "timerewarder-model-schema-v1",
        "tensors": {
            "weight": {"shape": [2, 2], "dtype": "float32", "byte_size": 16}
        },
    }
    schema_path = tmp_path / "schema.json"
    schema_path.write_text(json.dumps(schema, sort_keys=True), encoding="utf-8")
    receipt = {
        "format": "timerewarder-conversion-receipt-v2",
        "approval_status": "pending_independent_reviewer",
        "task": "tiny",
        "checkpoint_file": "tiny.pth",
        "checkpoint_sha256": "1" * 64,
        "checkpoint_bytes": 1,
        "model_repository": "example/model",
        "model_revision": "2" * 40,
        "converter": "converter",
        "status": "success",
        "static_globals": ["yacs.config.CfgNode"],
        "load": {"weights_only": True, "mmap": True, "map_location": "cpu"},
        "safe_globals_empty": True,
        "child_exit": 0,
        "schema_sha256": _sha256(schema_path),
        "output_sha256": _sha256(output),
        "output_bytes": output.stat().st_size,
        "tensor_checks": {"exact_schema": True, "overlapping_storage": False},
        "sandbox": {"network_namespace": "none", "inherited_network_fds": 0},
    }
    receipt_path = tmp_path / "receipt.json"
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    approval = approve_conversion(receipt_path, "reviewer", output)
    approval_path = tmp_path / "approval.json"
    approval_path.write_text(json.dumps(approval), encoding="utf-8")
    return output, approval_path, receipt_path, schema_path


def test_loader_validates_tiny_approved_safetensors(tmp_path: Path) -> None:
    output, approval, receipt, schema = _tiny_approved_fixture(tmp_path)

    model = load_approved_model(output, approval, receipt, schema)

    assert isinstance(model, torch.nn.Module)
    assert model.training is False
    assert {parameter.device.type for parameter in model.parameters()} <= {"cpu"}


def test_loader_rejects_legacy_extension_before_model_construction(
    tmp_path: Path,
) -> None:
    legacy = tmp_path / "model.pth"
    legacy.write_bytes(b"never open")

    with pytest.raises(ValueError, match="safetensors"):
        load_approved_model(legacy, tmp_path / "a", tmp_path / "r", tmp_path / "s")


def test_preprocess_rgb_uses_fixed_resize_crop_and_normalization() -> None:
    frame = np.zeros((300, 400, 3), dtype=np.uint8)
    frame[..., 0] = 123
    frame[..., 1] = 116
    frame[..., 2] = 103

    result = preprocess_rgb(frame)

    assert result.shape == (3, 224, 224)
    assert result.dtype == torch.float32
    assert result[:, 112, 112].tolist() == pytest.approx(
        [
            (123.0 - 123.675) / 58.395,
            (116.0 - 116.28) / 57.12,
            (103.0 - 103.53) / 57.375,
        ],
        rel=1e-4,
        abs=1e-6,
    )


def test_predict_distances_encodes_frames_once() -> None:
    class CountingModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.calls = 0

        def encode_frames(self, frames: torch.Tensor) -> torch.Tensor:
            self.calls += 1
            return frames[:, :1, 0, 0]

        def pair_logits(
            self, features: torch.Tensor, pairs: list[tuple[int, int]]
        ) -> torch.Tensor:
            logits = torch.zeros((len(pairs), 20), dtype=torch.float64)
            for row, (start, end) in enumerate(pairs):
                logits[row, 19 if features[end] > features[start] else 0] = 50
            return logits

    model = CountingModel()
    frames = torch.arange(3, dtype=torch.float32).reshape(3, 1, 1, 1)

    values = predict_distances(model, frames, [(0, 1), (2, 0)])

    assert model.calls == 1
    assert values.shape == (2,)
    assert values[0] > 0
    assert values[1] < 0


def test_runtime_module_has_no_unsafe_import_or_load_reference() -> None:
    import timerewarder_repro.model as model_module

    source = inspect.getsource(model_module)
    for forbidden in ("conversion", "torch.load", "pickle", "yacs"):
        assert forbidden not in source
