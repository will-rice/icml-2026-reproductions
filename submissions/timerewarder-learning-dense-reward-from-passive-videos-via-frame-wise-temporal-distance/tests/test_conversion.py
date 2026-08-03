import hashlib
import inspect
import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

import timerewarder_repro.conversion as conversion
from timerewarder_repro.conversion import (
    ConversionRejected,
    approve_conversion,
    extract_and_validate_state_dict,
    inspect_checkpoint,
    validate_approval,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def schema() -> dict[str, object]:
    return {
        "top_level_keys": ["model"],
        "top_level_types": {"model": "builtins.dict"},
        "tensors": {
            "linear.bias": {"shape": [2], "dtype": "float32", "byte_size": 8},
            "linear.weight": {
                "shape": [2, 2],
                "dtype": "float32",
                "byte_size": 16,
            },
        },
        "minimum_tensor_bytes": 24,
        "maximum_tensor_bytes": 24,
        "maximum_output_bytes": 4096,
    }


def test_inspection_requires_exact_cfg_node_global(monkeypatch, tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint.pth"
    checkpoint.write_bytes(b"checkpoint")
    monkeypatch.setattr(
        torch.serialization,
        "get_unsafe_globals_in_checkpoint",
        lambda path: ["yacs.config.CfgNode"],
    )

    assert inspect_checkpoint(checkpoint) == ("yacs.config.CfgNode",)


def test_inspection_rejects_additional_global(monkeypatch, tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint.pth"
    checkpoint.write_bytes(b"checkpoint")
    monkeypatch.setattr(
        torch.serialization,
        "get_unsafe_globals_in_checkpoint",
        lambda path: ["yacs.config.CfgNode", "unexpected.Global"],
    )

    with pytest.raises(ConversionRejected, match="static_global_set"):
        inspect_checkpoint(checkpoint)


def test_converter_source_never_requests_unsafe_load() -> None:
    source = inspect.getsource(conversion)

    assert "weights_only=" + "False" not in source
    assert "weights_only=True" in source
    assert "safe_globals([CfgNode])" in source


def test_conversion_rejects_overlapping_tensor_storage(
    schema: dict[str, object],
) -> None:
    storage = torch.arange(5, dtype=torch.float32)
    loaded = {
        "model": {
            "linear.weight": storage[:4].reshape(2, 2),
            "linear.bias": storage[3:5],
        }
    }

    with pytest.raises(ConversionRejected, match="overlapping_storage"):
        extract_and_validate_state_dict(loaded, schema)


def test_conversion_rejects_unreviewed_root_metadata(
    schema: dict[str, object],
) -> None:
    schema = schema | {
        "top_level_keys": ["config", "model"],
        "top_level_types": {
            "config": "builtins.dict",
            "model": "collections.OrderedDict",
        },
    }
    loaded = {
        "config": object(),
        "model": {
            "linear.weight": torch.zeros((2, 2), dtype=torch.float32),
            "linear.bias": torch.zeros(2, dtype=torch.float32),
        },
    }

    with pytest.raises(ConversionRejected, match="top_level_metadata"):
        extract_and_validate_state_dict(loaded, schema)


def test_committed_basketball_schema_matches_isolated_probe() -> None:
    schema_path = Path(__file__).parents[1] / "artifacts" / "model-schema.json"
    schema = json.loads(schema_path.read_text(encoding="utf-8"))

    assert schema["source"] == {
        "checkpoint": "basketball_20bins.pth",
        "lfs_sha256": "4697468cdba176467df4ee0d8d766c6f53ad2305f87101a2511ec84e20410e50",
        "model_revision": "23eded140eb8c8d9f194243a115d218b5072d800",
        "size_bytes": 1293753599,
    }
    assert schema["top_level_types"] == {
        "config": "yacs.config.CfgNode",
        "epoch": "builtins.int",
        "lr_scheduler": "builtins.dict",
        "max_accuracy": "builtins.float",
        "model": "collections.OrderedDict",
        "optimizer": "builtins.dict",
    }
    assert len(schema["tensors"]) == 312
    assert schema["minimum_tensor_bytes"] == 603822160
    assert schema["maximum_tensor_bytes"] == 603822160


@pytest.fixture
def valid_receipt(tmp_path: Path, schema: dict[str, object]) -> tuple[Path, Path]:
    output = tmp_path / "model.safetensors"
    save_file({"linear.bias": torch.zeros(2, dtype=torch.float32)}, output)
    schema_path = tmp_path / "model-schema.json"
    schema_path.write_text(json.dumps(schema, sort_keys=True), encoding="utf-8")
    receipt = {
        "format": "timerewarder-conversion-receipt-v2",
        "approval_status": "pending_independent_reviewer",
        "task": "basketball-v3",
        "checkpoint_file": "basketball_20bins.pth",
        "checkpoint_sha256": "4" * 64,
        "checkpoint_bytes": 1293753599,
        "model_repository": "CowAndSheep/timerewarder",
        "model_revision": "23eded140eb8c8d9f194243a115d218b5072d800",
        "converter": "converter-a",
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
    receipt_path.write_text(json.dumps(receipt, sort_keys=True), encoding="utf-8")
    return receipt_path, output


def test_approval_requires_different_reviewer(valid_receipt: tuple[Path, Path]) -> None:
    receipt, output = valid_receipt

    with pytest.raises(ValueError, match="reviewer must differ"):
        approve_conversion(receipt, reviewer="converter-a", output_path=output)


def test_approval_binds_exact_output_hash(valid_receipt: tuple[Path, Path]) -> None:
    receipt, output = valid_receipt
    approval = approve_conversion(receipt, reviewer="reviewer-b", output_path=output)
    output.write_bytes(output.read_bytes() + b"x")

    with pytest.raises(ValueError, match="safetensors hash"):
        validate_approval(approval, output)


def test_approval_rejects_non_safetensors(valid_receipt: tuple[Path, Path]) -> None:
    receipt, output = valid_receipt
    output.write_bytes(b"not a safetensors container")
    updated_receipt = json.loads(receipt.read_text(encoding="utf-8"))
    updated_receipt["output_sha256"] = _sha256(output)
    updated_receipt["output_bytes"] = output.stat().st_size
    receipt.write_text(json.dumps(updated_receipt, sort_keys=True), encoding="utf-8")

    with pytest.raises(ValueError, match="safetensors container"):
        approve_conversion(receipt, reviewer="reviewer-b", output_path=output)
