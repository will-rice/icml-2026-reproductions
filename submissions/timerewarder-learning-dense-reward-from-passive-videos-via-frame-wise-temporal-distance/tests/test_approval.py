import hashlib
import inspect
import json
from pathlib import Path

import pytest

import timerewarder_repro.approval as approval_module
from timerewarder_repro.approval import validate_approval_record


SCHEMA_SHA256 = "b85388515bb8e5eef2735b4a0a3c62889682a2d4e0958f492631b3c1fbc5bab3"
MODEL_REVISION = "23eded140eb8c8d9f194243a115d218b5072d800"


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def _records(tmp_path: Path) -> tuple[dict[str, object], dict[str, object], Path]:
    output = tmp_path / "model.safetensors"
    output.write_bytes(b"tensor-only-output")
    receipt = {
        "format": "timerewarder-conversion-receipt-v2",
        "approval_status": "pending_independent_reviewer",
        "task": "basketball-v3",
        "checkpoint_file": "basketball_20bins.pth",
        "checkpoint_sha256": "4" * 64,
        "checkpoint_bytes": 1293753599,
        "model_repository": "CowAndSheep/timerewarder",
        "model_revision": MODEL_REVISION,
        "converter": "converter-a",
        "status": "success",
        "static_globals": ["yacs.config.CfgNode"],
        "load": {"weights_only": True, "mmap": True, "map_location": "cpu"},
        "safe_globals_empty": True,
        "child_exit": 0,
        "schema_sha256": SCHEMA_SHA256,
        "output_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "output_bytes": output.stat().st_size,
        "tensor_checks": {"exact_schema": True, "overlapping_storage": False},
        "sandbox": {"network_namespace": "none", "inherited_network_fds": 0},
    }
    approval = {
        "format": "timerewarder-conversion-approval-v1",
        "status": "approved",
        "receipt": json.loads(json.dumps(receipt)),
        "receipt_sha256": _canonical_sha256(receipt),
        "converter": "converter-a",
        "reviewer": "reviewer-b",
        "checkpoint_sha256": receipt["checkpoint_sha256"],
        "checkpoint_bytes": receipt["checkpoint_bytes"],
        "model_revision": MODEL_REVISION,
        "schema_sha256": SCHEMA_SHA256,
        "output_sha256": receipt["output_sha256"],
        "output_bytes": receipt["output_bytes"],
    }
    approval["approval_sha256"] = _canonical_sha256(approval)
    return approval, receipt, output


def _resign(approval: dict[str, object]) -> None:
    approval.pop("approval_sha256", None)
    approval["approval_sha256"] = _canonical_sha256(approval)


def test_valid_approval_binds_receipt_and_output(tmp_path: Path) -> None:
    approval, receipt, output = _records(tmp_path)

    validated = validate_approval_record(
        approval,
        receipt=receipt,
        output_path=output,
        expected_schema_sha256=SCHEMA_SHA256,
    )

    assert validated == approval


@pytest.mark.parametrize("status", ["pending", "rejected", "", None])
def test_approval_requires_approved_status(tmp_path: Path, status: object) -> None:
    approval, receipt, output = _records(tmp_path)
    approval["status"] = status
    _resign(approval)

    with pytest.raises(ValueError, match="status"):
        validate_approval_record(
            approval,
            receipt=receipt,
            output_path=output,
            expected_schema_sha256=SCHEMA_SHA256,
        )


def test_approval_rejects_self_review(tmp_path: Path) -> None:
    approval, receipt, output = _records(tmp_path)
    approval["reviewer"] = approval["converter"]
    _resign(approval)

    with pytest.raises(ValueError, match="reviewer"):
        validate_approval_record(
            approval,
            receipt=receipt,
            output_path=output,
            expected_schema_sha256=SCHEMA_SHA256,
        )


@pytest.mark.parametrize(
    "field",
    [
        "checkpoint_sha256",
        "checkpoint_bytes",
        "model_revision",
        "schema_sha256",
        "output_sha256",
        "output_bytes",
    ],
)
def test_approval_rejects_identity_drift(tmp_path: Path, field: str) -> None:
    approval, receipt, output = _records(tmp_path)
    approval[field] = "0" * 64 if isinstance(approval[field], str) else 1
    _resign(approval)

    with pytest.raises(ValueError, match="identity|schema|output"):
        validate_approval_record(
            approval,
            receipt=receipt,
            output_path=output,
            expected_schema_sha256=SCHEMA_SHA256,
        )


def test_approval_rejects_mutated_receipt(tmp_path: Path) -> None:
    approval, receipt, output = _records(tmp_path)
    receipt["checkpoint_bytes"] = 1

    with pytest.raises(ValueError, match="receipt"):
        validate_approval_record(
            approval,
            receipt=receipt,
            output_path=output,
            expected_schema_sha256=SCHEMA_SHA256,
        )


def test_approval_rejects_mutated_output(tmp_path: Path) -> None:
    approval, receipt, output = _records(tmp_path)
    output.write_bytes(b"changed")

    with pytest.raises(ValueError, match="output"):
        validate_approval_record(
            approval,
            receipt=receipt,
            output_path=output,
            expected_schema_sha256=SCHEMA_SHA256,
        )


def test_approval_rejects_non_safetensors_path(tmp_path: Path) -> None:
    approval, receipt, output = _records(tmp_path)
    wrong = output.with_suffix(".pth")
    output.rename(wrong)

    with pytest.raises(ValueError, match="safetensors"):
        validate_approval_record(
            approval,
            receipt=receipt,
            output_path=wrong,
            expected_schema_sha256=SCHEMA_SHA256,
        )


def test_approval_rejects_missing_field(tmp_path: Path) -> None:
    approval, receipt, output = _records(tmp_path)
    approval.pop("reviewer")
    _resign(approval)

    with pytest.raises(ValueError, match="schema"):
        validate_approval_record(
            approval,
            receipt=receipt,
            output_path=output,
            expected_schema_sha256=SCHEMA_SHA256,
        )


def test_approval_module_has_only_standard_library_boundary() -> None:
    source = inspect.getsource(approval_module)
    imported_modules = {
        value.__name__.split(".", 1)[0]
        for value in approval_module.__dict__.values()
        if inspect.ismodule(value)
    }

    assert {"hashlib", "json"} <= imported_modules
    assert imported_modules <= {"hashlib", "json"}
    assert "torch.load" not in source
    assert "pickle" not in source
    assert "yacs" not in source
    assert "timerewarder_repro.conversion" not in source
