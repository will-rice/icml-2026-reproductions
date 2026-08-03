import hashlib
import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

import timerewarder_repro.cli as cli
from timerewarder_repro.checkpoint import (
    MODEL_REPOSITORY,
    MODEL_REVISION,
    MODEL_SCHEMA_SHA256,
    TASK_CHECKPOINTS,
)
from timerewarder_repro.conversion import ConversionRejected


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_registry(project: Path, selected_payload: bytes = b"selected") -> Path:
    artifacts = project / "artifacts"
    artifacts.mkdir(parents=True)
    schema = artifacts / "model-schema.json"
    schema.write_bytes(b"schema")
    entries = []
    for index, (task, filename) in enumerate(TASK_CHECKPOINTS):
        payload_hash = hashlib.sha256(f"{task}-{index}".encode()).hexdigest()
        size = 1000 + index
        if task == "door-open-v2":
            payload_hash = hashlib.sha256(selected_payload).hexdigest()
            size = len(selected_payload)
        entries.append(
            {
                "task": task,
                "file": filename,
                "repository": MODEL_REPOSITORY,
                "model_revision": MODEL_REVISION,
                "lfs_sha256": payload_hash,
                "size_bytes": size,
                "schema_sha256": MODEL_SCHEMA_SHA256,
                "receipt": f"artifacts/conversion-receipts/{Path(filename).stem}.json",
                "approval": f"artifacts/conversion-approvals/{Path(filename).stem}.json",
            }
        )
    registry = {
        "format": "timerewarder-checkpoint-registry-v1",
        "model": {"repository": MODEL_REPOSITORY, "revision": MODEL_REVISION},
        "schema": {
            "path": "artifacts/model-schema.json",
            "sha256": MODEL_SCHEMA_SHA256,
        },
        "checkpoints": entries,
    }
    path = artifacts / "checkpoints.json"
    path.write_text(json.dumps(registry), encoding="utf-8")
    return path


def _successful_receipt(request: dict[str, object], output: Path) -> dict[str, object]:
    output.write_bytes(b"converted")
    return {
        "format": "timerewarder-conversion-receipt-v2",
        "approval_status": "pending_independent_reviewer",
        "task": request["task"],
        "checkpoint_file": request["checkpoint_file"],
        "checkpoint_sha256": request["checkpoint_sha256"],
        "checkpoint_bytes": request["checkpoint_bytes"],
        "model_repository": request["model_repository"],
        "model_revision": request["model_revision"],
        "converter": request["converter"],
        "status": "success",
        "static_globals": ["yacs.config.CfgNode"],
        "load": {"weights_only": True, "mmap": True, "map_location": "cpu"},
        "safe_globals_empty": True,
        "child_exit": 0,
        "schema_sha256": request["schema_sha256"],
        "output_sha256": _sha256(output),
        "output_bytes": output.stat().st_size,
        "tensor_checks": {"exact_schema": True, "overlapping_storage": False},
        "sandbox": {"network_namespace": "none", "inherited_network_fds": 0},
    }


def test_convert_selects_only_named_registry_entry(tmp_path: Path, monkeypatch) -> None:
    project = tmp_path / "project"
    registry = _write_registry(project)
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "door_open_20bins.pth").write_bytes(b"selected")
    output_dir = tmp_path / "outputs"
    observed = {}
    stale_rejection = project / "artifacts/conversion-rejections/door_open_20bins.json"
    stale_rejection.parent.mkdir(parents=True)
    stale_rejection.write_text('{"failure_category":"outer_sandbox"}\n')

    def fake_convert(request_path: Path) -> dict[str, object]:
        request = json.loads(request_path.read_text(encoding="utf-8"))
        observed.update(request)
        output = Path(request["output_dir"]) / "model.safetensors"
        return _successful_receipt(request, output)

    monkeypatch.setattr(cli, "convert_checkpoint", fake_convert)

    result = cli.convert_registered_checkpoint(
        task="door-open-v2",
        registry_path=registry,
        cache_dir=cache,
        output_dir=output_dir,
        converter="converter-a",
    )

    assert observed["checkpoint_file"] == "door_open_20bins.pth"
    assert observed["checkpoint"] == str(cache / "door_open_20bins.pth")
    assert result["approval_status"] == "pending_independent_reviewer"
    receipt = project / "artifacts/conversion-receipts/door_open_20bins.json"
    assert json.loads(receipt.read_text(encoding="utf-8")) == result
    assert not stale_rejection.exists()
    assert not (
        project / "artifacts/conversion-approvals/door_open_20bins.json"
    ).exists()


@pytest.mark.parametrize("payload", [b"wrong", b"selected-extra"])
def test_convert_rejects_checkpoint_identity_before_child(
    tmp_path: Path, monkeypatch, payload: bytes
) -> None:
    project = tmp_path / "project"
    registry = _write_registry(project)
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "door_open_20bins.pth").write_bytes(payload)
    called = False

    def forbidden_child(request_path: Path) -> dict[str, object]:
        nonlocal called
        called = True
        return {}

    monkeypatch.setattr(cli, "convert_checkpoint", forbidden_child)

    with pytest.raises(ConversionRejected, match="input_identity"):
        cli.convert_registered_checkpoint(
            task="door-open-v2",
            registry_path=registry,
            cache_dir=cache,
            output_dir=tmp_path / "outputs",
            converter="converter-a",
        )

    assert called is False


def test_convert_writes_sanitized_deterministic_rejection(
    tmp_path: Path, monkeypatch
) -> None:
    project = tmp_path / "project"
    registry = _write_registry(project)
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "door_open_20bins.pth").write_bytes(b"selected")

    def reject(request_path: Path) -> dict[str, object]:
        raise ConversionRejected("static_global_set", "/private/host/path")

    monkeypatch.setattr(cli, "convert_checkpoint", reject)

    with pytest.raises(ConversionRejected, match="static_global_set"):
        cli.convert_registered_checkpoint(
            task="door-open-v2",
            registry_path=registry,
            cache_dir=cache,
            output_dir=tmp_path / "outputs",
            converter="converter-a",
        )

    rejection_path = project / "artifacts/conversion-rejections/door_open_20bins.json"
    rejection = json.loads(rejection_path.read_text(encoding="utf-8"))
    assert rejection == {
        "checkpoint": {
            "file": "door_open_20bins.pth",
            "lfs_sha256": hashlib.sha256(b"selected").hexdigest(),
            "model_revision": MODEL_REVISION,
            "repository": MODEL_REPOSITORY,
            "size_bytes": len(b"selected"),
        },
        "failure_category": "static_global_set",
        "format": "timerewarder-conversion-rejection-v1",
        "task": "door-open-v2",
    }
    assert "/private/host/path" not in rejection_path.read_text(encoding="utf-8")


def _write_review_fixture(
    tmp_path: Path,
) -> tuple[Path, Path, Path, Path]:
    project = tmp_path / "project"
    registry = _write_registry(project)
    output = tmp_path / "model.safetensors"
    save_file({"weight": torch.zeros(1)}, output)
    receipt_path = project / "artifacts/conversion-receipts/door_open_20bins.json"
    receipt_path.parent.mkdir(parents=True)
    receipt = {
        "format": "timerewarder-conversion-receipt-v2",
        "approval_status": "pending_independent_reviewer",
        "task": "door-open-v2",
        "checkpoint_file": "door_open_20bins.pth",
        "checkpoint_sha256": hashlib.sha256(b"selected").hexdigest(),
        "checkpoint_bytes": len(b"selected"),
        "model_repository": MODEL_REPOSITORY,
        "model_revision": MODEL_REVISION,
        "converter": "converter-a",
        "status": "success",
        "static_globals": ["yacs.config.CfgNode"],
        "load": {"weights_only": True, "mmap": True, "map_location": "cpu"},
        "safe_globals_empty": True,
        "child_exit": 0,
        "schema_sha256": MODEL_SCHEMA_SHA256,
        "output_sha256": _sha256(output),
        "output_bytes": output.stat().st_size,
        "tensor_checks": {"exact_schema": True, "overlapping_storage": False},
        "sandbox": {"network_namespace": "none", "inherited_network_fds": 0},
    }
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    approval_path = project / "artifacts/conversion-approvals/door_open_20bins.json"
    return registry, receipt_path, output, approval_path


def test_review_command_rejects_converter_as_reviewer(tmp_path: Path) -> None:
    registry, receipt, output, approval = _write_review_fixture(tmp_path)

    with pytest.raises(ValueError, match="reviewer must differ"):
        cli.review_registered_conversion(
            task="door-open-v2",
            registry_path=registry,
            receipt_path=receipt,
            output_path=output,
            reviewer="converter-a",
            approval_path=approval,
        )

    assert not approval.exists()


def test_review_command_writes_approval_only_after_validation(tmp_path: Path) -> None:
    registry, receipt, output, approval = _write_review_fixture(tmp_path)

    result = cli.review_registered_conversion(
        task="door-open-v2",
        registry_path=registry,
        receipt_path=receipt,
        output_path=output,
        reviewer="reviewer-b",
        approval_path=approval,
    )

    assert result["status"] == "approved"
    assert result["reviewer"] == "reviewer-b"
    assert json.loads(approval.read_text(encoding="utf-8")) == result


def test_representative_command_writes_canonical_result(
    tmp_path: Path, monkeypatch
) -> None:
    observed = {}

    def fake_evaluate(registry, dataset_manifest, schema, cache_dir):
        observed.update(
            {
                "registry": registry,
                "dataset_manifest": dataset_manifest,
                "schema": schema,
                "cache_dir": cache_dir,
            }
        )
        return {"format": "timerewarder-representative-v1", "tasks": []}

    monkeypatch.setattr(cli, "evaluate_representative", fake_evaluate)
    output = tmp_path / "representative.json"

    cli.main(
        [
            "representative",
            "--registry",
            str(tmp_path / "registry.json"),
            "--dataset-manifest",
            str(tmp_path / "dataset.json"),
            "--schema",
            str(tmp_path / "schema.json"),
            "--cache-dir",
            str(tmp_path / "cache"),
            "--output",
            str(output),
        ]
    )

    assert observed["cache_dir"] == tmp_path / "cache"
    assert output.read_text(encoding="utf-8") == (
        '{"format":"timerewarder-representative-v1","tasks":[]}\n'
    )


def test_build_evidence_and_fixture_commands_write_canonical_json(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(
        cli,
        "build_evidence_bundle",
        lambda *args: {"measurement_sha256": "a" * 64},
    )
    monkeypatch.setattr(cli, "run_fixture", lambda: {"diagnostic_only": True})
    evidence = tmp_path / "evidence.json"
    fixture = tmp_path / "fixture.json"

    cli.main(
        [
            "build-evidence",
            "--manifest",
            str(tmp_path / "manifest"),
            "--acquisition",
            str(tmp_path / "acquisition"),
            "--registry",
            str(tmp_path / "registry"),
            "--source-root",
            str(tmp_path / "source"),
            "--representative",
            str(tmp_path / "representative"),
            "--output",
            str(evidence),
        ]
    )
    cli.main(["fixture", "--output", str(fixture)])

    assert evidence.read_text() == '{"measurement_sha256":"' + "a" * 64 + '"}\n'
    assert fixture.read_text() == '{"diagnostic_only":true}\n'
