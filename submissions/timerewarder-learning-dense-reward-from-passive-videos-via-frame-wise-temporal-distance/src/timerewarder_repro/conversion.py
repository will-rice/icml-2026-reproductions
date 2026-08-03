"""Fail-closed conversion boundary for legacy TimeRewarder checkpoints."""

import hashlib
import json
import os
import resource
import shutil
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from timerewarder_repro.approval import validate_approval_record

APPROVED_UNSAFE_GLOBALS = ("yacs.config.CfgNode",)
_LOAD_ARGUMENTS = {"weights_only": True, "mmap": True, "map_location": "cpu"}


class ConversionRejected(ValueError):
    """A conversion boundary or approval requirement was not met."""

    def __init__(self, gate: str, detail: str = "") -> None:
        super().__init__(f"{gate}: {detail}" if detail else gate)
        self.gate = gate


def inspect_checkpoint(path: Path) -> tuple[str, ...]:
    """Allow only the independently reviewed static global set before loading."""
    observed = tuple(sorted(torch.serialization.get_unsafe_globals_in_checkpoint(path)))
    if observed != APPROVED_UNSAFE_GLOBALS:
        raise ConversionRejected("static_global_set", repr(observed))
    return observed


def extract_and_validate_state_dict(
    loaded: object, schema: dict[str, object]
) -> dict[str, torch.Tensor]:
    """Copy only exact, finite, plain tensors matching the reviewed schema."""
    expected = _validate_schema(schema)
    if not isinstance(loaded, Mapping) or set(loaded) != set(schema["top_level_keys"]):
        raise ConversionRejected("top_level_layout")
    for name, expected_type in schema["top_level_types"].items():
        if _qualified_type(loaded[name]) != expected_type:
            raise ConversionRejected("top_level_metadata", name)
    model = loaded.get("model")
    if not isinstance(model, Mapping) or not all(
        isinstance(name, str) for name in model
    ):
        raise ConversionRejected("model_state_dict")
    if set(model) != set(expected):
        raise ConversionRejected("tensor_key_set")

    total_bytes = 0
    storage_pointers: set[int] = set()
    result: dict[str, torch.Tensor] = {}
    for name in sorted(expected):
        tensor = model[name]
        specification = expected[name]
        if type(tensor) is not torch.Tensor:
            raise ConversionRejected("tensor_type", name)
        if (
            tensor.device.type != "cpu"
            or tensor.layout is not torch.strided
            or tensor.is_sparse
            or tensor.is_quantized
            or not tensor.is_contiguous()
        ):
            raise ConversionRejected("tensor_layout", name)
        pointer = tensor.untyped_storage().data_ptr()
        if pointer in storage_pointers:
            raise ConversionRejected("overlapping_storage", name)
        storage_pointers.add(pointer)
        dtype = str(tensor.dtype).removeprefix("torch.")
        byte_size = tensor.numel() * tensor.element_size()
        if (
            list(tensor.shape) != specification["shape"]
            or dtype != specification["dtype"]
            or byte_size != specification["byte_size"]
        ):
            raise ConversionRejected("tensor_schema", name)
        if tensor.is_floating_point() and not torch.isfinite(tensor).all().item():
            raise ConversionRejected("nonfinite_tensor", name)
        total_bytes += byte_size
        result[name] = tensor.detach().contiguous().clone()
    if (
        not schema["minimum_tensor_bytes"]
        <= total_bytes
        <= schema["maximum_tensor_bytes"]
    ):
        raise ConversionRejected("aggregate_tensor_bytes")
    return result


def child_convert(
    checkpoint: Path, output: Path, schema: dict[str, object]
) -> dict[str, object]:
    """Load one checkpoint only in the isolated converter process."""
    _validate_schema(schema)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".tmp")
    loaded = None
    tensors = None
    torch.serialization.clear_safe_globals()
    try:
        resource.setrlimit(resource.RLIMIT_AS, (16 * 1024**3, 16 * 1024**3))
        resource.setrlimit(resource.RLIMIT_CPU, (600, 600))
        resource.setrlimit(
            resource.RLIMIT_FSIZE,
            (int(schema["maximum_output_bytes"]), int(schema["maximum_output_bytes"])),
        )
        torch.set_num_threads(2)
        from yacs.config import CfgNode

        with torch.serialization.safe_globals([CfgNode]):
            loaded = torch.load(
                checkpoint, weights_only=True, mmap=True, map_location="cpu"
            )
            tensors = extract_and_validate_state_dict(loaded, schema)
            save_file(tensors, temporary)
            if temporary.stat().st_size > schema["maximum_output_bytes"]:
                raise ConversionRejected("output_size")
            os.replace(temporary, output)
        return {
            "output_sha256": _sha256_file(output),
            "output_bytes": output.stat().st_size,
            "safe_globals_empty": True,
        }
    finally:
        torch.serialization.clear_safe_globals()
        temporary.unlink(missing_ok=True)
        del tensors, loaded
        if torch.serialization.get_safe_globals():
            raise ConversionRejected("safe_globals_not_empty")


def convert_checkpoint(request_path: Path) -> dict[str, object]:
    """Run a verified conversion request only inside a bubblewrap sandbox."""
    if shutil.which("bwrap") is None or shutil.which("timeout") is None:
        raise ConversionRejected("sandbox_unavailable")
    request = _read_json_mapping(request_path, "conversion_request")
    checkpoint = _request_path(request, "checkpoint")
    schema_path = _request_path(request, "schema")
    output_dir = _request_path(request, "output_dir")
    runtime = _request_path(request, "runtime")
    package_root = _request_path(request, "package_root")
    python_root = _request_path(request, "python_root")
    converter = request.get("converter")
    if not isinstance(converter, str) or not converter:
        raise ConversionRejected("converter_identity")
    task = _request_text(request, "task")
    checkpoint_file = _request_text(request, "checkpoint_file")
    model_repository = _request_text(request, "model_repository")
    model_revision = _request_text(request, "model_revision")
    if checkpoint_file != checkpoint.name:
        raise ConversionRejected("conversion_request", "checkpoint_file")
    if len(model_revision) != 40 or any(
        character not in "0123456789abcdef" for character in model_revision
    ):
        raise ConversionRejected("conversion_request", "model_revision")
    _verify_input_identity(checkpoint, request, "checkpoint")
    _verify_input_identity(schema_path, request, "schema")
    schema = _read_json_mapping(schema_path, "model_schema")
    _validate_schema(schema)
    static_globals = inspect_checkpoint(checkpoint)
    if (
        not (runtime / "bin" / "python").is_file()
        or not package_root.is_dir()
        or not (python_root / "bin" / "python3.12").is_file()
        or not output_dir.is_dir()
    ):
        raise ConversionRejected("sandbox_runtime")
    output = output_dir / "model.safetensors"
    command = _sandbox_command(
        checkpoint=checkpoint,
        schema_path=schema_path,
        output_dir=output_dir,
        runtime=runtime,
        package_root=package_root,
        python_root=python_root,
    )
    completed = subprocess.run(
        command,
        close_fds=True,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=610,
        check=False,
    )
    if completed.returncode != 0:
        raise ConversionRejected("sandbox_child_exit", completed.stderr.strip())
    child = _parse_child_result(completed.stdout)
    if not output.is_file() or child["output_sha256"] != _sha256_file(output):
        raise ConversionRejected("output_identity")
    return {
        "format": "timerewarder-conversion-receipt-v2",
        "approval_status": "pending_independent_reviewer",
        "task": task,
        "checkpoint_file": checkpoint_file,
        "checkpoint_sha256": request["checkpoint_sha256"],
        "checkpoint_bytes": request["checkpoint_bytes"],
        "model_repository": model_repository,
        "model_revision": model_revision,
        "converter": converter,
        "status": "success",
        "static_globals": list(static_globals),
        "load": _LOAD_ARGUMENTS,
        "safe_globals_empty": child["safe_globals_empty"],
        "child_exit": completed.returncode,
        "schema_sha256": _sha256_file(schema_path),
        "output_sha256": child["output_sha256"],
        "output_bytes": child["output_bytes"],
        "tensor_checks": {"exact_schema": True, "overlapping_storage": False},
        "sandbox": {"network_namespace": "none", "inherited_network_fds": 0},
    }


def _sandbox_command(
    *,
    checkpoint: Path,
    schema_path: Path,
    output_dir: Path,
    runtime: Path,
    package_root: Path,
    python_root: Path,
) -> list[str]:
    return [
        "timeout",
        "--signal=KILL",
        "600",
        "bwrap",
        "--die-with-parent",
        "--unshare-all",
        "--new-session",
        "--ro-bind",
        str(checkpoint),
        "/input/checkpoint.pth",
        "--ro-bind",
        str(schema_path),
        "/input/model-schema.json",
        "--ro-bind",
        str(runtime),
        "/runtime",
        "--ro-bind",
        str(package_root),
        "/package",
        "--ro-bind",
        str(python_root),
        "/python",
        "--ro-bind",
        "/lib",
        "/lib",
        "--ro-bind",
        "/lib64",
        "/lib64",
        "--ro-bind",
        "/usr/lib",
        "/usr/lib",
        "--bind",
        str(output_dir),
        "/output",
        "--tmpfs",
        "/tmp",
        "--proc",
        "/proc",
        "--dev",
        "/dev",
        "--setenv",
        "PYTHONPATH",
        "/package:/runtime/lib/python3.12/site-packages",
        "--setenv",
        "PYTHONHOME",
        "/python",
        "/python/bin/python3.12",
        "-m",
        "timerewarder_repro.conversion",
        "child",
        "/input/checkpoint.pth",
        "/input/model-schema.json",
        "/output/model.safetensors",
    ]


def approve_conversion(
    receipt_path: Path, reviewer: str, output_path: Path
) -> dict[str, object]:
    """Create a content-addressed approval for one independently reviewed output."""
    receipt = _read_json_mapping(receipt_path, "conversion_receipt")
    _validate_receipt(receipt, output_path)
    converter = receipt["converter"]
    if not isinstance(reviewer, str) or not reviewer or reviewer == converter:
        raise ValueError("reviewer must differ from converter")
    approval = {
        "format": "timerewarder-conversion-approval-v1",
        "status": "approved",
        "receipt": receipt,
        "receipt_sha256": _canonical_sha256(receipt),
        "converter": converter,
        "reviewer": reviewer,
        "checkpoint_sha256": receipt["checkpoint_sha256"],
        "checkpoint_bytes": receipt["checkpoint_bytes"],
        "model_revision": receipt["model_revision"],
        "output_sha256": receipt["output_sha256"],
        "output_bytes": receipt["output_bytes"],
        "schema_sha256": receipt["schema_sha256"],
    }
    result = approval | {"approval_sha256": _canonical_sha256(approval)}
    validate_approval_record(
        result,
        receipt=receipt,
        output_path=output_path,
        expected_schema_sha256=receipt["schema_sha256"],
    )
    return result


def validate_approval(approval: dict[str, object], output_path: Path) -> None:
    """Require an untampered approval and its exact tensor-only output bytes."""
    receipt = approval.get("receipt")
    if not isinstance(receipt, dict):
        raise ValueError("approval receipt")
    schema_sha256 = receipt.get("schema_sha256")
    if not isinstance(schema_sha256, str):
        raise ValueError("approval schema identity")
    _validate_receipt(receipt, output_path)
    validate_approval_record(
        approval,
        receipt=receipt,
        output_path=output_path,
        expected_schema_sha256=schema_sha256,
    )


def _validate_schema(schema: dict[str, object]) -> dict[str, dict[str, object]]:
    tensors = schema.get("tensors")
    required = {
        "top_level_keys",
        "top_level_types",
        "minimum_tensor_bytes",
        "maximum_tensor_bytes",
        "maximum_output_bytes",
    }
    if not isinstance(tensors, dict) or not required <= set(schema):
        raise ConversionRejected("model_schema")
    if (
        not isinstance(schema["top_level_keys"], list)
        or "model" not in schema["top_level_keys"]
        or not isinstance(schema["top_level_types"], dict)
        or set(schema["top_level_types"]) != set(schema["top_level_keys"])
        or not all(
            isinstance(name, str) and isinstance(type_name, str)
            for name, type_name in schema["top_level_types"].items()
        )
    ):
        raise ConversionRejected("model_schema")
    for field in required - {"top_level_keys", "top_level_types"}:
        if (
            not isinstance(schema[field], int)
            or isinstance(schema[field], bool)
            or schema[field] < 0
        ):
            raise ConversionRejected("model_schema")
    if schema["minimum_tensor_bytes"] > schema["maximum_tensor_bytes"]:
        raise ConversionRejected("model_schema")
    result: dict[str, dict[str, object]] = {}
    for name, specification in tensors.items():
        if not isinstance(name, str) or not isinstance(specification, dict):
            raise ConversionRejected("model_schema")
        shape = specification.get("shape")
        dtype = specification.get("dtype")
        byte_size = specification.get("byte_size")
        if (
            not isinstance(shape, list)
            or not all(isinstance(size, int) and size >= 0 for size in shape)
            or not isinstance(dtype, str)
            or not isinstance(byte_size, int)
            or byte_size < 0
        ):
            raise ConversionRejected("model_schema")
        result[name] = specification
    return result


def _validate_receipt(receipt: dict[str, object], output_path: Path) -> None:
    required = {
        "format",
        "approval_status",
        "task",
        "checkpoint_file",
        "checkpoint_sha256",
        "checkpoint_bytes",
        "model_repository",
        "model_revision",
        "converter",
        "status",
        "static_globals",
        "load",
        "safe_globals_empty",
        "child_exit",
        "schema_sha256",
        "output_sha256",
        "output_bytes",
        "tensor_checks",
        "sandbox",
    }
    if (
        not required <= set(receipt)
        or receipt["format"] != "timerewarder-conversion-receipt-v2"
        or receipt["approval_status"] != "pending_independent_reviewer"
        or receipt["status"] != "success"
    ):
        raise ValueError("conversion receipt gates")
    if receipt["static_globals"] != list(APPROVED_UNSAFE_GLOBALS):
        raise ValueError("conversion receipt globals")
    if receipt["load"] != _LOAD_ARGUMENTS or receipt["safe_globals_empty"] is not True:
        raise ValueError("conversion receipt load policy")
    if receipt["child_exit"] != 0 or receipt["tensor_checks"] != {
        "exact_schema": True,
        "overlapping_storage": False,
    }:
        raise ValueError("conversion receipt tensor checks")
    if receipt["sandbox"] != {"network_namespace": "none", "inherited_network_fds": 0}:
        raise ValueError("conversion receipt sandbox")
    if not isinstance(receipt["converter"], str) or not receipt["converter"]:
        raise ValueError("conversion receipt converter")
    if receipt["output_bytes"] != output_path.stat().st_size or receipt[
        "output_sha256"
    ] != _sha256_file(output_path):
        raise ValueError("safetensors hash")
    try:
        with safe_open(output_path, framework="pt", device="cpu") as tensors:
            tuple(tensors.keys())
    except Exception as error:
        raise ValueError("safetensors container") from error


def _read_json_mapping(path: Path, gate: str) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ConversionRejected(gate, type(error).__name__) from error
    if not isinstance(value, dict):
        raise ConversionRejected(gate)
    return value


def _request_path(request: dict[str, object], name: str) -> Path:
    value = request.get(name)
    if not isinstance(value, str) or not value:
        raise ConversionRejected("conversion_request", name)
    return Path(value)


def _request_text(request: dict[str, object], name: str) -> str:
    value = request.get(name)
    if not isinstance(value, str) or not value:
        raise ConversionRejected("conversion_request", name)
    return value


def _verify_input_identity(
    checkpoint: Path, request: dict[str, object], name: str
) -> None:
    expected_hash = request.get(f"{name}_sha256")
    expected_bytes = request.get(f"{name}_bytes")
    if (
        not checkpoint.is_file()
        or checkpoint.is_symlink()
        or not isinstance(expected_hash, str)
        or expected_hash != _sha256_file(checkpoint)
        or not isinstance(expected_bytes, int)
        or expected_bytes != checkpoint.stat().st_size
    ):
        raise ConversionRejected("input_identity", name)


def _parse_child_result(stdout: str) -> dict[str, object]:
    try:
        value = json.loads(stdout)
    except json.JSONDecodeError as error:
        raise ConversionRejected("child_receipt", type(error).__name__) from error
    if not isinstance(value, dict) or not {
        "output_sha256",
        "output_bytes",
        "safe_globals_empty",
    } <= set(value):
        raise ConversionRejected("child_receipt")
    return value


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _qualified_type(value: object) -> str:
    value_type = type(value)
    return f"{value_type.__module__}.{value_type.__qualname__}"


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _child_main(arguments: list[str]) -> None:
    if len(arguments) != 3:
        raise ConversionRejected("child_arguments")
    checkpoint, schema_path, output = map(Path, arguments)
    schema = _read_json_mapping(schema_path, "model_schema")
    print(json.dumps(child_convert(checkpoint, output, schema), sort_keys=True))


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] != "child":
        raise SystemExit(
            "usage: python -m timerewarder_repro.conversion child CHECKPOINT SCHEMA OUTPUT"
        )
    _child_main(sys.argv[2:])
