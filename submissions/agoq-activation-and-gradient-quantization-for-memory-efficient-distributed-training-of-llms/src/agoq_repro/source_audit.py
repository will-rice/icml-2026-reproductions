"""Semantic audit of hash-verified released AGoQ source files."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from .provenance import load_verified_sources


class SemanticAuditError(ValueError):
    """Pinned source bytes do not contain a required semantic relationship."""


@dataclass(frozen=True)
class SourceObservation:
    observation_id: str
    disposition: Literal["verified", "partial", "absent"]
    files: tuple[str, ...]
    symbol_names: tuple[str, ...]
    detail: str


@dataclass(frozen=True)
class _Facts:
    imports: frozenset[tuple[str, str]]
    calls: frozenset[str]
    definitions: frozenset[str]


def _qualified_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = _qualified_name(node.value)
        return f"{parent}.{node.attr}" if parent else node.attr
    return None


def _facts(path: Path) -> _Facts:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=path.name)
    except (OSError, UnicodeError, SyntaxError) as exc:
        raise SemanticAuditError(f"cannot parse {path.name}: {exc}") from exc
    imports: set[tuple[str, str]] = set()
    calls: set[str] = set()
    definitions: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            module = node.module or ""
            imports.update((module, item.name) for item in node.names)
        elif isinstance(node, ast.Import):
            imports.update((item.name, item.asname or item.name) for item in node.names)
        elif isinstance(node, ast.Call):
            name = _qualified_name(node.func)
            if name:
                calls.add(name)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            definitions.add(node.name)
    return _Facts(frozenset(imports), frozenset(calls), frozenset(definitions))


def _has_suffix(values: frozenset[str], suffix: str) -> bool:
    return any(value == suffix or value.endswith(f".{suffix}") for value in values)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SemanticAuditError(message)


def audit_released_source(project_root: Path) -> tuple[SourceObservation, ...]:
    verified = load_verified_sources(project_root)
    verified_paths = {item.path for item in verified}
    source_root = project_root / "evidence/inputs/upstream"
    required_paths = {
        "megatron/core/tensor_parallel/layers.py",
        "megatron/core/quantizer/activation_quantization.py",
        "megatron/core/distributed/distributed_data_parallel.py",
        "megatron/core/distributed/param_and_grad_buffer.py",
        "megatron/core/pipeline_parallel/schedules.py",
        "changes_te/linear.py",
        "changes_te/layernorm_linear.py",
        "changes_te/layernorm_mlp.py",
    }
    _require(required_paths <= verified_paths, "required source files are not pinned")
    facts = {
        relative: _facts(source_root / relative) for relative in sorted(required_paths)
    }

    layers_path = "megatron/core/tensor_parallel/layers.py"
    quantizer_path = "megatron/core/quantizer/activation_quantization.py"
    layers = facts[layers_path]
    quantizer = facts[quantizer_path]
    for name in ("activation_quantize", "activation_dequantize"):
        _require(
            any(symbol == name for _, symbol in layers.imports)
            and _has_suffix(layers.calls, name),
            f"layers.py is missing {name} integration",
        )
    for name in ("op_quantize", "op_dequantize"):
        _require(
            any(
                module == "gact.ops" and symbol == name
                for module, symbol in quantizer.imports
            )
            and _has_suffix(quantizer.calls, name),
            f"activation_quantization.py is missing {name}",
        )

    ddp_path = "megatron/core/distributed/distributed_data_parallel.py"
    ddp = facts[ddp_path]
    _require("add_to_8bit" in ddp.definitions, "missing local add_to_8bit")
    for name in ("dequantize_blockwise", "quantize_blockwise", "add_"):
        _require(
            _has_suffix(ddp.calls, name),
            f"local gradient accumulation is missing {name}",
        )

    collective_path = "megatron/core/distributed/param_and_grad_buffer.py"
    collective = facts[collective_path]
    _require("a2a_ag" in collective.definitions, "missing a2a_ag")
    for name in (
        "all_to_all_single",
        "dequantize_blockwise",
        "quantize_blockwise",
        "_all_gather_base",
    ):
        _require(
            _has_suffix(collective.calls, name),
            f"quantized collective path is missing {name}",
        )

    te_paths = (
        "changes_te/linear.py",
        "changes_te/layernorm_linear.py",
        "changes_te/layernorm_mlp.py",
    )
    for relative in te_paths:
        item = facts[relative]
        for name in ("op_quantize", "op_dequantize"):
            _require(
                any(
                    module == "gact.ops" and symbol == name
                    for module, symbol in item.imports
                )
                and _has_suffix(item.calls, name),
                f"{relative} is missing gact.ops {name} call site",
            )
        _require(
            _has_suffix(item.calls, "gemm") or _has_suffix(item.calls, "fp8_gemm"),
            f"{relative} is missing adjacent GEMM context",
        )

    schedule_path = "megatron/core/pipeline_parallel/schedules.py"
    schedule = facts[schedule_path]
    _require(
        "forward_backward_pipelining_with_interleaving" in schedule.definitions,
        "pipeline schedule context is missing",
    )

    rows = (
        SourceObservation(
            "activation_quantization_integration",
            "verified",
            (layers_path, quantizer_path),
            (
                "activation_quantize",
                "activation_dequantize",
                "op_quantize",
                "op_dequantize",
            ),
            "Megatron tensor-parallel layers call the pinned activation quantizer, "
            "whose gact path calls op_quantize and op_dequantize.",
        ),
        SourceObservation(
            "all_to_all_reduce_all_gather_path",
            "verified",
            (collective_path,),
            (
                "a2a_ag",
                "all_to_all_single",
                "dequantize_blockwise",
                "quantize_blockwise",
                "_all_gather_base",
            ),
            "The pinned gradient path performs All-to-All, local dequantize/reduce/"
            "requantize, and AllGather operations.",
        ),
        SourceObservation(
            "local_gradient_accumulation",
            "verified",
            (ddp_path,),
            ("add_to_8bit", "dequantize_blockwise", "add_", "quantize_blockwise"),
            "The pinned DDP path dequantizes, accumulates locally, and requantizes.",
        ),
        SourceObservation(
            "pipeline_schedule_context",
            "partial",
            (schedule_path,),
            ("forward_backward_pipelining_with_interleaving",),
            "The release contains interleaved pipeline scheduling context but no "
            "source implementation of the paper's discrete bit-allocation rule.",
        ),
        SourceObservation(
            "single_gpu_fused_kernel_body",
            "absent",
            te_paths,
            ("op_quantize", "op_dequantize", "gemm", "fp8_gemm"),
            "Pinned Transformer Engine changes contain quantize/dequantize call sites "
            "next to GEMM calls, but no gact kernel body proving a single fused kernel.",
        ),
    )
    return tuple(sorted(rows, key=lambda row: row.observation_id))
