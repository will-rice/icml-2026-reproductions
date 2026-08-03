import hashlib
import json
from shutil import copytree

import pytest

from agoq_repro.provenance import IntegrityError
from agoq_repro.source_audit import SemanticAuditError, audit_released_source


def test_activation_quantization_trace_is_source_bound(project_root):
    rows = {row.observation_id: row for row in audit_released_source(project_root)}
    row = rows["activation_quantization_integration"]
    assert row.disposition == "verified"
    assert row.files == (
        "megatron/core/tensor_parallel/layers.py",
        "megatron/core/quantizer/activation_quantization.py",
    )
    assert {"op_quantize", "op_dequantize"} <= set(row.symbol_names)


def test_gradient_collective_trace_is_source_bound(project_root):
    rows = {row.observation_id: row for row in audit_released_source(project_root)}
    assert rows["local_gradient_accumulation"].disposition == "verified"
    assert rows["all_to_all_reduce_all_gather_path"].disposition == "verified"
    assert rows["all_to_all_reduce_all_gather_path"].files == (
        "megatron/core/distributed/param_and_grad_buffer.py",
    )


def test_fused_kernel_body_is_not_claimed(project_root):
    rows = {row.observation_id: row for row in audit_released_source(project_root)}
    row = rows["single_gpu_fused_kernel_body"]
    assert row.disposition == "absent"
    assert row.files == (
        "changes_te/linear.py",
        "changes_te/layernorm_linear.py",
        "changes_te/layernorm_mlp.py",
    )
    assert "call sites" in row.detail
    assert "kernel body" in row.detail


def test_byte_mutation_fails_before_semantic_audit(project_root, tmp_path):
    copied = copytree(project_root, tmp_path / "project")
    target = copied / "evidence/inputs/upstream/changes_te/linear.py"
    target.write_bytes(target.read_bytes().replace(b"op_quantize", b"op_quantizf", 1))
    with pytest.raises(IntegrityError, match="SHA-256"):
        audit_released_source(copied)


def test_verified_semantic_mutation_is_rejected(project_root, tmp_path):
    copied = copytree(project_root, tmp_path / "project")
    relative = "changes_te/linear.py"
    target = copied / "evidence/inputs/upstream" / relative
    payload = target.read_bytes().replace(b"op_quantize", b"removed_quantize")
    target.write_bytes(payload)
    manifest_path = copied / "evidence/inputs/upstream_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    entry = next(item for item in manifest["files"] if item["path"] == relative)
    entry["size_bytes"] = len(payload)
    entry["sha256"] = hashlib.sha256(payload).hexdigest()
    entry["git_blob"] = hashlib.sha1(
        f"blob {len(payload)}\0".encode() + payload
    ).hexdigest()
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(SemanticAuditError, match="op_quantize"):
        audit_released_source(copied)
