import json
from shutil import copytree

import pytest

from agoq_repro.provenance import (
    IntegrityError,
    load_verified_sources,
    load_verified_transcription,
)


def test_all_pinned_sources_verify(project_root):
    files = load_verified_sources(project_root)
    assert len(files) == 10
    assert {item.path for item in files} >= {
        "LICENSE",
        "megatron/core/tensor_parallel/layers.py",
        "megatron/core/distributed/param_and_grad_buffer.py",
    }


def test_modified_source_is_rejected(project_root, tmp_path):
    copied = copytree(project_root, tmp_path / "project")
    target = copied / "evidence/inputs/upstream/changes_te/linear.py"
    target.write_bytes(target.read_bytes() + b"\n")
    with pytest.raises(IntegrityError, match="SHA-256"):
        load_verified_sources(copied)


def test_transcription_is_bound_to_paper_hash(project_root):
    data = load_verified_transcription(project_root)
    assert data["paper"]["arxiv_id"] == "2605.00539v2"
    assert data["paper"]["pdf_sha256"] == (
        "6a5095edf64e730a824fc076a0cbf3d97922b370dc827f173e872e17eb95e0d7"
    )


def test_manifest_rejects_duplicate_keys(project_root, tmp_path):
    copied = copytree(project_root, tmp_path / "project")
    manifest = copied / "evidence/inputs/upstream_manifest.json"
    value = json.loads(manifest.read_text())
    manifest.write_text(
        '{"schema_version":1,"schema_version":1,'
        f'"repository":{json.dumps(value["repository"])},'
        f'"commit":{json.dumps(value["commit"])},'
        f'"license_file":"LICENSE","files":[]}}'
    )
    with pytest.raises(IntegrityError, match="duplicate JSON key"):
        load_verified_sources(copied)
