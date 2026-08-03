import json
from pathlib import Path
import shutil
import pytest

from lrr_repro.provenance import (
    IntegrityError,
    git_blob_id,
    load_paper_context,
    load_verified_inputs,
    read_manifest,
    validate_manifest,
)


@pytest.fixture
def project_root() -> Path:
    return Path(__file__).resolve().parent.parent


@pytest.fixture
def dummy_cache_dir(tmp_path: Path) -> Path:
    return tmp_path / "cache"


def test_pins_exact_upstreams(project_root, dummy_cache_dir):
    # Dummy cache populated with fake paper PDFs for testing provenance verification structure
    dummy_cache_dir.mkdir(parents=True, exist_ok=True)
    v1_path = dummy_cache_dir / "2412.18134v1.pdf"
    v5_path = dummy_cache_dir / "2412.18134v5.pdf"

    # Fill fake bytes that won't match true sha256 unless downloaded, but test manifest parsing
    items = {item.artifact_id: item for item in load_verified_inputs(project_root, dummy_cache_dir, verify_files=False)}
    assert items["paper-v1"].sha256 == "abaac08eabec2e77c8af7ae3ca028691b9cd862e21bfa779452b9fd729e3222f"
    assert items["paper-v5"].sha256 == "93cab4aa8cec06434b704e639bab87dd15ea95ac46a335961138a94fc1bae2b8"
    assert items["results-csv"].git_blob == "0432241ef42d1be06179546c7b96d6bf6f598986"


def test_tampered_input_fails_closed(project_root, cache_dir, tmp_path):

    # Copy upstream directory to a temp project root
    temp_proj = tmp_path / "proj"
    temp_proj.mkdir(parents=True, exist_ok=True)
    shutil.copytree(project_root / "evidence", temp_proj / "evidence")

    # Tamper with the CSV in temp project
    csv_file = temp_proj / "evidence/inputs/upstream/results/Bitween-Results(Sheet1-ICML).csv"
    csv_file.write_bytes(csv_file.read_bytes() + b"\n")

    with pytest.raises(IntegrityError, match="results-csv"):
        load_verified_inputs(temp_proj, cache_dir, verify_files=True)



def test_manifest_rejects_unsafe_or_duplicate_paths(project_root):
    manifest = read_manifest(project_root)
    manifest_copy = json.loads(json.dumps(manifest))
    manifest_copy["artifacts"][0]["relative_path"] = "../escape"
    with pytest.raises(IntegrityError, match="safe relative path"):
        validate_manifest(manifest_copy)


@pytest.fixture
def set_uv_cache_env(monkeypatch, tmp_path):
    uv_cache = tmp_path / "uv_cache"
    lrr_cache = uv_cache / "lrr-upstream"
    lrr_cache.mkdir(parents=True)
    monkeypatch.setenv("UV_CACHE_DIR", str(uv_cache))
    return lrr_cache


def test_cache_dir_fixture_resolves_uv_cache_dir(set_uv_cache_env, cache_dir):
    assert cache_dir == set_uv_cache_env
