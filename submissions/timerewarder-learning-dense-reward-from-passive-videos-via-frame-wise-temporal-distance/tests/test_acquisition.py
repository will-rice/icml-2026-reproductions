import hashlib
import json
import subprocess
from pathlib import Path

import pytest

from timerewarder_repro.acquisition import acquire_inert_sources, verify_acquisition


REVISION = "f54234b67bd3f1fa190f62498d38513a2140f23f"
MODEL_REVISION = "23eded140eb8c8d9f194243a115d218b5072d800"
DATASET_REVISION = "b966abcebc110dd97dd96018e395180e069756c4"
PAPER_REVISION = "arxiv:2509.26627v3"


def source_record(tmp_path: Path, revision: str = REVISION) -> dict[str, object]:
    payload = b"inert source\n"
    return {
        "repository": "owner/repository",
        "url": (tmp_path / "upstream").as_uri(),
        "revision": revision,
        "path": "source.py",
        "git_blob": "8b36ad940f7762a0d2234fc68e31cfce346d2605",
        "sha256": hashlib.sha256(payload).hexdigest(),
        "byte_size": len(payload),
        "license": "MIT",
    }


def expected_commands(
    tmp_path: Path, revision: str = REVISION
) -> list[dict[str, object]]:
    checkout = "<checkout>"
    url = (tmp_path / "upstream").as_uri()
    commands = [
        ["git", "clone", "--no-checkout", url, checkout],
        ["git", "-C", checkout, "checkout", "--detach", revision],
        ["git", "-C", checkout, "rev-parse", "HEAD"],
        ["git", "-C", checkout, "status", "--porcelain"],
        ["git", "-C", checkout, "fsck", "--full"],
        ["git", "-C", checkout, "rev-parse", f"{revision}:source.py"],
        ["git", "-C", checkout, "show", f"{revision}:source.py"],
    ]
    return [{"command": command, "status": 0} for command in commands]


def write_manifest(tmp_path: Path, revision: str = REVISION) -> Path:
    manifest = {
        "paper": {"revision": PAPER_REVISION},
        "model": {"revision": MODEL_REVISION},
        "dataset": {"revision": DATASET_REVISION},
        "sources": [source_record(tmp_path, revision)],
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


def write_receipt(tmp_path: Path, revision: str = REVISION) -> Path:
    receipt = {
        "commands": expected_commands(tmp_path, revision),
        "sources": [source_record(tmp_path, revision)],
    }
    path = tmp_path / "receipt.json"
    path.write_text(json.dumps(receipt), encoding="utf-8")
    return path


def verified_source_fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    source = tmp_path / "source" / "source.py"
    source.parent.mkdir()
    source.write_bytes(b"inert source\n")
    return write_manifest(tmp_path), write_receipt(tmp_path), source


def test_verify_acquisition_rejects_manifest_only_attestation(tmp_path: Path) -> None:
    manifest = write_manifest(tmp_path)
    with pytest.raises(ValueError, match="acquisition receipt"):
        verify_acquisition(manifest, tmp_path / "missing.json", tmp_path / "source")


@pytest.mark.parametrize("revision", ["main", "v1", "f54234b"])
def test_verify_acquisition_rejects_mutable_or_abbreviated_revision(
    tmp_path: Path, revision: str
) -> None:
    manifest = write_manifest(tmp_path, revision=revision)
    with pytest.raises(ValueError, match="full immutable revision"):
        verify_acquisition(manifest, write_receipt(tmp_path), tmp_path / "source")


def test_verify_acquisition_checks_blob_hash_size_and_license(tmp_path: Path) -> None:
    manifest, receipt, source = verified_source_fixture(tmp_path)
    source.write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="sha256"):
        verify_acquisition(manifest, receipt, source.parent)


def test_verify_acquisition_rejects_missing_command_receipt(tmp_path: Path) -> None:
    manifest, receipt, source = verified_source_fixture(tmp_path)
    data = json.loads(receipt.read_text(encoding="utf-8"))
    del data["commands"]
    receipt.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="command receipt"):
        verify_acquisition(manifest, receipt, source.parent)


def test_verify_acquisition_rejects_failed_command_receipt(tmp_path: Path) -> None:
    manifest, receipt, source = verified_source_fixture(tmp_path)
    data = json.loads(receipt.read_text(encoding="utf-8"))
    data["commands"][0]["status"] = 1
    receipt.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="command receipt"):
        verify_acquisition(manifest, receipt, source.parent)


def test_verify_acquisition_rejects_boolean_command_status(tmp_path: Path) -> None:
    manifest, receipt, source = verified_source_fixture(tmp_path)
    data = json.loads(receipt.read_text(encoding="utf-8"))
    data["commands"][0]["status"] = False
    receipt.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="command receipt"):
        verify_acquisition(manifest, receipt, source.parent)


def test_verify_acquisition_rejects_wrong_command_receipt(tmp_path: Path) -> None:
    manifest, receipt, source = verified_source_fixture(tmp_path)
    data = json.loads(receipt.read_text(encoding="utf-8"))
    data["commands"][0]["command"][1] = "fetch"
    receipt.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="command receipt"):
        verify_acquisition(manifest, receipt, source.parent)


def test_verify_acquisition_rejects_extra_command_receipt(tmp_path: Path) -> None:
    manifest, receipt, source = verified_source_fixture(tmp_path)
    data = json.loads(receipt.read_text(encoding="utf-8"))
    data["commands"].append({"command": ["git", "status"], "status": 0})
    receipt.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="command receipt"):
        verify_acquisition(manifest, receipt, source.parent)


def test_verify_acquisition_rejects_duplicate_receipt_path(tmp_path: Path) -> None:
    manifest, receipt, source = verified_source_fixture(tmp_path)
    data = json.loads(receipt.read_text(encoding="utf-8"))
    data["sources"].append(data["sources"][0])
    receipt.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="receipt source paths"):
        verify_acquisition(manifest, receipt, source.parent)


def test_verify_acquisition_rejects_extra_receipt_path(tmp_path: Path) -> None:
    manifest, receipt, source = verified_source_fixture(tmp_path)
    data = json.loads(receipt.read_text(encoding="utf-8"))
    extra = data["sources"][0] | {"path": "extra.py"}
    data["sources"].append(extra)
    receipt.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="receipt source paths"):
        verify_acquisition(manifest, receipt, source.parent)


@pytest.mark.parametrize("name", ["stale.txt", "checkpoint.pth", "video.mp4", "run.py"])
def test_verify_acquisition_rejects_extra_source_file(
    tmp_path: Path, name: str
) -> None:
    manifest, receipt, source = verified_source_fixture(tmp_path)
    extra = source.parent / name
    extra.write_bytes(b"unexpected")
    if name == "run.py":
        extra.chmod(0o755)
    with pytest.raises(ValueError, match="source inventory"):
        verify_acquisition(manifest, receipt, source.parent)


@pytest.mark.parametrize(
    ("field", "value"),
    [("git_blob", "0" * 40), ("byte_size", 999), ("license", "unknown")],
)
def test_verify_acquisition_rejects_altered_source_identity(
    tmp_path: Path, field: str, value: object
) -> None:
    manifest, receipt, source = verified_source_fixture(tmp_path)
    data = json.loads(receipt.read_text(encoding="utf-8"))
    data["sources"][0][field] = value
    receipt.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match=field):
        verify_acquisition(manifest, receipt, source.parent)


@pytest.mark.parametrize(
    ("section", "revision"),
    [
        ("source", "main"),
        ("model", "main"),
        ("dataset", "b966abce"),
        ("paper", "arxiv:2509.26627"),
    ],
)
def test_verify_acquisition_requires_all_immutable_revisions(
    tmp_path: Path, section: str, revision: str
) -> None:
    manifest, receipt, source = verified_source_fixture(tmp_path)
    data = json.loads(manifest.read_text(encoding="utf-8"))
    target = data["sources"][0] if section == "source" else data[section]
    target["revision"] = revision
    manifest.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="full immutable revision"):
        verify_acquisition(manifest, receipt, source.parent)


def test_acquire_inert_sources_rejects_nonempty_output_root(tmp_path: Path) -> None:
    manifest = write_manifest(tmp_path)
    output = tmp_path / "output"
    output.mkdir()
    stale = output / "stale.txt"
    stale.write_bytes(b"stale")
    with pytest.raises(ValueError, match="empty output root"):
        acquire_inert_sources(manifest, output, tmp_path / "receipt.json")
    assert stale.read_bytes() == b"stale"


@pytest.mark.parametrize(
    "path",
    [
        "/absolute.py",
        ".",
        "./source.py",
        "dir/./source.py",
        "../source.py",
        "dir/../source.py",
        "",
        "dir//source.py",
        "dir\\source.py",
    ],
)
def test_acquire_inert_sources_rejects_invalid_manifest_path_before_writing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, path: str
) -> None:
    manifest = write_manifest(tmp_path)
    data = json.loads(manifest.read_text(encoding="utf-8"))
    data["sources"][0]["path"] = path
    manifest.write_text(json.dumps(data), encoding="utf-8")
    output = tmp_path / "output"

    def reject_git(*args: object, **kwargs: object) -> None:
        pytest.fail("Git was invoked before manifest paths were validated")

    monkeypatch.setattr("timerewarder_repro.acquisition.subprocess.run", reject_git)
    with pytest.raises(ValueError, match="manifest source path"):
        acquire_inert_sources(manifest, output, tmp_path / "receipt.json")
    assert not output.exists()


@pytest.mark.parametrize(
    "paths", [["source.py", "source.py"], ["tree", "tree/source.py"]]
)
def test_acquire_inert_sources_rejects_manifest_path_collision_before_writing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, paths: list[str]
) -> None:
    manifest = write_manifest(tmp_path)
    data = json.loads(manifest.read_text(encoding="utf-8"))
    first = data["sources"][0]
    data["sources"] = [first | {"path": path} for path in paths]
    manifest.write_text(json.dumps(data), encoding="utf-8")
    output = tmp_path / "output"

    def reject_git(*args: object, **kwargs: object) -> None:
        pytest.fail("Git was invoked before manifest paths were validated")

    monkeypatch.setattr("timerewarder_repro.acquisition.subprocess.run", reject_git)
    with pytest.raises(ValueError, match="manifest source path"):
        acquire_inert_sources(manifest, output, tmp_path / "receipt.json")
    assert not output.exists()


@pytest.mark.parametrize("name", ["cache", "__pycache__", "empty"])
def test_verify_acquisition_rejects_unexpected_directory(
    tmp_path: Path, name: str
) -> None:
    manifest, receipt, source = verified_source_fixture(tmp_path)
    (source.parent / name).mkdir()
    with pytest.raises(ValueError, match="source inventory"):
        verify_acquisition(manifest, receipt, source.parent)


def test_verify_acquisition_permits_only_exact_source_ancestors(tmp_path: Path) -> None:
    manifest, receipt, source = verified_source_fixture(tmp_path)
    nested_path = "repository/models/source.py"
    nested = source.parent / nested_path
    nested.parent.mkdir(parents=True)
    source.replace(nested)
    manifest_data = json.loads(manifest.read_text(encoding="utf-8"))
    manifest_data["sources"][0].update(
        {"path": nested_path, "upstream_path": "source.py"}
    )
    manifest.write_text(json.dumps(manifest_data), encoding="utf-8")
    receipt_data = json.loads(receipt.read_text(encoding="utf-8"))
    receipt_data["sources"][0].update(
        {"path": nested_path, "upstream_path": "source.py"}
    )
    receipt.write_text(json.dumps(receipt_data), encoding="utf-8")

    assert verify_acquisition(manifest, receipt, source.parent) == tuple(
        receipt_data["sources"]
    )


def test_acquire_inert_sources_extracts_verified_git_blob(tmp_path: Path) -> None:
    upstream = tmp_path / "upstream"
    upstream.mkdir()
    subprocess.run(["git", "init", "-q", str(upstream)], check=True)
    subprocess.run(
        ["git", "-C", str(upstream), "config", "user.email", "test@example.com"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(upstream), "config", "user.name", "Test"], check=True
    )
    payload = b"inert source\n"
    (upstream / "source.py").write_bytes(payload)
    subprocess.run(["git", "-C", str(upstream), "add", "source.py"], check=True)
    subprocess.run(
        ["git", "-C", str(upstream), "commit", "-q", "-m", "fixture"], check=True
    )
    revision = subprocess.run(
        ["git", "-C", str(upstream), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    blob = subprocess.run(
        ["git", "-C", str(upstream), "rev-parse", f"{revision}:source.py"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    manifest = write_manifest(tmp_path, revision=revision)
    data = json.loads(manifest.read_text(encoding="utf-8"))
    data["sources"][0]["git_blob"] = blob
    manifest.write_text(json.dumps(data), encoding="utf-8")
    output = tmp_path / "output"
    receipt_path = tmp_path / "acquisition.json"

    receipt = acquire_inert_sources(manifest, output, receipt_path)

    assert (output / "source.py").read_bytes() == payload
    assert receipt["commands"] and all(
        item["status"] == 0 for item in receipt["commands"]
    )
    assert "timerewarder-acquisition-" not in json.dumps(receipt)
    assert receipt["commands"] == expected_commands(tmp_path, revision)
    assert verify_acquisition(manifest, receipt_path, output) == tuple(
        receipt["sources"]
    )
