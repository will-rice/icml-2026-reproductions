import hashlib
import json
from pathlib import Path
import pytest
from reward_free_alignment.provenance import (
    load_live_claims,
    load_manifest,
    load_verified_artifacts,
    IntegrityError,
    _git_blob_id,
)

EXPECTED_HASHES = (
    "e9a35e34b57a7273bf84d3d5981ab19f8ff1088adef8363f4640dcf436183944",
    "7c0aa54e034d03f2d0905417a024af4db41338cd9a803a0b42e441945c307cf9",
    "85abbc8a21d5c4537409f6e9f2af6bffc7e4c15e2311dfa078bf816ea0cffc9e",
    "dac93f364ac0469302894920781b034bfcd205816fbe16537c2f8e7c10d8995d",
    "269d8a5053e224206036399bccb2435455565149086de6439a9046d89682772b",
    "0d457572ea8a502fa8a489fef3e15da21b13cc39dd3a3730843d1cbe833059b0",
    "50719d645042a500f9c4d53fbdfedf719ee56429ddb73a231912f1eaeadb1b31",
    "5ec835ce150ff60d1e2bbd4fbdf7d1ebacf91bb2b6b2d65f72c44c3b3ed65229",
    "b74a0ea75967144b210934fd40fd23449d3ef985df878d5a9e14c4b04025ba4b",
    "58b31f527bb5e1bccb05c0dab775a74c2f2bdcd8e92ef2c0dd578733b5fb058e",
)


@pytest.fixture
def project_root() -> Path:
    return Path(__file__).parent.parent


def test_live_claims_match_admitted_snapshot(project_root):
    claims = load_live_claims(project_root / "evidence/inputs/live_claims.json")
    assert [claim.ordinal for claim in claims] == list(range(1, 11))
    assert tuple(claim.sha256 for claim in claims) == EXPECTED_HASHES
    assert [claim.targeted for claim in claims] == [
        False, False, False, False, False, True, True, True, True, False
    ]
    for claim in claims:
        assert hashlib.sha256(claim.text.encode("utf-8")).hexdigest() == claim.sha256


def test_manifest_binds_attempt_snapshot_and_upstream(project_root):
    manifest = load_manifest(project_root)
    assert manifest["attempt_id"] == "97e213a5-7ca3-4a1b-a500-1ec52d94d87a"
    assert manifest["paper_id"] == "vSzRJyg6k0"
    assert manifest["snapshot_id"] == (
        "09017559ff2c5746f1a37458ba9a330bd4e18654ae9c3f873bb0785c76626199"
    )
    assert manifest["upstream_revision"] == (
        "arxiv:2602.02495v3+"
        "github:PeterLauLukChen/RACO@84a943c34f38520c7e0c9dd3066517c111b3c8fa"
    )


def test_verified_artifacts_accepts_empty_list(project_root):
    """Empty artifacts list is valid (no upstream files pinned yet)."""
    artifacts = load_verified_artifacts(project_root)
    assert artifacts == ()


# --- Adversarial regressions for fail-closed provenance ---


def test_duplicate_json_keys_rejected(project_root, tmp_path):
    """Duplicate JSON keys in live_claims.json must be rejected."""
    dup_json = tmp_path / "dup_claims.json"
    # Create a JSON file with a duplicate key inside an object
    dup_json.write_text(
        '[{"ordinal": 1, "text": "a", "sha256": "x", "targeted": false, "ordinal": 2}]',
        encoding="utf-8",
    )
    with pytest.raises(IntegrityError, match="[Dd]uplicate"):
        load_live_claims(dup_json)


def test_extra_manifest_keys_rejected(tmp_path):
    """Extra keys in the manifest must be rejected (fail-closed)."""
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({
        "attempt_id": "test",
        "paper_id": "test",
        "snapshot_id": "test",
        "upstream_revision": "test",
        "artifacts": [],
        "sneaky_extra_key": "should be rejected",
    }), encoding="utf-8")
    with pytest.raises(IntegrityError, match="[Ee]xtra"):
        load_manifest(path=manifest)


def test_duplicate_artifact_ids_rejected(tmp_path):
    """Duplicate artifact IDs must be rejected."""
    # Create a manifest with duplicate artifact IDs
    test_file = tmp_path / "test.txt"
    test_file.write_text("hello")
    file_hash = hashlib.sha256(b"hello").hexdigest()

    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({
        "attempt_id": "test",
        "paper_id": "test",
        "snapshot_id": "test",
        "upstream_revision": "test",
        "artifacts": [
            {"artifact_id": "dup", "relative_path": "test.txt",
             "sha256": file_hash, "size_bytes": 5},
            {"artifact_id": "dup", "relative_path": "test2.txt",
             "sha256": file_hash, "size_bytes": 5},
        ],
    }), encoding="utf-8")
    with pytest.raises(IntegrityError, match="[Dd]uplicate"):
        load_verified_artifacts(tmp_path)


def test_empty_artifact_entry_rejected(tmp_path):
    """Empty artifact entries must be rejected."""
    manifest = tmp_path / "evidence" / "inputs" / "upstream_manifest.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(json.dumps({
        "attempt_id": "test",
        "paper_id": "test",
        "snapshot_id": "test",
        "upstream_revision": "test",
        "artifacts": [{}],
    }), encoding="utf-8")
    with pytest.raises(IntegrityError):
        load_verified_artifacts(tmp_path)


def test_git_blob_drift_rejected(tmp_path):
    """Git blob ID must be recomputed and verified; drift is rejected."""
    test_file = tmp_path / "test.txt"
    test_file.write_text("hello")
    file_hash = hashlib.sha256(b"hello").hexdigest()

    manifest = tmp_path / "evidence" / "inputs" / "upstream_manifest.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(json.dumps({
        "attempt_id": "test",
        "paper_id": "test",
        "snapshot_id": "test",
        "upstream_revision": "test",
        "artifacts": [
            {"artifact_id": "a1", "relative_path": "test.txt",
             "sha256": file_hash, "size_bytes": 5,
             "git_blob": "0000000000000000000000000000000000000000"},
        ],
    }), encoding="utf-8")
    with pytest.raises(IntegrityError, match="[Gg]it blob"):
        load_verified_artifacts(tmp_path)


def test_git_blob_id_computation():
    """Verify Git blob ID computation matches 'git hash-object'."""
    payload = b"hello"
    blob_id = _git_blob_id(payload)
    # git hash-object computes: SHA1("blob 5\0hello")
    expected = hashlib.sha1(b"blob 5\0hello").hexdigest()
    assert blob_id == expected


def test_extra_claim_keys_rejected(tmp_path):
    """Extra keys in individual claim entries must be rejected."""
    claims_file = tmp_path / "claims.json"
    claims_file.write_text(json.dumps([
        {"ordinal": 1, "text": "test", "sha256": hashlib.sha256(b"test").hexdigest(),
         "targeted": False, "extra_field": "bad"},
    ]), encoding="utf-8")
    with pytest.raises(IntegrityError, match="[Ee]xtra"):
        load_live_claims(claims_file)


def test_duplicate_manifest_json_keys_rejected(tmp_path):
    """Duplicate keys in the manifest JSON itself must be rejected."""
    manifest = tmp_path / "manifest.json"
    # Write raw JSON with duplicate keys
    manifest.write_text(
        '{"attempt_id": "a", "paper_id": "b", "snapshot_id": "c", '
        '"upstream_revision": "d", "artifacts": [], "attempt_id": "e"}',
        encoding="utf-8",
    )
    with pytest.raises(IntegrityError, match="[Dd]uplicate"):
        load_manifest(path=manifest)
