"""Tests for direct, transactional schema-v3 to schema-v6 migration."""

import importlib
import json
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "skills" / "icml-repro-loop" / "scripts"
FIXTURES = Path(__file__).with_name("fixtures")
sys.path.insert(0, str(SCRIPTS))
store = importlib.import_module("store")


def migration_module():
    sys.modules.pop("migrate_v6", None)
    return importlib.import_module("migrate_v6")


def load_fixture(name: str) -> dict:
    with (FIXTURES / name).open(encoding="utf-8") as file:
        return json.load(file)


def migrate_fixture(tmp_path: Path, source: dict):
    migrate_v6 = migration_module()
    paths = store.StatePaths(tmp_path / "repro-loop.json")
    migrate_v6.apply_v6_migration(paths, migrate_v6.plan_v6_migration(source))
    return paths


def only_active_attempt(paths) -> dict:
    index = store.read_json(paths.index)
    [attempt_id] = index["attempts"]
    return store.read_json(paths.attempt(attempt_id))


def assert_no_dangling_index_references(paths) -> None:
    if not paths.index.exists():
        return
    index = store.read_json(paths.index)
    if index.get("version") != 6:
        return
    for reference in (*index["attempts"].values(), *index["history"].values()):
        shard = paths.index.parent / reference["path"]
        assert shard.exists()
        store.validate_attempt(store.read_json(shard))


def interrupt_migration(tmp_path: Path, fail_after: int):
    migrate_v6 = migration_module()
    source = load_fixture("repro-loop-v3-eeg.json")
    paths = store.StatePaths(tmp_path / "repro-loop.json")
    store.atomic_json_write(paths.index, source, migrate_v6.legacy_state.validate_state)
    plan = migrate_v6.plan_v6_migration(source)
    writes = 0
    real_write = migrate_v6._write_json

    def interrupted_write(path, value, validator):
        nonlocal writes
        writes += 1
        if writes == fail_after:
            raise OSError("simulated interruption")
        return real_write(path, value, validator)

    migrate_v6._write_json = interrupted_write
    with pytest.raises(OSError, match="simulated interruption"):
        migrate_v6.apply_v6_migration(paths, plan)
    migrate_v6._write_json = real_write
    return paths


def test_migration_preserves_active_eeg_attempt(tmp_path):
    source = load_fixture("repro-loop-v3-eeg.json")
    paths = migrate_fixture(tmp_path, source)
    attempt = only_active_attempt(paths)
    for field in (
        "paper_id",
        "title",
        "slug",
        "project_path",
        "upstream_revision",
        "target_claims",
        "design_approved",
        "estimated_api_cost_usd",
    ):
        assert attempt[field] == source["current"][field]
    assert attempt["phase"] == "implementing"


def test_migration_preserves_history_rejections_and_cost(tmp_path):
    source = load_fixture("repro-loop-v3-eeg.json")
    paths = migrate_fixture(tmp_path, source)
    index = store.read_json(paths.index)
    assert len(index["history"]) == len(source["history"])
    assert index["rejections"] == source["rejections"]
    assert index["total_api_cost_usd"] == source["total_api_cost_usd"]
    [history_id] = index["history"]
    archived = store.read_json(paths.attempt(history_id))
    assert archived["paper_id"] == "HMu24dTKkJ"
    assert archived["blocked_from"] == "implementing"


@pytest.mark.parametrize("fail_after", range(1, 12))
def test_interrupted_migration_recovers_atomically(tmp_path, fail_after):
    paths = interrupt_migration(tmp_path, fail_after)
    migrate_v6 = migration_module()
    migrate_v6.recover_transactions(paths)
    assert_no_dangling_index_references(paths)
    assert store.read_json(paths.index)["version"] == 6


def test_every_expected_migration_write_boundary_is_fault_injected(tmp_path):
    migrate_v6 = migration_module()
    source = load_fixture("repro-loop-v3-eeg.json")
    paths = store.StatePaths(tmp_path / "repro-loop.json")
    plan = migrate_v6.plan_v6_migration(source)
    writes = []
    real_write = migrate_v6._write_json

    def recording_write(path, value, validator):
        writes.append(path)
        return real_write(path, value, validator)

    migrate_v6._write_json = recording_write
    migrate_v6.apply_v6_migration(paths, plan)
    migrate_v6._write_json = real_write
    assert writes == [
        paths.root / "transactions" / f"{plan.source_sha256}.json",
        *[
            paths.index.parent / entry["staging"]
            for entry in migrate_v6._manifest(paths, plan)["targets"]
        ],
        paths.root / "transactions" / f"{plan.source_sha256}.json",
        *[
            paths.index.parent / entry["target"]
            for entry in sorted(
                migrate_v6._manifest(paths, plan)["targets"],
                key=lambda entry: entry["target"] == paths.index.name,
            )
        ],
        paths.root / "transactions" / f"{plan.source_sha256}.json",
    ]


def test_migration_is_byte_idempotent_and_retains_hashed_backup(tmp_path):
    migrate_v6 = migration_module()
    source = load_fixture("repro-loop-v3-eeg.json")
    paths = store.StatePaths(tmp_path / "repro-loop.json")
    plan = migrate_v6.plan_v6_migration(source)
    migrate_v6.apply_v6_migration(paths, plan)
    bytes_before = {
        path: path.read_bytes()
        for path in paths.root.rglob("*.json")
        if "transactions" not in path.parts
    }
    migrate_v6.apply_v6_migration(paths, plan)
    assert {
        path: path.read_bytes()
        for path in paths.root.rglob("*.json")
        if "transactions" not in path.parts
    } == bytes_before
    backup = paths.root / "v3-backups" / f"{plan.source_sha256}.json"
    assert store.read_json(backup) == source


def test_manifest_records_source_targets_hashes_staging_and_completion(tmp_path):
    migrate_v6 = migration_module()
    source = load_fixture("repro-loop-v3-eeg.json")
    paths = migrate_fixture(tmp_path, source)
    plan = migrate_v6.plan_v6_migration(source)
    manifest = store.read_json(
        paths.root / "transactions" / f"{plan.source_sha256}.json"
    )
    assert manifest["source_sha256"] == plan.source_sha256
    assert manifest["status"] == "complete"
    assert {entry["target"] for entry in manifest["targets"]} == {
        str(path.relative_to(paths.index.parent))
        for path in plan.resolved_targets(paths)
    }
    assert all(
        set(entry) == {"sha256", "staging", "target"}
        for entry in manifest["targets"]
    )


@pytest.mark.parametrize(
    "mutate",
    [
        lambda manifest: manifest.update(source_sha256="z" * 64),
        lambda manifest: manifest["targets"][0].update(sha256="z" * 64),
        lambda manifest: manifest["targets"][0].update(target="../escape.json"),
        lambda manifest: manifest["targets"][0].update(target="/tmp/escape.json"),
        lambda manifest: manifest["targets"][0].update(staging="../escape.json"),
        lambda manifest: manifest["targets"][0].update(staging="/tmp/escape.json"),
        lambda manifest: manifest["targets"].__setitem__(
            0, dict(manifest["targets"][1])
        ),
        lambda manifest: manifest["targets"].pop(),
        lambda manifest: manifest["targets"][0].update(sha256="0" * 64),
        lambda manifest: manifest["targets"][0].update(
            staging=manifest["targets"][1]["staging"]
        ),
    ],
    ids=[
        "non-hex-source-hash",
        "non-hex-target-hash",
        "parent-target",
        "absolute-target",
        "parent-staging",
        "absolute-staging",
        "duplicate-target",
        "missing-target",
        "wrong-target-hash",
        "wrong-staging",
    ],
)
def test_manifest_must_exactly_authenticate_the_deterministic_plan(
    tmp_path, mutate
):
    migrate_v6 = migration_module()
    source = load_fixture("repro-loop-v3-eeg.json")
    paths = migrate_fixture(tmp_path, source)
    plan = migrate_v6.plan_v6_migration(source)
    manifest_path = (
        paths.root / "transactions" / f"{plan.source_sha256}.json"
    )
    manifest = store.read_json(manifest_path)
    mutate(manifest)
    store.atomic_json_write(manifest_path, manifest, lambda value: None)

    with pytest.raises(ValueError, match="manifest"):
        migrate_v6.apply_v6_migration(paths, plan)


def test_archived_effective_phase_matches_shard_and_index_reference(tmp_path):
    source = load_fixture("repro-loop-v3-eeg.json")
    paths = migrate_fixture(tmp_path, source)
    index = store.read_json(paths.index)
    [attempt_id] = index["history"]
    archived = store.read_json(paths.attempt(attempt_id))
    expected_phase = archived.get("blocked_from", "complete")
    assert archived["phase"] == expected_phase
    assert index["history"][attempt_id]["phase"] == expected_phase


@pytest.mark.parametrize(
    ("record", "expected"),
    [
        ({"paper_id": "completed"}, "complete"),
        ({"paper_id": "blocked", "blocked_from": "implementing"}, "implementing"),
    ],
)
def test_archived_effective_phase_is_explicitly_defined(record, expected):
    migrate_v6 = migration_module()
    assert migrate_v6.effective_archived_phase(record) == expected


def run_migrate_cli(path: Path, *arguments: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [
            sys.executable,
            str(SCRIPTS / "state.py"),
            "migrate-v6",
            str(path),
            *arguments,
        ],
        check=False,
        capture_output=True,
        text=True,
    )


def source_sha256(path: Path) -> str:
    return migration_module().plan_for_existing_migration(
        store.StatePaths(path)
    ).source_sha256


def test_migrate_v6_cli_requires_an_explicit_mode_before_writing(tmp_path):
    source_path = tmp_path / "repro-loop.json"
    source_path.write_text(
        (FIXTURES / "repro-loop-v3-eeg.json").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    before = source_path.read_bytes()

    result = run_migrate_cli(source_path)

    assert result.returncode != 0
    assert source_path.read_bytes() == before
    assert not source_path.with_suffix("").exists()


@pytest.mark.parametrize(
    "arguments",
    [
        ("--apply",),
        ("--apply", "--expected-source-sha256", "0" * 64),
    ],
)
def test_migrate_v6_cli_apply_requires_matching_source_digest(tmp_path, arguments):
    source_path = tmp_path / "repro-loop.json"
    source_path.write_text(
        (FIXTURES / "repro-loop-v3-eeg.json").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    before = source_path.read_bytes()

    result = run_migrate_cli(source_path, *arguments)

    assert result.returncode != 0
    assert source_path.read_bytes() == before
    assert not source_path.with_suffix("").exists()


def test_migrate_v6_apply_rejects_source_swap_before_any_migration_write(
    tmp_path,
    monkeypatch,
):
    migrate_v6 = migration_module()
    state_cli = importlib.import_module("state")
    source_path = tmp_path / "repro-loop.json"
    source = load_fixture("repro-loop-v3-eeg.json")
    changed = json.loads(json.dumps(source))
    changed["current"]["title"] = "adversarially swapped source"
    store.atomic_json_write(
        source_path,
        source,
        migrate_v6.legacy_state.validate_state,
    )
    real_plan = migrate_v6.plan_for_existing_migration

    def swap_after_checked_plan(paths):
        plan = real_plan(paths)
        store.atomic_json_write(
            source_path,
            changed,
            migrate_v6.legacy_state.validate_state,
        )
        return plan

    monkeypatch.setattr(
        migrate_v6,
        "plan_for_existing_migration",
        swap_after_checked_plan,
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(SCRIPTS / "state.py"),
            "migrate-v6",
            str(source_path),
            "--apply",
            "--expected-source-sha256",
            (
                "f9fb0c976243de61b8fe90441e100c6b"
                "c88f341a50adb5326ffd12c8d7e99354"
            ),
        ],
    )

    with pytest.raises(ValueError, match="source_state_sha256"):
        state_cli.main()

    assert store.read_json(source_path) == changed
    assert not source_path.with_suffix("").exists()


def test_checked_migration_cas_does_not_overwrite_last_moment_source_swap(
    tmp_path,
    monkeypatch,
):
    migrate_v6 = migration_module()
    source_path = tmp_path / "repro-loop.json"
    paths = store.StatePaths(source_path)
    source = load_fixture("repro-loop-v3-eeg.json")
    changed = json.loads(json.dumps(source))
    changed["current"]["title"] = "last-moment adversarial source"
    store.atomic_json_write(
        source_path,
        source,
        migrate_v6.legacy_state.validate_state,
    )
    plan = migrate_v6.plan_v6_migration(source)
    real_matches = migrate_v6._matches
    swapped = False

    def swap_at_index_install(path, expected_sha256):
        nonlocal swapped
        matches = real_matches(path, expected_sha256)
        if path == source_path and not swapped:
            swapped = True
            store._atomic_json_write(source_path, changed)
        return matches

    monkeypatch.setattr(migrate_v6, "_matches", swap_at_index_install)

    with pytest.raises(ValueError, match="source_state_sha256"):
        migrate_v6.apply_checked_v6_migration(paths, plan)

    assert swapped is True
    assert store.read_json(source_path) == changed
    assert store.read_json(source_path)["version"] == 3


def test_migrate_v6_cli_dry_run_reports_plan_without_writes(tmp_path):
    source_path = tmp_path / "repro-loop.json"
    source_path.write_text(
        (FIXTURES / "repro-loop-v3-eeg.json").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    before = source_path.read_bytes()

    result = run_migrate_cli(source_path, "--dry-run")

    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == {
        "active_attempts": 1,
        "archived_attempts": 1,
        "rejections": 9,
        "max_runnable_attempts": 20,
        "source_state_sha256": source_sha256(source_path),
        "total_api_cost_usd": 0.0,
    }
    assert source_path.read_bytes() == before
    assert not source_path.with_suffix("").exists()


def test_migrate_v6_cli_rerun_is_idempotent(tmp_path):
    source_path = tmp_path / "repro-loop.json"
    source_path.write_text(
        (FIXTURES / "repro-loop-v3-eeg.json").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    expected_source_sha256 = source_sha256(source_path)
    first = run_migrate_cli(
        source_path,
        "--apply",
        "--expected-source-sha256",
        expected_source_sha256,
    )
    bytes_before = {
        path: path.read_bytes()
        for path in tmp_path.rglob("*.json")
    }

    second = run_migrate_cli(
        source_path,
        "--apply",
        "--expected-source-sha256",
        expected_source_sha256,
    )

    assert first.returncode == second.returncode == 0
    assert json.loads(second.stdout) == json.loads(first.stdout)
    assert {path: path.read_bytes() for path in tmp_path.rglob("*.json")} == (
        bytes_before
    )


@pytest.mark.parametrize("fail_after", [10, 11])
def test_migrate_v6_cli_recovers_after_index_install(tmp_path, fail_after):
    paths = interrupt_migration(tmp_path, fail_after)

    result = run_migrate_cli(
        paths.index,
        "--apply",
        "--expected-source-sha256",
        source_sha256(paths.index),
    )

    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["active_attempts"] == 1
    assert store.read_json(paths.index)["version"] == 6
    assert_no_dangling_index_references(paths)


def test_semantic_equivalence_verifies_every_legacy_record(tmp_path):
    migrate_v6 = migration_module()
    source = load_fixture("repro-loop-v3-eeg.json")
    paths = migrate_fixture(tmp_path, source)
    summary = migrate_v6.verify_semantic_equivalence(source, paths)
    assert summary == {
        "active_attempts": 1,
        "archived_attempts": 1,
        "rejections": 9,
        "max_runnable_attempts": 20,
        "total_api_cost_usd": 0.0,
    }
