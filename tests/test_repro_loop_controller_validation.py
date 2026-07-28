"""Controller-owned local validation and worktree isolation tests."""

from __future__ import annotations

import argparse
from collections import defaultdict
import copy
from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "skills" / "icml-repro-loop" / "scripts"
STATE = SCRIPTS / "state.py"
MANIFEST = ROOT / "tests" / "fixtures" / "validation-manifest.json"
sys.path.insert(0, str(SCRIPTS))

import attestations
import attempts
import controller
import leases
import store
import state
import telemetry


NOW = datetime(2026, 7, 25, 18, 0, tzinfo=timezone.utc)
HEAD = "2" * 40
TREE = "3" * 40
SOURCE_TREE_SHA256 = (
    "f356bf8e6186efe51cf1ec9403ab1eeb71f744a4d3b81ff1cb1195c0e518035d"
)


@pytest.fixture(autouse=True)
def fixed_validation_clock(monkeypatch):
    monkeypatch.setattr(controller, "validation_now", lambda: NOW, raising=False)


class FakeRunner:
    """Return controlled real-command-shaped outputs without running pipelines."""

    def __init__(
        self,
        manifest: dict,
        *,
        status: tuple[str, ...] = ("", ""),
        branches: tuple[str, ...] | None = None,
        heads: tuple[str, ...] = (HEAD, HEAD),
        changed_paths: str | None = None,
        project_status: tuple[str, ...] = ("", ""),
        failed_command: int | None = None,
        top_level: str | None = None,
    ) -> None:
        branch = manifest["branch"]
        self.responses = defaultdict(list)
        self.responses[("git", "rev-parse", "--show-toplevel")] = [
            top_level or manifest["worktree"]
        ]
        self.responses[("git", "status", "--porcelain")] = list(status)
        self.responses[
            (
                "git",
                "status",
                "--porcelain",
                "--ignored",
                "--untracked-files=all",
                "--",
                manifest["project_path"],
            )
        ] = list(project_status)
        self.responses[("git", "branch", "--show-current")] = list(
            branches or (branch, branch)
        )
        self.responses[("git", "rev-parse", "HEAD")] = list(heads)
        self.responses[
            (
                "git",
                "diff",
                "--name-only",
                f"{manifest['base_sha']}...HEAD",
            )
        ] = [
            changed_paths
            if changed_paths is not None
            else (
                f"{manifest['project_path']}/evidence.py\n"
                f"{manifest['design_path']}\n"
            )
        ]
        self.responses[("git", "rev-parse", "HEAD^{tree}")] = [TREE]
        self.failed_argv = (
            None
            if failed_command is None
            else tuple(manifest["commands"][failed_command])
        )
        self.calls: list[tuple[tuple[str, ...], Path]] = []

    def __call__(
        self, argv: tuple[str, ...], worktree: Path
    ) -> controller.CommandResult:
        self.calls.append((argv, worktree))
        if argv == self.failed_argv:
            return controller.CommandResult(argv, 7, "partial\n", "failed\n")
        queued = self.responses[argv]
        if queued:
            return controller.CommandResult(argv, 0, queued.pop(0) + "\n", "")
        return controller.CommandResult(
            argv, 0, f"visible standard output for {argv!r}\n", "warning\n"
        )


@pytest.fixture
def validation_case(tmp_path: Path):
    paths = store.StatePaths(tmp_path / "repro-loop.json")
    store.atomic_json_write(paths.index, store.new_index(), store.validate_index)
    lease = leases.acquire_lease(
        paths,
        "attempt:a1",
        "controller-1",
        "a1",
        NOW,
        timedelta(hours=2),
    )
    attempts.create_attempt(
        paths,
        "a1",
        {
            "paper_id": "paper-1",
            "title": "Paper One",
            "slug": "paper-1",
            "project_path": "submissions/paper-1",
            "upstream_revision": "upstream-commit",
            "target_claims": ["claim-1", "claim-2"],
            "estimated_api_cost_usd": 0.0,
        },
        lease,
        "snapshot-1",
        NOW,
    )
    attempts.transition_attempt(paths, "a1", "design-pending", lease, NOW)
    attempts.record_design(
        paths, "a1", lease, "author-1", "docs/designs/paper-1.md", NOW
    )
    attempts.record_design_review(
        paths, "a1", lease, "reviewer-1", "approved", NOW
    )
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    worktree = tmp_path / "paper-worktree"
    worktree.mkdir()
    project = worktree / "submissions" / "paper-1"
    project.mkdir(parents=True)
    (project / "README.md").write_text(
        "---\ntitle: Paper One\n---\n", encoding="utf-8"
    )
    (project / "app.py").write_text(
        "print('paper one')\n", encoding="utf-8"
    )
    manifest["worktree"] = str(worktree)
    return paths, lease, manifest


def rewrite_attempt(paths, field, value) -> None:
    attempt = store.read_json(paths.attempt("a1"))
    attempt[field] = value
    store.atomic_json_write(
        paths.attempt("a1"), attempt, store.validate_attempt
    )


@pytest.mark.parametrize(
    ("runner_updates", "error"),
    [
        ({"status": (" M submissions/paper-1/evidence.py", "")}, "clean"),
        (
            {"branches": ("different-branch", "different-branch")},
            "branch",
        ),
        (
            {"changed_paths": "submissions/other-paper/evidence.py\n"},
            "changed path",
        ),
        ({"heads": (HEAD, "4" * 40)}, "source commit"),
    ],
)
def test_validation_rejects_git_boundary_mismatch(
    validation_case, runner_updates, error
):
    paths, lease, manifest = validation_case
    runner = FakeRunner(manifest, **runner_updates)

    with pytest.raises(ValueError, match=error):
        controller.attest_validation(
            paths, "a1", lease, manifest, runner, NOW
        )

    assert attempts.read_attempt(paths, "a1")["phase"] == "implementing"
    assert not paths.attestation("validation", "a1").exists()


def test_validation_rejects_ignored_file_inside_project(validation_case):
    paths, lease, manifest = validation_case
    runner = FakeRunner(
        manifest,
        project_status=("!! submissions/paper-1/ignored.bin\n", ""),
    )

    with pytest.raises(ValueError, match="clean"):
        controller.attest_validation(
            paths, "a1", lease, manifest, runner, NOW
        )

    assert attempts.read_attempt(paths, "a1")["phase"] == "implementing"
    assert not paths.attestation("validation", "a1").exists()


@pytest.mark.parametrize("failed_command", range(5))
def test_validation_rejects_any_failed_declared_command(
    validation_case, failed_command
):
    paths, lease, manifest = validation_case
    runner = FakeRunner(manifest, failed_command=failed_command)

    with pytest.raises(ValueError, match="validation command"):
        controller.attest_validation(
            paths, "a1", lease, manifest, runner, NOW
        )

    assert attempts.read_attempt(paths, "a1")["phase"] == "implementing"
    assert not paths.attestation("validation", "a1").exists()


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("project_path", "submissions/other-paper"),
        ("design_path", "docs/designs/other-paper.md"),
    ],
)
def test_validation_requires_attempt_registered_paths(
    validation_case, field, replacement
):
    paths, lease, manifest = validation_case
    manifest[field] = replacement

    with pytest.raises(ValueError, match=field):
        controller.attest_validation(
            paths, "a1", lease, manifest, FakeRunner(manifest), NOW
        )


def test_validation_persists_deterministic_project_path_for_scheduled_attempt(
    validation_case,
):
    paths, lease, manifest = validation_case
    with store.locked_json(paths.attempt("a1"), store.validate_attempt) as attempt:
        attempt.pop("project_path")

    validated = controller.attest_validation(
        paths, "a1", lease, manifest, FakeRunner(manifest), NOW
    )

    assert validated["phase"] == "validated"
    assert validated["project_path"] == "submissions/paper-1"


def test_validation_requires_exact_registered_worktree(validation_case):
    paths, lease, manifest = validation_case
    runner = FakeRunner(manifest, top_level=str(Path(manifest["worktree"]).parent))

    with pytest.raises(ValueError, match="worktree"):
        controller.attest_validation(
            paths, "a1", lease, manifest, runner, NOW
        )


@pytest.mark.parametrize(
    "commands",
    [
        [["uv", "run", "true"]] * 5,
        [
            ["uv", "run", "python", "submissions/paper-1/evidence.py"],
            ["uv", "run", "pytest", "submissions/paper-1/tests", "-q"],
            ["uv", "run", "pre-commit", "run", "-a"],
            [
                "uv",
                "run",
                "/opt/codex/skill-creator/scripts/quick_validate.py",
                "skills/icml-repro-loop",
            ],
            ["uv", "run", "pytest", "-q"],
        ],
    ],
)
def test_validation_rejects_noncanonical_command_manifest(
    validation_case, commands
):
    paths, lease, manifest = validation_case
    manifest["commands"] = commands

    with pytest.raises(ValueError, match="commands"):
        controller.attest_validation(
            paths, "a1", lease, manifest, FakeRunner(manifest), NOW
        )


def test_validation_accepts_paper_local_uv_project_environment(validation_case):
    paths, lease, manifest = validation_case
    manifest["commands"][0] = [
        "uv",
        "run",
        "--project",
        "submissions/paper-1",
        "python",
        "-m",
        "paper_1.evidence",
    ]
    manifest["commands"][1] = [
        "uv",
        "run",
        "--project",
        "submissions/paper-1",
        "python",
        "-m",
        "pytest",
        "submissions/paper-1/tests",
        "-q",
    ]

    result = controller.attest_validation(
        paths, "a1", lease, manifest, FakeRunner(manifest), NOW
    )

    assert result["phase"] == "validated"


@pytest.mark.parametrize(
    "paper_pytest",
    [
        [
            "uv",
            "run",
            "pytest",
            "submissions/paper-1/tests",
            "-q",
        ],
        [
            "uv",
            "run",
            "--project",
            "submissions/paper-1",
            "pytest",
            "submissions/paper-1/tests",
            "-q",
        ],
        [
            "uv",
            "run",
            "--project",
            "submissions/other-paper",
            "python",
            "-m",
            "pytest",
            "submissions/paper-1/tests",
            "-q",
        ],
        [
            "uv",
            "run",
            "--project",
            "submissions/paper-1",
            "pytest",
            "--rootdir",
            "submissions/paper-1",
            "-q",
        ],
        [
            "uv",
            "run",
            "--project",
            "submissions/paper-1",
            "pytest",
            "submissions/paper-1/tests/test_claim.py::test_one",
            "-q",
        ],
    ],
    ids=[
        "path-pytest-shim-root",
        "path-pytest-shim-project",
        "other-project",
        "rootdir-without-target",
        "single-test-node",
    ],
)
def test_validation_rejects_noncanonical_paper_local_pytest(
    validation_case, paper_pytest
):
    paths, lease, manifest = validation_case
    manifest["commands"][1] = paper_pytest

    with pytest.raises(ValueError, match="commands"):
        controller.attest_validation(
            paths, "a1", lease, manifest, FakeRunner(manifest), NOW
        )


@pytest.mark.parametrize(
    ("status", "branches", "error"),
    [
        (("", "?? generated-output\n"), None, "clean"),
        (
            ("", ""),
            ("attempt-paper-1", "changed-after-validation"),
            "branch",
        ),
    ],
)
def test_validation_rejects_post_command_git_drift(
    validation_case, status, branches, error
):
    paths, lease, manifest = validation_case

    with pytest.raises(ValueError, match=error):
        controller.attest_validation(
            paths,
            "a1",
            lease,
            manifest,
            FakeRunner(manifest, status=status, branches=branches),
            NOW,
        )

    assert attempts.read_attempt(paths, "a1")["phase"] == "implementing"


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("phase", "design-pending", "phase"),
        ("design_review", None, "design_review"),
    ],
)
def test_validation_requires_approved_implementing_attempt(
    validation_case, field, value, error
):
    paths, lease, manifest = validation_case
    rewrite_attempt(paths, field, value)

    with pytest.raises(ValueError, match=error):
        controller.attest_validation(
            paths, "a1", lease, manifest, FakeRunner(manifest), NOW
        )


def test_validation_rejects_project_root_symlink_outside_worktree(
    validation_case, tmp_path: Path
):
    paths, lease, manifest = validation_case
    project = Path(manifest["worktree"]) / manifest["project_path"]
    outside = tmp_path / "outside-project"
    project.rename(outside)
    project.symlink_to(outside, target_is_directory=True)
    runner = FakeRunner(manifest)

    with pytest.raises(ValueError, match="project_path"):
        controller.attest_validation(
            paths, "a1", lease, manifest, runner, NOW
        )

    assert runner.calls == []
    assert attempts.read_attempt(paths, "a1")["phase"] == "implementing"
    assert not paths.attestation("validation", "a1").exists()


def test_source_tree_hash_rejects_symlinked_root(tmp_path: Path):
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "payload.py").write_text("print('outside')\n", encoding="utf-8")
    source_link = tmp_path / "source-link"
    source_link.symlink_to(outside, target_is_directory=True)

    with pytest.raises(ValueError, match="source_dir"):
        controller._source_tree_sha256(source_link)


@pytest.mark.parametrize(
    "bypass",
    [
        ["--collect-only"],
        ["--ignore", "submissions/paper-1/tests/test_claim.py"],
        ["--ignore-glob", "submissions/paper-1/tests/test_*"],
        ["--deselect", "submissions/paper-1/tests/test_claim.py::test_claim"],
        ["-k", "not claim"],
        ["-m", "not slow"],
        ["--lf"],
        ["--ff"],
        ["-o", "addopts=--collect-only"],
        ["--override-ini", "addopts=--collect-only"],
    ],
)
def test_validation_rejects_paper_pytest_bypass_flags(
    validation_case, bypass
):
    paths, lease, manifest = validation_case
    manifest["commands"][1].extend(bypass)

    with pytest.raises(ValueError, match="commands"):
        controller.attest_validation(
            paths, "a1", lease, manifest, FakeRunner(manifest), NOW
        )


def test_validation_rechecks_fence_at_fresh_time_after_commands(
    validation_case, monkeypatch
):
    paths, lease, manifest = validation_case
    runner = FakeRunner(manifest)
    monkeypatch.setattr(
        controller,
        "validation_now",
        lambda: NOW + timedelta(hours=3),
        raising=False,
    )

    with pytest.raises(leases.StaleFence):
        controller.attest_validation(
            paths, "a1", lease, manifest, runner, NOW
        )

    assert all(
        tuple(command) in [argv for argv, _ in runner.calls]
        for command in manifest["commands"]
    )
    assert attempts.read_attempt(paths, "a1")["phase"] == "implementing"
    assert not paths.attestation("validation", "a1").exists()


def test_validation_rejects_other_attempt_lease_before_commands(validation_case):
    paths, _, manifest = validation_case
    other = leases.acquire_lease(
        paths,
        "attempt:a2",
        "controller-2",
        "a2",
        NOW,
        timedelta(hours=2),
    )
    runner = FakeRunner(manifest)

    with pytest.raises(leases.StaleFence):
        controller.attest_validation(
            paths, "a1", other, manifest, runner, NOW
        )

    assert runner.calls == []


def test_valid_run_attests_hashed_outputs_commit_and_tree(
    validation_case, monkeypatch
):
    paths, lease, manifest = validation_case
    monkeypatch.setenv("HF_TOKEN", "must-not-enter-the-attestation")
    runner = FakeRunner(manifest)

    transitioned = controller.attest_validation(
        paths, "a1", lease, manifest, runner, NOW
    )

    attestation_id = transitioned["transitions"][-1]["attestation_id"]
    record = attestations.read(paths, attestation_id)
    declared_results = record["commands"]
    assert transitioned["phase"] == "validated"
    assert record["source_commit"] == HEAD
    assert record["source_tree"] == TREE
    assert record["source_tree_sha256"] == SOURCE_TREE_SHA256
    assert [result["argv"] for result in record["environment"]] == [
        ["git", "--version"],
        ["uv", "--version"],
        [sys.executable, "--version"],
    ]
    assert [result["argv"] for result in declared_results] == manifest["commands"]
    assert all(result["returncode"] == 0 for result in declared_results)
    first_stdout = (
        f"visible standard output for {tuple(manifest['commands'][0])!r}\n"
    )
    assert declared_results[0]["stdout_sha256"] == hashlib.sha256(
        first_stdout.encode()
    ).hexdigest()
    assert declared_results[0]["stderr_sha256"] == hashlib.sha256(
        b"warning\n"
    ).hexdigest()
    serialized = json.dumps(record)
    assert "must-not-enter-the-attestation" not in serialized
    assert "visible standard output" not in serialized
    assert "warning" not in serialized
    assert paths.attestation("validation", "a1").exists()
    assert all(Path(manifest["worktree"]) == cwd for _, cwd in runner.calls)


def test_controller_revalidates_one_predeployment_correction(
    validation_case,
):
    paths, lease, manifest = validation_case
    first = controller.attest_validation(
        paths, "a1", lease, manifest, FakeRunner(manifest), NOW
    )
    first_id = first["transitions"][-1]["attestation_id"]
    first_slot = paths.attestation("validation", "a1", 1)
    first_bytes = first_slot.read_bytes()
    attempts.transition_attempt(
        paths,
        "a1",
        "improving",
        lease,
        NOW,
        improvement_attempts=1,
        improvement_reason="Correct the evidence generation",
    )

    second = controller.attest_validation(
        paths,
        "a1",
        lease,
        manifest,
        FakeRunner(manifest, heads=("4" * 40, "4" * 40)),
        NOW,
    )

    second_id = second["transitions"][-1]["attestation_id"]
    assert second["phase"] == "validated"
    assert second["improvement_attempts"] == 1
    assert first_slot.read_bytes() == first_bytes
    assert attestations.read(paths, first_id)["attempt_number"] == 1
    assert attestations.read(paths, second_id)["attempt_number"] == 2
    assert paths.attestation("validation", "a1", 2).exists()
    assert first_id != second_id


def test_real_runner_uses_sanitized_environment(tmp_path: Path, monkeypatch):
    captured = {}

    def fake_run(argv, **kwargs):
        captured["argv"] = argv
        captured.update(kwargs)
        captured["home_entries"] = list(Path(kwargs["env"]["HOME"]).iterdir())
        captured["hf_entries"] = list(Path(kwargs["env"]["HF_HOME"]).iterdir())
        return subprocess.CompletedProcess(argv, 0, "out", "err")

    monkeypatch.setenv("HF_TOKEN", "secret")
    monkeypatch.setenv("HUGGING_FACE_HUB_TOKEN", "second-secret")
    monkeypatch.setenv(
        "UV_PROJECT_ENVIRONMENT",
        str(tmp_path / "submissions" / "paper-1" / ".venv"),
    )
    inherited_venv = tmp_path / "attacker-venv"
    monkeypatch.setenv("VIRTUAL_ENV", str(inherited_venv))
    monkeypatch.setenv(
        "PATH",
        os.pathsep.join(
            (
                str(inherited_venv / "bin"),
                "/usr/local/bin",
                "/usr/bin",
                "/bin",
            )
        ),
    )
    attacker_tmp = tmp_path / "attacker-tmp"
    attacker_tmp.mkdir()
    monkeypatch.setenv("TMPDIR", str(attacker_tmp))
    pre_commit_home = tmp_path / "pre-commit-home"
    monkeypatch.setenv("PRE_COMMIT_HOME", str(pre_commit_home))
    monkeypatch.setattr(controller.tempfile, "tempdir", None)
    monkeypatch.setattr(controller.subprocess, "run", fake_run)

    result = controller.run_command(("git", "status"), tmp_path)

    assert result == controller.CommandResult(("git", "status"), 0, "out", "err")
    assert captured["argv"] == ("git", "status")
    assert captured["cwd"] == tmp_path
    assert captured["text"] is True
    assert captured["capture_output"] is True
    assert captured["check"] is False
    assert "HF_TOKEN" not in captured["env"]
    assert "HUGGING_FACE_HUB_TOKEN" not in captured["env"]
    assert captured["env"]["HF_HUB_DISABLE_IMPLICIT_TOKEN"] == "1"
    assert captured["env"]["PATH"] == os.pathsep.join(
        ("/usr/local/bin", "/usr/bin", "/bin")
    )
    assert "VIRTUAL_ENV" not in captured["env"]
    assert captured["env"]["HOME"] != os.environ.get("HOME")
    isolated_root = Path(captured["env"]["HOME"]).parent
    assert not isolated_root.is_relative_to(tmp_path)
    assert captured["env"]["TMPDIR"] == str(isolated_root / "tmp")
    assert captured["env"].get("UV_PROJECT_ENVIRONMENT") == str(
        isolated_root / "uv-project-environment"
    )
    assert not Path(captured["env"]["UV_PROJECT_ENVIRONMENT"]).is_relative_to(
        tmp_path
    )
    assert captured["env"]["PYTHONDONTWRITEBYTECODE"] == "1"
    assert captured["env"]["PYTEST_ADDOPTS"] == "-p no:cacheprovider"
    assert captured["env"]["PRE_COMMIT_HOME"] == str(pre_commit_home)
    assert Path(captured["env"]["HF_HOME"]).parent == Path(
        captured["env"]["HOME"]
    ).parent
    assert captured["home_entries"] == []
    assert captured["hf_entries"] == []


def test_real_runner_keeps_generated_caches_outside_source_tree(tmp_path: Path):
    module = tmp_path / "sample_module.py"
    module.write_text("VALUE = 7\n", encoding="utf-8")
    tests = tmp_path / "tests"
    tests.mkdir()
    (tests / "test_sample.py").write_text(
        "from sample_module import VALUE\n\n"
        "def test_value():\n"
        "    assert VALUE == 7\n",
        encoding="utf-8",
    )
    before = controller._source_tree_sha256(tmp_path)

    result = controller.run_command(
        (sys.executable, "-m", "pytest", str(tests), "-q"),
        tmp_path,
    )

    assert result.returncode == 0, result.stderr
    assert controller._source_tree_sha256(tmp_path) == before
    assert not list(tmp_path.rglob("__pycache__"))
    assert not list(tmp_path.rglob(".pytest_cache"))


def test_state_measures_injected_validation_operation(validation_case):
    paths, lease, _manifest = validation_case
    result = {"transitions": [{"attestation_id": "a" * 64}]}
    arguments = argparse.Namespace(
        command="attest-validation",
        path=paths.index,
        attempt_id="a1",
        owner=lease.owner,
        fencing_token=lease.fencing_token,
        manifest=Path("unused-injected-manifest.json"),
        now=NOW.isoformat(),
    )

    returned = state._run_v6_command(
        arguments,
        validation_operation=lambda: result,
        utc_now=iter(
            ["2026-07-27T01:00:00+00:00", "2026-07-27T01:00:03+00:00"]
        ).__next__,
        monotonic_ns=iter([2_000_000_000, 5_000_000_000]).__next__,
        session_id_factory=lambda: "validation-stage",
    )

    assert returned is result
    finished = telemetry.read_session(paths, "validation-stage")[-1]
    assert finished["outcome"] == "passed"
    assert finished["elapsed_seconds"] == 3.0
    assert finished["attestation_id"] == "a" * 64


def test_state_cli_exposes_attest_validation_command():
    result = subprocess.run(
        [sys.executable, str(STATE), "--help"],
        check=False,
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": str(SCRIPTS)},
    )

    assert result.returncode == 0
    assert "attest-validation" in result.stdout


def test_state_cli_rejects_invalid_validation_manifest(
    validation_case, tmp_path
):
    paths, lease, manifest = validation_case
    manifest["commands"][2] = ["uv", "run", "pytest", "--collect-only"]
    manifest_path = tmp_path / "invalid-manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            str(STATE),
            "attest-validation",
            str(paths.index),
            "--attempt-id",
            "a1",
            "--owner",
            lease.owner,
            "--fencing-token",
            str(lease.fencing_token),
            "--manifest",
            str(manifest_path),
            "--now",
            NOW.isoformat(),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "commands" in result.stderr
    assert attempts.read_attempt(paths, "a1")["phase"] == "implementing"
