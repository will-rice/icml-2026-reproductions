"""Controller-owned local validation and worktree isolation tests."""

from __future__ import annotations

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


NOW = datetime(2026, 7, 25, 18, 0, tzinfo=timezone.utc)
HEAD = "2" * 40
TREE = "3" * 40


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
        failed_command: int | None = None,
        top_level: str | None = None,
    ) -> None:
        branch = manifest["branch"]
        self.responses = defaultdict(list)
        self.responses[("git", "rev-parse", "--show-toplevel")] = [
            top_level or manifest["worktree"]
        ]
        self.responses[("git", "status", "--porcelain")] = list(status)
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
    manifest["worktree"] = str(worktree)
    return paths, lease, manifest


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


def test_real_runner_uses_sanitized_environment(tmp_path: Path, monkeypatch):
    captured = {}

    def fake_run(argv, **kwargs):
        captured["argv"] = argv
        captured.update(kwargs)
        return subprocess.CompletedProcess(argv, 0, "out", "err")

    monkeypatch.setenv("HF_TOKEN", "secret")
    monkeypatch.setattr(controller.subprocess, "run", fake_run)

    result = controller.run_command(("git", "status"), tmp_path)

    assert result == controller.CommandResult(("git", "status"), 0, "out", "err")
    assert captured["argv"] == ("git", "status")
    assert captured["cwd"] == tmp_path
    assert captured["text"] is True
    assert captured["capture_output"] is True
    assert captured["check"] is False
    assert captured["env"] == controller.clean_validation_environment()
    assert "HF_TOKEN" not in captured["env"]


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
