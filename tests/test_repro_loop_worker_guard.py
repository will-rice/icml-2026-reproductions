"""Untrusted paper-worker launch boundary tests."""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "skills" / "icml-repro-loop" / "scripts"
sys.path.insert(0, str(SCRIPTS))

import worker_guard


def write_contract(
    worktree: Path,
    *,
    mode: str = "implementation",
    project_path: str = "submissions/paper-a",
    contract_path: Path | None = None,
) -> Path:
    path = contract_path or worktree / ".superpowers" / "worker-contract.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "version": 1,
                "attempt_id": "attempt-a",
                "paper_id": "paper-a",
                "worktree": str(worktree),
                "project_path": project_path,
                "mode": mode,
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def pass_preflight(runtime: str, worktree: Path) -> None:
    def isolated(request: worker_guard.PreflightProbe) -> None:
        assert request.runtime == runtime
        assert request.cwd == worktree
        assert request.control_path.is_relative_to(worktree)
        assert not request.outside_write_path.is_relative_to(worktree)
        request.control_path.parent.mkdir(parents=True, exist_ok=True)
        request.control_path.write_text(request.marker, encoding="utf-8")

    worker_guard.preflight_runtime(runtime, worktree, isolated)


@pytest.mark.parametrize(
    "key",
    [
        "HF_TOKEN",
        "HUGGING_FACE_HUB_TOKEN",
        "GH_TOKEN",
        "GITHUB_TOKEN",
        "GIT_ASKPASS",
        "SSH_ASKPASS",
        "GIT_CREDENTIAL_HELPER",
        "GIT_CONFIG_COUNT",
        "GIT_CONFIG_KEY_0",
        "GIT_CONFIG_VALUE_0",
        "GCM_CREDENTIAL_STORE",
        "HF_TOKEN_PATH",
        "HF_HOME",
        "HUGGINGFACE_HUB_CACHE",
        "TRANSFORMERS_CACHE",
    ],
)
def test_clean_environment_removes_credentials_and_inherited_hf_cache(
    key: str,
):
    cleaned = worker_guard.clean_environment(
        {
            "PATH": os.environ.get("PATH", ""),
            "LANG": "C.UTF-8",
            key: "must-not-reach-worker",
        }
    )

    assert key not in cleaned
    assert cleaned["HF_HUB_DISABLE_IMPLICIT_TOKEN"] == "1"
    assert cleaned["GIT_TERMINAL_PROMPT"] == "0"
    assert cleaned["LANG"] == "C.UTF-8"


def test_antigravity_command_requires_sandbox_and_rejects_unsafe_flags():
    worker_guard.validate_worker_command(
        (
            "agy",
            "--model",
            "gemini-3.1-pro-high",
            "--mode",
            "accept-edits",
            "--sandbox",
            "--print",
            "Follow the assigned worker contract.",
        ),
        "antigravity",
    )

    for argv in (
        ("agy", "--model", "m", "--mode", "accept-edits", "--print", "work"),
        (
            "agy",
            "--sandbox",
            "--dangerously-skip-permissions",
            "--print",
            "work",
        ),
        ("agy", "--sandbox", "--add-dir", "/tmp/other", "--print", "work"),
    ):
        with pytest.raises(ValueError):
            worker_guard.validate_worker_command(argv, "antigravity")


def test_codex_command_requires_a_bounded_sandbox_and_rejects_expansion(
    tmp_path: Path,
):
    worker_guard.validate_worker_command(
        (
            "codex",
            "exec",
            "-s",
            "workspace-write",
            "-C",
            str(tmp_path),
            "--ephemeral",
            "Follow the assigned worker contract.",
        ),
        "codex",
    )

    for argv in (
        ("codex", "exec", "-C", str(tmp_path), "work"),
        (
            "codex",
            "exec",
            "-s",
            "danger-full-access",
            "-C",
            str(tmp_path),
            "work",
        ),
        (
            "codex",
            "exec",
            "-s",
            "workspace-write",
            "-C",
            str(tmp_path),
            "--add-dir",
            "/tmp/other",
            "work",
        ),
        (
            "codex",
            "exec",
            "-s",
            "workspace-write",
            "-C",
            str(tmp_path),
            "-c",
            "sandbox_permissions=['disk-full-read-access']",
            "work",
        ),
        (
            "codex",
            "exec",
            "-s",
            "workspace-write",
            "-C",
            str(tmp_path),
            "--dangerously-bypass-approvals-and-sandbox",
            "work",
        ),
    ):
        with pytest.raises(ValueError):
            worker_guard.validate_worker_command(argv, "codex")


@pytest.mark.parametrize(
    "coordinator_path",
    [
        "/repo/state/repro-loop.json",
        "/repo/skills/icml-repro-loop/SKILL.md",
        "/repo/docs/REMOTE_SETUP.md",
    ],
)
def test_worker_command_rejects_coordinator_paths(
    tmp_path: Path,
    coordinator_path: str,
):
    with pytest.raises(ValueError, match="coordinator"):
        worker_guard.validate_worker_command(
            (
                "codex",
                "exec",
                "-s",
                "workspace-write",
                "-C",
                str(tmp_path),
                f"Modify {coordinator_path}",
            ),
            "codex",
        )


@pytest.mark.parametrize("runtime", ["codex", "antigravity"])
def test_launch_spec_is_rooted_at_one_worktree_and_uses_empty_hf_cache(
    tmp_path: Path,
    monkeypatch,
    runtime: str,
):
    worktree = tmp_path / "paper-worktree"
    worktree.mkdir()
    contract = write_contract(worktree)
    pass_preflight(runtime, worktree)
    monkeypatch.setenv("HF_TOKEN", "secret")
    monkeypatch.setenv("GH_TOKEN", "secret")
    monkeypatch.setenv("HF_HOME", "/tmp/inherited-hf-cache")

    spec = worker_guard.launch_spec(runtime, "model-a", worktree, contract)

    assert spec.cwd == worktree
    assert spec.contract == contract
    assert spec.mode == "implementation"
    assert "--add-dir" not in spec.argv
    assert "HF_TOKEN" not in spec.env
    assert "GH_TOKEN" not in spec.env
    assert spec.env["HF_HUB_DISABLE_IMPLICIT_TOKEN"] == "1"
    hf_home = Path(spec.env["HF_HOME"])
    assert hf_home.is_relative_to(worktree)
    assert hf_home.is_dir()
    assert list(hf_home.iterdir()) == []
    if runtime == "codex":
        assert ("-s", "workspace-write") == (
            spec.argv[spec.argv.index("-s") : spec.argv.index("-s") + 2]
        )
        assert spec.argv[spec.argv.index("-C") + 1] == str(worktree)
    else:
        assert "--sandbox" in spec.argv
        assert "--new-project" in spec.argv
    worker_guard.validate_worker_command(spec.argv, runtime)


def test_preflight_rejects_outside_write_or_credential_read(
    tmp_path: Path,
):
    worktree = tmp_path / "paper-worktree"
    worktree.mkdir()

    def outside_write(request: worker_guard.PreflightProbe) -> None:
        request.control_path.parent.mkdir(parents=True, exist_ok=True)
        request.control_path.write_text(request.marker, encoding="utf-8")
        request.outside_write_path.write_text("escaped", encoding="utf-8")

    with pytest.raises(RuntimeError, match="outside"):
        worker_guard.preflight_runtime("codex", worktree, outside_write)

    def credential_read(request: worker_guard.PreflightProbe) -> None:
        request.control_path.parent.mkdir(parents=True, exist_ok=True)
        request.control_path.write_text(request.marker, encoding="utf-8")
        request.credential_leak_path.write_text(
            request.credential_marker,
            encoding="utf-8",
        )

    with pytest.raises(RuntimeError, match="credential"):
        worker_guard.preflight_runtime(
            "antigravity",
            worktree,
            credential_read,
        )


def test_preflight_requires_proof_that_the_probe_executed(tmp_path: Path):
    worktree = tmp_path / "paper-worktree"
    worktree.mkdir()

    with pytest.raises(RuntimeError, match="control"):
        worker_guard.preflight_runtime(
            "codex",
            worktree,
            lambda _request: None,
        )


@pytest.mark.parametrize("runtime", ["codex", "antigravity"])
def test_unproven_runtime_is_limited_to_read_only_research(
    tmp_path: Path,
    runtime: str,
):
    worktree = tmp_path / "paper-worktree"
    worktree.mkdir()
    implementation = write_contract(worktree)

    with pytest.raises(RuntimeError, match="preflight"):
        worker_guard.launch_spec(
            runtime,
            "model-a",
            worktree,
            implementation,
        )

    research = write_contract(
        worktree,
        mode="research",
        contract_path=worktree / ".superpowers" / "research-contract.json",
    )
    spec = worker_guard.launch_spec(
        runtime,
        "model-a",
        worktree,
        research,
    )
    assert spec.mode == "research"
    if runtime == "codex":
        assert spec.argv[spec.argv.index("-s") + 1] == "read-only"
    else:
        assert spec.argv[spec.argv.index("--mode") + 1] == "plan"


def test_contract_and_project_paths_must_remain_inside_assigned_worktree(
    tmp_path: Path,
):
    worktree = tmp_path / "paper-worktree"
    worktree.mkdir()
    outside_contract = write_contract(
        worktree,
        mode="research",
        contract_path=tmp_path / "outside-contract.json",
    )
    with pytest.raises(ValueError, match="contract"):
        worker_guard.launch_spec(
            "codex",
            "model-a",
            worktree,
            outside_contract,
        )

    escaping_project = write_contract(
        worktree,
        mode="research",
        project_path="../other-paper",
    )
    with pytest.raises(ValueError, match="project_path"):
        worker_guard.launch_spec(
            "codex",
            "model-a",
            worktree,
            escaping_project,
        )
