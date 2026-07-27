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
import store


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


def launch_spec(
    runtime: str,
    model: str,
    worktree: Path,
    contract: Path,
) -> worker_guard.LaunchSpec:
    return worker_guard.launch_spec(
        runtime,
        model,
        worktree,
        contract,
        attempt_id="attempt-a",
        paper_id="paper-a",
        project_path="submissions/paper-a",
    )


class CompletedProcess:
    pid = 4242
    returncode = 0

    def wait(self, timeout=None):
        assert timeout == 30
        return self.returncode


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

    spec = launch_spec(runtime, "model-a", worktree, contract)

    assert spec.cwd == worktree
    assert spec.contract == contract
    assert spec.mode == "implementation"
    assert spec.attempt_id == "attempt-a"
    assert spec.paper_id == "paper-a"
    assert spec.project_path == "submissions/paper-a"
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
        launch_spec(
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
    spec = launch_spec(
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
        launch_spec(
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
        launch_spec(
            "codex",
            "model-a",
            worktree,
            escaping_project,
        )


def test_run_worker_wraps_process_with_launch_and_exit_events(
    tmp_path, monkeypatch
):
    worktree = tmp_path / "paper-worktree"
    worktree.mkdir()
    contract = write_contract(worktree)
    pass_preflight("codex", worktree)
    spec = worker_guard.launch_spec(
        "codex",
        "model-a",
        worktree,
        contract,
        attempt_id="attempt-a",
        paper_id="paper-a",
        project_path="submissions/paper-a",
    )
    paths = store.StatePaths(tmp_path / "repro-loop.json")
    clock_values = iter(
        [
            "2026-07-27T00:00:00+00:00",
            "2026-07-27T00:00:01+00:00",
            "2026-07-27T00:00:06+00:00",
        ]
    )
    monotonic_values = iter([1_000_000_000, 6_000_000_000])

    result = worker_guard.run_worker(
        paths,
        spec,
        timeout_seconds=30,
        process_factory=lambda *args, **kwargs: CompletedProcess(),
        utc_now=lambda: next(clock_values),
        monotonic_ns=lambda: next(monotonic_values),
        session_id_factory=lambda: "session-a",
        git_head=lambda _path: "a" * 40,
    )

    events = worker_guard.telemetry.read_session(paths, "session-a")
    assert [event["event"] for event in events] == [
        "worker-queued",
        "worker-launched",
        "worker-exited",
    ]
    assert events[1]["pid"] == 4242
    assert events[2]["exit_code"] == 0
    assert events[2]["outcome"] == "proposal"
    assert result["elapsed_seconds"] == 5.0


def test_run_worker_records_nonzero_exit_as_failed(tmp_path: Path):
    class FailedProcess(CompletedProcess):
        returncode = 7

    paths, spec = worker_run_fixture(tmp_path)

    result = worker_guard.run_worker(
        paths,
        spec,
        timeout_seconds=30,
        process_factory=lambda *args, **kwargs: FailedProcess(),
        utc_now=iter_values(
            "2026-07-27T00:00:00+00:00",
            "2026-07-27T00:00:01+00:00",
            "2026-07-27T00:00:06+00:00",
        ),
        monotonic_ns=iter_values(1_000_000_000, 6_000_000_000),
        session_id_factory=lambda: "session-failed",
        git_head=lambda _path: "b" * 40,
    )

    exit_event = worker_guard.telemetry.read_session(
        paths, "session-failed"
    )[-1]
    assert exit_event["exit_code"] == 7
    assert exit_event["signal"] is None
    assert exit_event["outcome"] == "failed"
    assert result["elapsed_seconds"] == 5.0


def test_run_worker_persists_exit_when_post_run_git_lookup_fails(
    tmp_path: Path,
):
    paths, spec = worker_run_fixture(tmp_path)
    git_results = iter(["f" * 40, RuntimeError("sensitive git failure")])

    def git_head(_path):
        result = next(git_results)
        if isinstance(result, Exception):
            raise result
        return result

    result = worker_guard.run_worker(
        paths,
        spec,
        timeout_seconds=30,
        process_factory=lambda *args, **kwargs: CompletedProcess(),
        utc_now=iter_values(
            "2026-07-27T00:00:00+00:00",
            "2026-07-27T00:00:01+00:00",
            "2026-07-27T00:00:06+00:00",
        ),
        monotonic_ns=iter_values(1_000_000_000, 6_000_000_000),
        session_id_factory=lambda: "session-git-failure",
        git_head=git_head,
    )

    exit_event = worker_guard.telemetry.read_session(
        paths, "session-git-failure"
    )[-1]
    assert exit_event["outcome"] == "proposal"
    assert exit_event["git_sha_after"] is None
    assert exit_event["monotonic_ns"] == 6_000_000_000
    assert result["elapsed_seconds"] == 5.0
    session_bytes = b"".join(
        path.read_bytes()
        for path in sorted(
            (
                paths.root
                / "telemetry"
                / "session-git-failure"
            ).iterdir()
        )
    )
    assert b"sensitive git failure" not in session_bytes


def test_run_worker_reaps_child_when_launch_telemetry_fails(
    tmp_path: Path, monkeypatch
):
    class LaunchTelemetryFailureProcess:
        pid = 4242

        def __init__(self):
            self.terminated = False
            self.killed = False
            self.wait_timeouts = []

        def wait(self, timeout=None):
            self.wait_timeouts.append(timeout)
            assert self.terminated is True
            if not self.killed:
                raise worker_guard.subprocess.TimeoutExpired("codex", timeout)
            return -9

        def terminate(self):
            self.terminated = True

        def kill(self):
            self.killed = True

    paths, spec = worker_run_fixture(tmp_path)
    process = LaunchTelemetryFailureProcess()
    telemetry_error = RuntimeError("launch telemetry unavailable")
    append_event = worker_guard.telemetry.append_event

    def fail_launch_event(*args, **kwargs):
        if args[3] == "worker-launched":
            raise telemetry_error
        return append_event(*args, **kwargs)

    monkeypatch.setattr(worker_guard.telemetry, "append_event", fail_launch_event)

    with pytest.raises(RuntimeError) as raised:
        worker_guard.run_worker(
            paths,
            spec,
            timeout_seconds=30,
            process_factory=lambda *args, **kwargs: process,
            utc_now=iter_values(
                "2026-07-27T00:00:00+00:00",
                "2026-07-27T00:00:01+00:00",
            ),
            monotonic_ns=iter_values(1_000_000_000),
            session_id_factory=lambda: "session-launch-telemetry-failure",
            git_head=lambda _path: "b" * 40,
            termination_grace_seconds=2,
            kill_reap_seconds=3,
        )

    assert raised.value is telemetry_error
    assert process.terminated is True
    assert process.killed is True
    assert process.wait_timeouts == [2, 3]


def test_run_worker_terminates_and_reraises_keyboard_interrupt(tmp_path: Path):
    class InterruptedProcess:
        pid = 4242
        returncode = None
        terminated = False
        killed = False
        wait_timeouts = []

        def wait(self, timeout=None):
            self.wait_timeouts.append(timeout)
            if not self.terminated:
                assert timeout == 30
                raise KeyboardInterrupt
            if not self.killed:
                raise worker_guard.subprocess.TimeoutExpired("codex", timeout)
            self.returncode = -9
            return self.returncode

        def terminate(self):
            self.terminated = True

        def kill(self):
            self.killed = True

    paths, spec = worker_run_fixture(tmp_path)
    process = InterruptedProcess()

    with pytest.raises(KeyboardInterrupt):
        worker_guard.run_worker(
            paths,
            spec,
            timeout_seconds=30,
            process_factory=lambda *args, **kwargs: process,
            utc_now=iter_values(
                "2026-07-27T00:00:00+00:00",
                "2026-07-27T00:00:01+00:00",
                "2026-07-27T00:00:06+00:00",
            ),
            monotonic_ns=iter_values(1_000_000_000, 6_000_000_000),
            session_id_factory=lambda: "session-interrupted",
            git_head=lambda _path: "c" * 40,
            termination_grace_seconds=2,
            kill_reap_seconds=3,
        )

    exit_event = worker_guard.telemetry.read_session(
        paths, "session-interrupted"
    )[-1]
    assert process.terminated is True
    assert process.killed is True
    assert process.wait_timeouts == [30, 2, 3]
    assert exit_event["exit_code"] is None
    assert exit_event["signal"] == 9
    assert exit_event["outcome"] == "interrupted"


def test_run_worker_terminates_timed_out_child(tmp_path: Path):
    class TimedOutProcess:
        pid = 4242
        returncode = None
        terminated = False
        killed = False
        wait_timeouts = []

        def wait(self, timeout=None):
            self.wait_timeouts.append(timeout)
            if not self.terminated:
                assert timeout == 30
                raise worker_guard.subprocess.TimeoutExpired("codex", timeout)
            if not self.killed:
                raise worker_guard.subprocess.TimeoutExpired("codex", timeout)
            self.returncode = -9
            return self.returncode

        def terminate(self):
            self.terminated = True

        def kill(self):
            self.killed = True

    paths, spec = worker_run_fixture(tmp_path)
    process = TimedOutProcess()

    result = worker_guard.run_worker(
        paths,
        spec,
        timeout_seconds=30,
        process_factory=lambda *args, **kwargs: process,
        utc_now=iter_values(
            "2026-07-27T00:00:00+00:00",
            "2026-07-27T00:00:01+00:00",
            "2026-07-27T00:00:06+00:00",
        ),
        monotonic_ns=iter_values(1_000_000_000, 6_000_000_000),
        session_id_factory=lambda: "session-timeout",
        git_head=lambda _path: "d" * 40,
        termination_grace_seconds=2,
        kill_reap_seconds=3,
    )

    exit_event = worker_guard.telemetry.read_session(
        paths, "session-timeout"
    )[-1]
    assert process.terminated is True
    assert process.killed is True
    assert process.wait_timeouts == [30, 2, 3]
    assert exit_event["signal"] == 9
    assert exit_event["outcome"] == "timed_out"
    assert result["elapsed_seconds"] == 5.0


def test_run_worker_does_not_invent_duration_when_killed_child_cannot_reap(
    tmp_path: Path,
):
    class UnreapedProcess:
        pid = 4242
        returncode = None
        wait_timeouts = []

        def wait(self, timeout=None):
            self.wait_timeouts.append(timeout)
            raise worker_guard.subprocess.TimeoutExpired("codex", timeout)

        def terminate(self):
            pass

        def kill(self):
            pass

    paths, spec = worker_run_fixture(tmp_path)
    process = UnreapedProcess()

    result = worker_guard.run_worker(
        paths,
        spec,
        timeout_seconds=30,
        process_factory=lambda *args, **kwargs: process,
        utc_now=iter_values(
            "2026-07-27T00:00:00+00:00",
            "2026-07-27T00:00:01+00:00",
            "2026-07-27T00:00:06+00:00",
        ),
        monotonic_ns=iter_values(1_000_000_000),
        session_id_factory=lambda: "session-unreaped",
        git_head=lambda _path: "e" * 40,
        termination_grace_seconds=2,
        kill_reap_seconds=3,
    )

    exit_event = worker_guard.telemetry.read_session(
        paths, "session-unreaped"
    )[-1]
    assert process.wait_timeouts == [30, 2, 3]
    assert exit_event["outcome"] == "timed_out"
    assert exit_event["monotonic_ns"] is None
    assert result["elapsed_seconds"] is None


@pytest.mark.parametrize(
    ("downloaded_bytes", "expected"),
    [(2048, 2048), (None, None)],
)
def test_run_worker_records_runtime_download_bytes_when_available(
    tmp_path: Path,
    downloaded_bytes: int | None,
    expected: int | None,
):
    class DownloadProcess(CompletedProcess):
        pass

    process = DownloadProcess()
    if downloaded_bytes is not None:
        process.downloaded_bytes = downloaded_bytes
    paths, spec = worker_run_fixture(tmp_path)

    worker_guard.run_worker(
        paths,
        spec,
        timeout_seconds=30,
        process_factory=lambda *args, **kwargs: process,
        utc_now=iter_values(
            "2026-07-27T00:00:00+00:00",
            "2026-07-27T00:00:01+00:00",
            "2026-07-27T00:00:06+00:00",
        ),
        monotonic_ns=iter_values(1_000_000_000, 6_000_000_000),
        session_id_factory=lambda: "session-download",
        git_head=lambda _path: "e" * 40,
    )

    exit_event = worker_guard.telemetry.read_session(
        paths, "session-download"
    )[-1]
    assert exit_event["downloaded_bytes"] == expected


def worker_run_fixture(
    tmp_path: Path,
) -> tuple[store.StatePaths, worker_guard.LaunchSpec]:
    worktree = tmp_path / "paper-worktree"
    worktree.mkdir()
    contract = write_contract(worktree)
    pass_preflight("codex", worktree)
    return (
        store.StatePaths(tmp_path / "repro-loop.json"),
        launch_spec("codex", "model-a", worktree, contract),
    )


def iter_values(*values):
    iterator = iter(values)
    return lambda: next(iterator)
