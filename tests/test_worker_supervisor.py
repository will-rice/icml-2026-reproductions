import json
from pathlib import Path
import subprocess

import pytest

from ops import worker_supervisor as supervisor


class FakeHost:
    def __init__(self):
        self.health = {}
        self.created = []
        self.sent = []

    def session_health(self, spec):
        return self.health.get(
            spec.session_name, supervisor.SessionHealth(False, False, "", "")
        )

    def ensure_session(self, spec, command):
        if not self.session_health(spec).exists:
            self.created.append(spec.session_name)
        self.sent.append((spec.session_name, command))


def test_desired_workers_are_stable_ten_agy_and_five_codex():
    workers = supervisor.desired_workers()
    assert [w.worker_id for w in workers[:10]] == [
        f"agy-paper-owner-{index:02d}" for index in range(1, 11)
    ]
    assert [w.worker_id for w in workers[10:]] == [
        f"codex-paper-owner-{index:02d}" for index in range(1, 6)
    ]
    assert len({w.session_name for w in workers}) == 15


def test_agy_launch_is_direct_full_approval_and_credential_safe():
    spec = supervisor.desired_workers()[0]
    profile = supervisor.agy_profiles()[0]
    command = supervisor.launch_shell_command(spec, profile, Path("/repo"))
    assert " opencode " not in f" {command} "
    assert " agy " in f" {command} "
    assert "--dangerously-skip-permissions" in command
    assert "$(hf auth token)" in command
    assert "$(gh auth token)" in command
    assert "Use the shared icml-repro-loop skill directly" in command
    assert spec.worker_id in command


def test_codex_launch_has_workspace_network_and_direct_prompt():
    spec = supervisor.desired_workers()[10]
    command = supervisor.launch_shell_command(
        spec, supervisor.codex_profile(), Path("/repo")
    )
    assert " codex exec " in f" {command} "
    assert "--sandbox workspace-write" in command
    assert "sandbox_workspace_write.network_access=true" in command
    assert "-C /repo" in command
    assert spec.worker_id in command


def test_sanitize_text_redacts_tokens_and_environment_assignments():
    raw = (
        "HF_TOKEN=hf_abcdefghijklmnopqrstuvwxyz "
        "GH_TOKEN=github_pat_abcdefghijklmnopqrstuvwxyz "
        "Authorization: Bearer secret-value"
    )
    clean = supervisor.sanitize_text(raw)
    assert "hf_abcdefghijklmnopqrstuvwxyz" not in clean
    assert "github_pat_abcdefghijklmnopqrstuvwxyz" not in clean
    assert "secret-value" not in clean
    assert clean.count("<redacted>") == 3


def test_only_expected_live_foreground_agent_is_healthy():
    agy = supervisor.desired_workers()[0]
    assert supervisor.is_healthy(
        agy, supervisor.SessionHealth(True, False, "agy", "")
    )
    assert not supervisor.is_healthy(
        agy, supervisor.SessionHealth(True, False, "bash", "")
    )
    assert not supervisor.is_healthy(
        agy, supervisor.SessionHealth(True, True, "agy", "")
    )
    assert not supervisor.is_healthy(
        agy, supervisor.SessionHealth(False, False, "", "")
    )


def test_host_adapter_reads_tmux_health_with_argument_arrays(monkeypatch):
    calls = []

    def run(argv, *, check, text, capture_output):
        assert isinstance(argv, list)
        assert check is False
        assert text is True
        assert capture_output is True
        calls.append(argv)
        if argv[1] == "list-panes":
            return subprocess.CompletedProcess(argv, 0, "0\tagy\t123\n", "")
        return subprocess.CompletedProcess(argv, 0, "recent output\n", "")

    monkeypatch.setattr(supervisor.subprocess, "run", run)

    health = supervisor.HostAdapter(Path("/repo")).session_health(
        supervisor.desired_workers()[0]
    )

    assert health == supervisor.SessionHealth(True, False, "agy", "recent output\n")
    assert calls == [
        [
            "tmux",
            "list-panes",
            "-t",
            "agy-paper-owner-01",
            "-F",
            "#{pane_dead}\t#{pane_current_command}\t#{pane_pid}",
        ],
        ["tmux", "capture-pane", "-pt", "agy-paper-owner-01", "-S", "-80"],
    ]


def test_host_adapter_missing_session_and_sanitized_failures(monkeypatch):
    spec = supervisor.desired_workers()[0]

    def missing_run(argv, *, check, text, capture_output):
        assert isinstance(argv, list)
        return subprocess.CompletedProcess(argv, 1, "", "can't find session: missing")

    monkeypatch.setattr(supervisor.subprocess, "run", missing_run)
    assert supervisor.HostAdapter(Path("/repo")).session_health(spec) == supervisor.SessionHealth(
        False, False, "", ""
    )

    def failed_run(argv, *, check, text, capture_output):
        return subprocess.CompletedProcess(
            argv, 1, "", "tmux denied hf_abcdefghijklmnopqrstuvwxyz"
        )

    monkeypatch.setattr(supervisor.subprocess, "run", failed_run)
    with pytest.raises(supervisor.HostCommandError) as exc_info:
        supervisor.HostAdapter(Path("/repo")).session_health(spec)
    assert "hf_abcdefghijklmnopqrstuvwxyz" not in str(exc_info.value)
    assert "<redacted>" in str(exc_info.value)


def test_host_adapter_creates_only_missing_session_then_sends_command(monkeypatch):
    calls = []

    def run(argv, *, check, text, capture_output):
        assert isinstance(argv, list)
        calls.append(argv)
        if argv[1] == "list-panes":
            return subprocess.CompletedProcess(argv, 1, "", "can't find session: missing")
        return subprocess.CompletedProcess(argv, 0, "", "")

    monkeypatch.setattr(supervisor.subprocess, "run", run)
    spec = supervisor.desired_workers()[0]

    supervisor.HostAdapter(Path("/repo")).ensure_session(spec, "exec agy")

    assert calls == [
        [
            "tmux",
            "list-panes",
            "-t",
            "agy-paper-owner-01",
            "-F",
            "#{pane_dead}\t#{pane_current_command}\t#{pane_pid}",
        ],
        [
            "tmux",
            "new-session",
            "-d",
            "-s",
            "agy-paper-owner-01",
            "-c",
            "/repo",
        ],
        ["tmux", "send-keys", "-t", "agy-paper-owner-01", "exec agy", "C-m"],
    ]


def test_ensure_session_adopts_live_process_and_reuses_idle_shell():
    fake_host = FakeHost()
    live, idle, missing = supervisor.desired_workers()[:3]
    fake_host.health[live.session_name] = supervisor.SessionHealth(
        True, False, "agy", ""
    )
    fake_host.health[idle.session_name] = supervisor.SessionHealth(
        True, False, "bash", ""
    )
    fake_host.health[missing.session_name] = supervisor.SessionHealth(
        False, False, "", ""
    )

    assert supervisor.is_healthy(live, fake_host.session_health(live))
    fake_host.ensure_session(idle, "exec agy")
    fake_host.ensure_session(missing, "exec agy")

    assert fake_host.created == [missing.session_name]
    assert fake_host.sent == [
        (idle.session_name, "exec agy"),
        (missing.session_name, "exec agy"),
    ]


def test_atomic_status_replaces_complete_json(tmp_path):
    path = tmp_path / "status.json"
    supervisor.atomic_write_json(path, {"workers": [{"id": "one"}]})
    assert json.loads(path.read_text())["workers"] == [{"id": "one"}]
    assert list(tmp_path.glob(".status.json.*.tmp")) == []


def test_exclusive_lock_rejects_second_holder(tmp_path):
    with supervisor.exclusive_lock(tmp_path / "supervisor.lock"):
        with pytest.raises(supervisor.AlreadyRunning):
            with supervisor.exclusive_lock(tmp_path / "supervisor.lock"):
                pass


def test_runtime_directory_uses_xdg_state_home(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path))

    runtime_path = supervisor.runtime_directory()

    assert runtime_path == tmp_path / "icml-worker-supervisor"
    assert runtime_path.is_dir()
