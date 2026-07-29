import json
from datetime import datetime, timedelta, timezone
import os
from pathlib import Path
import subprocess
import threading
import time

import pytest

from ops import worker_supervisor as supervisor


FIXED_NOW = datetime(2026, 7, 29, 12, 0, tzinfo=timezone.utc)


def test_remote_setup_documents_direct_supervisor_operations():
    text = Path("docs/REMOTE_SETUP.md").read_text()
    assert "ops/worker_supervisor.py install" in text
    assert "ops/worker_supervisor.py status" in text
    assert "ops/worker_supervisor.py reconcile --dry-run" in text
    assert "10 Agy" in text
    assert "5 Codex" in text
    assert "OpenCode" in text


class FakeHost:
    def __init__(self):
        self.health = {}
        self.created = []
        self.sent = []
        self.events = []
        self.stopped = []
        self.restored = []
        self.production_pids = {}
        self.disposable_foreground = {}
        self.restore_succeeds = True
        self.mutate_production_pid_on_restore = False
        self.smoke_requests_seen = []
        self.service_quiesced = threading.Event()
        self.stop_lock_path = None
        self.inflight_reconcile = False
        self.health_errors = set()
        self.ensure_errors = set()
        self.restore_error = False

    def session_health(self, spec):
        if "reconcile" not in self.events:
            self.events.append("reconcile")
        if spec.session_name in self.health_errors:
            raise supervisor.HostCommandError("tmux health lookup failed")
        return self.health.get(
            spec.session_name, supervisor.SessionHealth(False, False, "", "")
        )

    def ensure_session(self, spec, command):
        health = self.session_health(spec)
        if supervisor.is_healthy(spec, health):
            return False
        if spec.session_name in self.ensure_errors:
            raise supervisor.HostCommandError("tmux launch failed")
        if not health.exists:
            self.created.append(spec.session_name)
        self.sent.append((spec.session_name, command))
        launch_marker = ""
        if supervisor.LAUNCH_MARKER_PREFIX in command:
            marker_suffix = command.split(
                supervisor.LAUNCH_MARKER_PREFIX, maxsplit=1
            )[1]
            launch_marker = marker_suffix.split(";", maxsplit=1)[0]
        self.health[spec.session_name] = supervisor.SessionHealth(
            True, False, spec.agent, "", launch_marker
        )
        if self.inflight_reconcile:
            self.events.append(f"recreate {spec.session_name}")
        return True

    def mark_started(self, worker_ids):
        by_worker_id = {spec.worker_id: spec for spec in supervisor.desired_workers()}
        for worker_id in worker_ids:
            spec = by_worker_id[worker_id]
            self.health[spec.session_name] = supervisor.SessionHealth(
                True, False, spec.agent, ""
            )

    def systemctl_user(self, *arguments):
        self.events.append(f"systemctl --user {' '.join(arguments)}")
        if arguments == ("stop", "icml-worker-supervisor.service"):
            self.service_quiesced.set()

    def stop_session(self, session_name):
        if self.stop_lock_path is not None:
            try:
                with supervisor.exclusive_lock(self.stop_lock_path):
                    lock_held = False
            except supervisor.AlreadyRunning:
                lock_held = True
            self.events.append("lock-held" if lock_held else "lock-not-held")
        self.events.append(f"stop {session_name}")
        self.stopped.append(session_name)

    def pane_pid(self, session_name):
        if session_name in self.production_pids:
            return self.production_pids[session_name]
        if session_name in self.disposable_foreground:
            return 999
        raise supervisor.HostCommandError(f"missing session: {session_name}")

    def create_disposable_session(self, session_name, command):
        assert session_name == "icml-supervisor-smoke-test"
        assert command == "/usr/bin/sleep 300"
        self.disposable_foreground[session_name] = "sleep"

    def interrupt_session(self, session_name):
        assert session_name == "icml-supervisor-smoke-test"
        self.disposable_foreground[session_name] = "bash"

    def restore_disposable_session(self, session_name, command):
        assert session_name == "icml-supervisor-smoke-test"
        assert command == "exec /usr/bin/sleep 300"
        if self.restore_error:
            raise supervisor.HostCommandError("missing disposable smoke session")
        self.restored.append(session_name)
        if self.restore_succeeds:
            self.disposable_foreground[session_name] = "sleep"
        if self.mutate_production_pid_on_restore and self.production_pids:
            first = next(iter(self.production_pids))
            self.production_pids[first] += 1

    def session_foreground_command(self, session_name):
        return self.disposable_foreground.get(session_name, "")

    def wait_for_smoke_restore(self, request_path, nonce, timeout):
        self.smoke_requests_seen.append(
            json.loads(request_path.read_text(encoding="utf-8"))
        )
        home = request_path.parents[3]
        supervisor.main(
            ["reconcile"], host=self, home=home, now=FIXED_NOW
        )
        request = json.loads(request_path.read_text(encoding="utf-8"))
        return request == {"nonce": nonce, "stage": "restored"}


@pytest.fixture
def fake_host():
    return FakeHost()


@pytest.fixture
def fixed_now():
    return FIXED_NOW


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


def test_launch_resolves_user_clis_with_restricted_service_path(tmp_path):
    bin_dir = tmp_path / ".local" / "bin"
    bin_dir.mkdir(parents=True)
    scripts = {
        "hf": "#!/bin/sh\nprintf 'hf-token\\n'\n",
        "gh": "#!/bin/sh\nprintf 'gh-token\\n'\n",
        "worker-probe": (
            "#!/bin/sh\n"
            'printf "%s|%s\\n" "$HF_TOKEN" "$GH_TOKEN" > "$HOME/result"\n'
        ),
    }
    for name, script in scripts.items():
        path = bin_dir / name
        path.write_text(script, encoding="utf-8")
        path.chmod(0o755)

    command = supervisor.launch_shell_command(
        supervisor.WorkerSpec("probe", "codex", "probe"),
        supervisor.ModelProfile("probe", ("worker-probe",)),
        Path("/repo"),
    )
    result = subprocess.run(
        ["/bin/bash", "-c", command],
        check=False,
        capture_output=True,
        text=True,
        env={"HOME": str(tmp_path), "PATH": "/usr/bin:/bin"},
    )

    assert result.returncode == 0, result.stderr
    assert (tmp_path / "result").read_text(encoding="utf-8") == (
        "hf-token|gh-token\n"
    )


def test_launch_replaces_wrapper_shell_with_worker_process(tmp_path):
    bin_dir = tmp_path / ".local" / "bin"
    bin_dir.mkdir(parents=True)
    scripts = {
        "hf": "#!/bin/sh\nprintf 'hf-token\\n'\n",
        "gh": "#!/bin/sh\nprintf 'gh-token\\n'\n",
        "worker-probe": (
            "#!/bin/sh\n"
            ': > "$HOME/ready"\n'
            "exec /bin/sleep 30\n"
        ),
    }
    for name, script in scripts.items():
        path = bin_dir / name
        path.write_text(script, encoding="utf-8")
        path.chmod(0o755)

    command = supervisor.launch_shell_command(
        supervisor.WorkerSpec("probe", "codex", "probe"),
        supervisor.ModelProfile("probe", ("worker-probe",)),
        Path("/repo"),
    )
    process = subprocess.Popen(
        ["/bin/bash", "-c", f"{command}; exec /bin/bash"],
        env={"HOME": str(tmp_path), "PATH": "/usr/bin:/bin"},
    )
    try:
        deadline = time.monotonic() + 2
        while not (tmp_path / "ready").exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert (tmp_path / "ready").exists()
        observed = subprocess.run(
            ["ps", "-o", "comm=", "-p", str(process.pid)],
            check=True,
            capture_output=True,
            text=True,
        )
        assert observed.stdout.strip() == "sleep"
    finally:
        process.terminate()
        process.wait(timeout=2)


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
            return subprocess.CompletedProcess(argv, 0, "0\tagy\t123\t\n", "")
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
            (
                "#{pane_dead}\t#{pane_current_command}\t#{pane_pid}"
                "\t#{pane_start_command}"
            ),
        ],
        ["tmux", "capture-pane", "-pt", "agy-paper-owner-01", "-S", "-80"],
    ]


def test_host_adapter_extracts_launch_generation_without_retaining_start_command(
    monkeypatch,
):
    marker = "0123456789abcdef"
    raw_start_command = (
        f"printf '%s' {supervisor.LAUNCH_MARKER_PREFIX}{marker}; "
        "HF_TOKEN=hf_secret agy secret prompt"
    )

    def run(argv, *, check, text, capture_output):
        if argv[1] == "list-panes":
            return subprocess.CompletedProcess(
                argv,
                0,
                f"0\tbash\t123\t{raw_start_command}\n",
                "",
            )
        return subprocess.CompletedProcess(
            argv, 0, "process exited with code 1\n", ""
        )

    monkeypatch.setattr(supervisor.subprocess, "run", run)

    health = supervisor.HostAdapter(Path("/repo")).session_health(
        supervisor.desired_workers()[0]
    )

    assert health.launch_marker == marker
    assert not hasattr(health, "start_command")
    assert raw_start_command not in repr(health)


@pytest.mark.parametrize(
    "stderr",
    [
        "can't find session: agy-paper-owner-01",
        "can't find window: agy-paper-owner-01",
        "no server running on /tmp/tmux-1000/default",
    ],
)
def test_host_adapter_maps_known_missing_targets_to_absent(monkeypatch, stderr):
    spec = supervisor.desired_workers()[0]

    def missing_run(argv, *, check, text, capture_output):
        assert isinstance(argv, list)
        return subprocess.CompletedProcess(argv, 1, "", stderr)

    monkeypatch.setattr(supervisor.subprocess, "run", missing_run)
    assert supervisor.HostAdapter(Path("/repo")).session_health(spec) == supervisor.SessionHealth(
        False, False, "", ""
    )


def test_host_adapter_unrelated_failure_is_sanitized(monkeypatch):
    spec = supervisor.desired_workers()[0]

    def failed_run(argv, *, check, text, capture_output):
        return subprocess.CompletedProcess(
            argv, 1, "", "tmux denied hf_abcdefghijklmnopqrstuvwxyz"
        )

    monkeypatch.setattr(supervisor.subprocess, "run", failed_run)
    with pytest.raises(supervisor.HostCommandError) as exc_info:
        supervisor.HostAdapter(Path("/repo")).session_health(spec)
    assert "hf_abcdefghijklmnopqrstuvwxyz" not in str(exc_info.value)
    assert "<redacted>" in str(exc_info.value)


def test_host_adapter_starts_missing_session_directly_in_repo(monkeypatch):
    calls = []

    def run(argv, *, check, text, capture_output):
        assert isinstance(argv, list)
        calls.append(argv)
        if argv[1] == "list-panes":
            return subprocess.CompletedProcess(argv, 1, "", "can't find session: missing")
        return subprocess.CompletedProcess(argv, 0, "", "")

    monkeypatch.setattr(supervisor.subprocess, "run", run)
    spec = supervisor.desired_workers()[0]

    started = supervisor.HostAdapter(Path("/repo")).ensure_session(
        spec, "exec agy"
    )

    assert started is True
    assert calls == [
        [
            "tmux",
            "list-panes",
            "-t",
            "agy-paper-owner-01",
            "-F",
            (
                "#{pane_dead}\t#{pane_current_command}\t#{pane_pid}"
                "\t#{pane_start_command}"
            ),
        ],
        [
            "tmux",
            "new-session",
            "-d",
            "-s",
            "agy-paper-owner-01",
            "-c",
            "/repo",
            "exec agy; exec /bin/bash",
        ],
    ]


@pytest.mark.parametrize(
    ("pane_dead", "foreground"),
    [
        ("1", "bash"),
        ("0", "python"),
    ],
)
def test_host_adapter_respawns_dead_or_non_shell_pane_in_repo(
    monkeypatch, pane_dead, foreground
):
    calls = []

    def run(argv, *, check, text, capture_output):
        calls.append(argv)
        if argv[1] == "list-panes":
            return subprocess.CompletedProcess(
                argv, 0, f"{pane_dead}\t{foreground}\t123\t\n", ""
            )
        if argv[1] == "capture-pane":
            return subprocess.CompletedProcess(argv, 0, "old history\n", "")
        return subprocess.CompletedProcess(argv, 0, "", "")

    monkeypatch.setattr(supervisor.subprocess, "run", run)
    spec = supervisor.desired_workers()[0]

    started = supervisor.HostAdapter(Path("/repo")).ensure_session(
        spec, "exec agy"
    )

    assert started is True
    assert calls[-1] == [
        "tmux",
        "respawn-pane",
        "-k",
        "-t",
        spec.session_name,
        "-c",
        "/repo",
        "exec agy; exec /bin/bash",
    ]
    assert not any(call[1] == "send-keys" for call in calls)


def test_host_adapter_adopts_lane_that_becomes_healthy_before_mutation(
    monkeypatch,
):
    calls = []
    observations = iter(("bash", "agy"))

    def run(argv, *, check, text, capture_output):
        calls.append(argv)
        if argv[1] == "list-panes":
            foreground = next(observations)
            return subprocess.CompletedProcess(
                argv, 0, f"0\t{foreground}\t123\t\n", ""
            )
        if argv[1] == "capture-pane":
            return subprocess.CompletedProcess(argv, 0, "", "")
        raise AssertionError(f"unexpected mutation: {argv}")

    monkeypatch.setattr(supervisor.subprocess, "run", run)
    host = supervisor.HostAdapter(Path("/repo"))
    spec = supervisor.desired_workers()[0]

    assert not supervisor.is_healthy(spec, host.session_health(spec))
    started = host.ensure_session(spec, "exec agy")

    assert started is False
    assert not any(
        call[1] in {"new-session", "respawn-pane", "send-keys"}
        for call in calls
    )


def test_stop_session_treats_no_tmux_server_as_idempotent_success(monkeypatch):
    def run(argv, *, check, text, capture_output):
        return subprocess.CompletedProcess(
            argv, 1, "", "no server running on /tmp/tmux-1000/default"
        )

    monkeypatch.setattr(supervisor.subprocess, "run", run)

    supervisor.HostAdapter(Path("/repo")).stop_session("agy-paper-owner-01")


def test_fake_host_contract_adopts_live_and_starts_idle_or_missing():
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


def test_reconcile_does_not_restart_healthy_workers(fake_host, fixed_now):
    for spec in supervisor.desired_workers():
        fake_host.health[spec.session_name] = supervisor.SessionHealth(
            True, False, spec.agent, ""
        )
    result = supervisor.reconcile(
        fake_host, supervisor.RuntimeState.empty(), fixed_now, Path("/repo")
    )
    assert result.started == ()
    assert fake_host.sent == []
    assert len(result.status["workers"]) == 15


def test_two_reconciliations_do_not_duplicate_started_lane(fake_host, fixed_now):
    state = supervisor.RuntimeState.empty()
    first = supervisor.reconcile(fake_host, state, fixed_now, Path("/repo"))
    fake_host.mark_started(first.started)
    second = supervisor.reconcile(
        fake_host, first.state, fixed_now + timedelta(seconds=30), Path("/repo")
    )
    assert len(first.started) == 15
    assert second.started == ()


def test_quota_error_rotates_profile_and_honors_reported_reset(fake_host, fixed_now):
    spec = supervisor.desired_workers()[0]
    for other_spec in supervisor.desired_workers()[1:]:
        fake_host.health[other_spec.session_name] = supervisor.SessionHealth(
            True, False, other_spec.agent, ""
        )
    fake_host.health[spec.session_name] = supervisor.SessionHealth(
        True,
        False,
        "bash",
        "Individual quota reached. Resets in 46m20s.",
    )
    state = supervisor.RuntimeState.with_lane(spec.worker_id, profile_index=0)
    result = supervisor.reconcile(fake_host, state, fixed_now, Path("/repo"))
    lane = result.state.lanes[spec.worker_id]
    assert lane.profile_index == 1
    assert lane.profile_backoff["gemini-3.1-pro-high"] == (
        fixed_now + timedelta(minutes=46, seconds=20)
    )
    assert result.started == (spec.worker_id,)


def test_all_agy_profiles_exhausted_waits_for_earliest_reset(fake_host, fixed_now):
    spec = supervisor.desired_workers()[0]
    resets = {
        profile.name: fixed_now + timedelta(minutes=minutes)
        for profile, minutes in zip(supervisor.agy_profiles(), (12, 8, 20))
    }
    state = supervisor.RuntimeState.with_lane(
        spec.worker_id, profile_index=0, profile_backoff=resets
    )
    result = supervisor.reconcile(fake_host, state, fixed_now, Path("/repo"))
    lane = result.state.lanes[spec.worker_id]
    assert spec.worker_id not in result.started
    assert lane.next_retry_at == fixed_now + timedelta(minutes=8)


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        (
            "Individual quota reached. Resets in 50m33s.",
            timedelta(minutes=50, seconds=33),
        ),
        ("Individual quota reached. Resets in 2h3m.", timedelta(hours=2, minutes=3)),
        ("process exited with code 1", None),
    ],
)
def test_parse_quota_reset_formats(message, expected, fixed_now):
    parsed = supervisor.parse_quota_reset(message, fixed_now)
    assert parsed == (None if expected is None else fixed_now + expected)


def test_ordinary_crash_uses_bounded_worker_specific_backoff(fake_host, fixed_now):
    spec = supervisor.desired_workers()[10]
    fake_host.health[spec.session_name] = supervisor.SessionHealth(
        True, False, "bash", "process exited with code 1"
    )
    state = supervisor.RuntimeState.with_lane(spec.worker_id)
    result = supervisor.reconcile(fake_host, state, fixed_now, Path("/repo"))
    lane = result.state.lanes[spec.worker_id]
    assert lane.ordinary_failures == 1
    assert fixed_now + timedelta(seconds=15) <= lane.next_retry_at
    assert lane.next_retry_at <= fixed_now + timedelta(seconds=25)


def test_healthy_worker_resets_ordinary_failure_backoff(fake_host, fixed_now):
    spec = supervisor.desired_workers()[10]
    fake_host.health[spec.session_name] = supervisor.SessionHealth(
        True, False, "codex", ""
    )
    retry_at = fixed_now + timedelta(minutes=1)
    initial = supervisor.RuntimeState(
        {
            spec.worker_id: supervisor.LaneState(
                ordinary_failures=2, next_retry_at=retry_at
            )
        }
    )
    result = supervisor.reconcile(fake_host, initial, fixed_now, Path("/repo"))
    lane = result.state.lanes[spec.worker_id]
    assert lane.ordinary_failures == 0
    assert lane.next_retry_at is None


def test_dry_run_reports_without_host_mutation_or_state_change(
    fake_host, fixed_now
):
    initial = supervisor.RuntimeState.empty()
    result = supervisor.reconcile(
        fake_host, initial, fixed_now, Path("/repo"), dry_run=True
    )
    assert len(result.proposed) == 15
    assert result.started == ()
    assert result.state == initial
    assert fake_host.created == []
    assert fake_host.sent == []


def test_status_contains_no_prompt_token_or_environment_dump(fake_host, fixed_now):
    spec = supervisor.desired_workers()[0]
    fake_host.health[spec.session_name] = supervisor.SessionHealth(
        True,
        False,
        "bash",
        "HF_TOKEN=hf_secret GH_TOKEN=github_pat_secret",
    )
    result = supervisor.reconcile(
        fake_host, supervisor.RuntimeState.empty(), fixed_now, Path("/repo")
    )
    encoded = json.dumps(result.status)
    assert "Use the shared" not in encoded
    assert "HF_TOKEN" not in encoded
    assert "hf_secret" not in encoded
    assert "github_pat_" not in encoded


def test_status_classifies_pane_output_without_persisting_prompt_or_command(
    fake_host, fixed_now
):
    spec = supervisor.desired_workers()[0]
    command = supervisor.launch_shell_command(
        spec, supervisor.agy_profiles()[0], Path("/repo")
    )
    pane_output = (
        "AWS_ACCESS_KEY_ID=AKIAEXAMPLE DATABASE_PASSWORD=correct-horse "
        f"{supervisor.PROMPT.format(worker_id=spec.worker_id)} {command}"
    )
    fake_host.health[spec.session_name] = supervisor.SessionHealth(
        True, False, "bash", pane_output
    )
    result = supervisor.reconcile(
        fake_host, supervisor.RuntimeState.empty(), fixed_now, Path("/repo")
    )
    worker = result.status["workers"][0]
    encoded = json.dumps(result.status)
    assert worker["last_error"] == "unhealthy-session"
    assert "Use the shared" not in encoded
    assert "--dangerously-skip-permissions" not in encoded
    assert "AWS_ACCESS_KEY_ID" not in encoded
    assert "DATABASE_PASSWORD" not in encoded
    assert "correct-horse" not in encoded


@pytest.mark.parametrize("profile_index", (1, 2))
def test_ordinary_agy_restart_retains_persisted_profile(
    fake_host, fixed_now, profile_index
):
    spec = supervisor.desired_workers()[0]
    fake_host.health[spec.session_name] = supervisor.SessionHealth(
        True, False, "bash", "process exited with code 1"
    )
    state = supervisor.RuntimeState.with_lane(spec.worker_id, profile_index=profile_index)
    supervisor.reconcile(fake_host, state, fixed_now, Path("/repo"))
    sent_command = next(command for session, command in fake_host.sent if session == spec.session_name)
    assert supervisor.agy_profiles()[profile_index].name in sent_command


def test_stale_quota_output_is_consumed_only_once(fake_host, fixed_now):
    spec = supervisor.desired_workers()[0]
    quota_output = "Individual quota reached. Resets in 46m20s."
    fake_host.health[spec.session_name] = supervisor.SessionHealth(
        True, False, "bash", quota_output
    )
    initial = supervisor.RuntimeState.with_lane(spec.worker_id, profile_index=0)
    first = supervisor.reconcile(fake_host, initial, fixed_now, Path("/repo"))
    second = supervisor.reconcile(
        fake_host, first.state, fixed_now + timedelta(seconds=30), Path("/repo")
    )
    lane = second.state.lanes[spec.worker_id]
    assert lane.profile_index == 1
    assert set(lane.profile_backoff) == {"gemini-3.1-pro-high"}


def test_stale_quota_record_with_new_terminal_output_is_not_reprocessed(
    fake_host, fixed_now
):
    spec = supervisor.desired_workers()[0]
    quota_output = "Individual quota reached. Resets in 46m20s."
    fake_host.health[spec.session_name] = supervisor.SessionHealth(
        True, False, "bash", quota_output
    )
    first = supervisor.reconcile(
        fake_host,
        supervisor.RuntimeState.with_lane(spec.worker_id, profile_index=0),
        fixed_now,
        Path("/repo"),
    )
    fake_host.health[spec.session_name] = supervisor.SessionHealth(
        True, False, "bash", f"{quota_output}\nordinary terminal output"
    )
    second = supervisor.reconcile(
        fake_host, first.state, fixed_now + timedelta(seconds=30), Path("/repo")
    )
    lane = second.state.lanes[spec.worker_id]
    assert lane.profile_index == 1
    assert lane.profile_backoff == {
        "gemini-3.1-pro-high": fixed_now + timedelta(minutes=46, seconds=20)
    }


def test_identical_quota_failures_are_attributed_to_successive_profile_launches(
    fake_host, fixed_now
):
    spec = supervisor.desired_workers()[0]
    for other_spec in supervisor.desired_workers()[1:]:
        fake_host.health[other_spec.session_name] = supervisor.SessionHealth(
            True, False, other_spec.agent, ""
        )
    quota_output = "Individual quota reached. Resets in 46m20s."
    fake_host.health[spec.session_name] = supervisor.SessionHealth(
        True, False, "bash", quota_output
    )

    first = supervisor.reconcile(
        fake_host,
        supervisor.RuntimeState.with_lane(spec.worker_id, profile_index=0),
        fixed_now,
        Path("/repo"),
    )
    first_marker = first.state.lanes[spec.worker_id].launch_marker
    fake_host.health[spec.session_name] = supervisor.SessionHealth(
        True,
        False,
        "bash",
        "\n".join(
            (
                quota_output,
                f"{supervisor.LAUNCH_MARKER_PREFIX}{first_marker}",
                quota_output,
            )
        ),
        first_marker,
    )

    second = supervisor.reconcile(
        fake_host, first.state, fixed_now + timedelta(seconds=30), Path("/repo")
    )

    lane = second.state.lanes[spec.worker_id]
    assert lane.profile_index == 2
    assert set(lane.profile_backoff) == {
        "gemini-3.1-pro-high",
        "gemini-3.6-flash-high",
    }


def test_stale_quota_before_current_launch_marker_does_not_poison_ordinary_exit(
    fake_host, fixed_now
):
    spec = supervisor.desired_workers()[0]
    for other_spec in supervisor.desired_workers()[1:]:
        fake_host.health[other_spec.session_name] = supervisor.SessionHealth(
            True, False, other_spec.agent, ""
        )
    marker = "profile-b-launch"
    old_reset = fixed_now + timedelta(minutes=46, seconds=20)
    fake_host.health[spec.session_name] = supervisor.SessionHealth(
        True,
        False,
        "bash",
        "\n".join(
            (
                "Individual quota reached. Resets in 46m20s.",
                f"{supervisor.LAUNCH_MARKER_PREFIX}{marker}",
                "process exited with code 1",
            )
        ),
        marker,
    )
    state = supervisor.RuntimeState(
        {
            spec.worker_id: supervisor.LaneState(
                profile_index=1,
                profile_backoff={"gemini-3.1-pro-high": old_reset},
                launch_marker=marker,
            )
        }
    )

    result = supervisor.reconcile(
        fake_host, state, fixed_now, Path("/repo")
    )

    lane = result.state.lanes[spec.worker_id]
    assert lane.profile_index == 1
    assert lane.profile_backoff == {"gemini-3.1-pro-high": old_reset}
    assert lane.ordinary_failures == 1
    assert lane.last_error == "ordinary-exit"


def test_healthy_observation_allows_later_identical_quota_event(fake_host, fixed_now):
    spec = supervisor.desired_workers()[0]
    quota_output = "Individual quota reached. Resets in 46m20s."
    fake_host.health[spec.session_name] = supervisor.SessionHealth(
        True, False, "bash", quota_output
    )
    first = supervisor.reconcile(
        fake_host,
        supervisor.RuntimeState.with_lane(spec.worker_id, profile_index=0),
        fixed_now,
        Path("/repo"),
    )
    fake_host.health[spec.session_name] = supervisor.SessionHealth(True, False, "agy", "")
    healthy = supervisor.reconcile(
        fake_host, first.state, fixed_now + timedelta(seconds=30), Path("/repo")
    )
    launch_marker = healthy.state.lanes[spec.worker_id].launch_marker
    fake_host.health[spec.session_name] = supervisor.SessionHealth(
        True,
        False,
        "bash",
        f"{supervisor.LAUNCH_MARKER_PREFIX}{launch_marker}\n{quota_output}",
        launch_marker,
    )
    result = supervisor.reconcile(
        fake_host, healthy.state, fixed_now + timedelta(seconds=60), Path("/repo")
    )
    lane = result.state.lanes[spec.worker_id]
    assert lane.profile_index == 2
    assert set(lane.profile_backoff) == {
        "gemini-3.1-pro-high",
        "gemini-3.6-flash-high",
    }


def test_dry_run_keeps_healthy_lane_failure_state_unchanged(fake_host, fixed_now):
    spec = supervisor.desired_workers()[10]
    fake_host.health[spec.session_name] = supervisor.SessionHealth(
        True, False, "codex", ""
    )
    initial = supervisor.RuntimeState(
        {
            spec.worker_id: supervisor.LaneState(
                ordinary_failures=2, next_retry_at=fixed_now + timedelta(minutes=1)
            )
        }
    )
    result = supervisor.reconcile(
        fake_host, initial, fixed_now, Path("/repo"), dry_run=True
    )
    assert result.state == initial


def test_cli_reconcile_dry_run_reports_without_host_or_state_mutation(
    capsys, fake_host, tmp_path
):
    state_dir = tmp_path / ".local/state/icml-worker-supervisor"
    request_path = state_dir / "smoke-request.json"
    request_path.parent.mkdir(parents=True)
    request = {"nonce": "0123456789abcdef", "stage": "interrupted"}
    request_path.write_text(json.dumps(request), encoding="utf-8")

    code = supervisor.main(
        ["reconcile", "--dry-run"],
        host=fake_host,
        home=tmp_path,
        now=FIXED_NOW,
    )

    output = capsys.readouterr().out.splitlines()
    assert code == 0
    assert len(output) == 15
    assert "proposed agy-paper-owner-01" in output
    assert "proposed codex-paper-owner-05" in output
    assert fake_host.created == []
    assert fake_host.sent == []
    assert fake_host.restored == []
    assert json.loads(request_path.read_text(encoding="utf-8")) == request
    assert not (state_dir / "runtime.json").exists()
    assert not (state_dir / "status.json").exists()


def test_status_prints_compact_worker_counts(capsys, fake_host, tmp_path):
    status_path = (
        tmp_path / ".local/state/icml-worker-supervisor" / "status.json"
    )
    status_path.parent.mkdir(parents=True)
    status_path.write_text(
        json.dumps(
            {
                "observed_at": FIXED_NOW.isoformat(),
                "state": "running",
                "supervisor_errors": [],
                "workers": [
                    {
                        "worker_id": spec.worker_id,
                        "agent": spec.agent,
                        "health": "healthy",
                    }
                    for spec in supervisor.desired_workers()
                ]
            }
        ),
        encoding="utf-8",
    )

    code = supervisor.main(
        ["status"], host=fake_host, home=tmp_path, now=FIXED_NOW
    )

    assert code == 0
    output = capsys.readouterr().out
    assert "agy 10/10" in output
    assert "codex 5/5" in output


def test_stop_requires_explicit_confirmation(fake_host, tmp_path):
    assert supervisor.main(["stop"], host=fake_host, home=tmp_path) == 2
    assert fake_host.stopped == []


def test_install_adopts_before_enabling_timer(fake_host, tmp_path):
    code = supervisor.main(
        ["install"], host=fake_host, home=tmp_path, now=FIXED_NOW
    )
    assert code == 0
    assert fake_host.events.index("reconcile") < fake_host.events.index(
        "systemctl --user enable --now icml-worker-supervisor.timer"
    )


def test_smoke_request_restores_only_disposable_session(fake_host, tmp_path):
    fake_host.production_pids = {
        spec.session_name: index
        for index, spec in enumerate(supervisor.desired_workers(), start=100)
    }
    code = supervisor.main(
        ["smoke-test", "--timeout", "1"],
        host=fake_host,
        home=tmp_path,
        now=FIXED_NOW,
    )
    assert code == 0
    assert fake_host.restored == ["icml-supervisor-smoke-test"]
    assert fake_host.stopped == ["icml-supervisor-smoke-test"]
    request = fake_host.smoke_requests_seen[0]
    assert set(request) == {"nonce", "stage"}
    assert request["stage"] == "interrupted"
    assert len(request["nonce"]) == 32
    int(request["nonce"], 16)
    assert fake_host.production_pids == {
        spec.session_name: index
        for index, spec in enumerate(supervisor.desired_workers(), start=100)
    }


def test_install_renders_exact_units_and_orders_systemd_commands(
    fake_host, tmp_path
):
    code = supervisor.main(
        ["install"], host=fake_host, home=tmp_path, now=FIXED_NOW
    )

    assert code == 0
    unit_dir = tmp_path / ".config/systemd/user"
    service = (
        unit_dir / "icml-worker-supervisor.service"
    ).read_text(encoding="utf-8")
    timer = (unit_dir / "icml-worker-supervisor.timer").read_text(
        encoding="utf-8"
    )
    repo_root = Path(supervisor.__file__).resolve().parents[1]
    python = Path(supervisor.sys.executable).resolve()
    assert "@REPO_ROOT@" not in service
    assert "@PYTHON@" not in service
    assert f"WorkingDirectory={repo_root}" in service
    assert (
        f"ExecStart={python} {repo_root}/ops/worker_supervisor.py reconcile"
        in service
    )
    assert "OnUnitActiveSec=30s" in timer
    assert "HF_TOKEN" not in service + timer
    assert "GH_TOKEN" not in service + timer
    assert fake_host.events.index("reconcile") < fake_host.events.index(
        "systemctl --user daemon-reload"
    )
    assert fake_host.events.index(
        "systemctl --user daemon-reload"
    ) < fake_host.events.index(
        "systemctl --user enable --now icml-worker-supervisor.timer"
    )
    assert fake_host.events.index(
        "systemctl --user enable --now icml-worker-supervisor.timer"
    ) < fake_host.events.index(
        "systemctl --user start icml-worker-supervisor.service"
    )


def test_reconcile_restores_strict_pending_smoke_request(fake_host, tmp_path):
    request_path = (
        tmp_path
        / ".local/state/icml-worker-supervisor"
        / "smoke-request.json"
    )
    request_path.parent.mkdir(parents=True)
    request_path.write_text(
        json.dumps({"nonce": "0123456789abcdef", "stage": "interrupted"}),
        encoding="utf-8",
    )

    code = supervisor.main(
        ["reconcile"], host=fake_host, home=tmp_path, now=FIXED_NOW
    )

    assert code == 0
    assert fake_host.restored == ["icml-supervisor-smoke-test"]
    assert json.loads(request_path.read_text(encoding="utf-8")) == {
        "nonce": "0123456789abcdef",
        "stage": "restored",
    }


def test_reconcile_quarantines_smoke_request_with_arbitrary_keys(
    fake_host, tmp_path
):
    marker = tmp_path / "smoke-request-command-was-used"
    request_path = (
        tmp_path
        / ".local/state/icml-worker-supervisor"
        / "smoke-request.json"
    )
    request_path.parent.mkdir(parents=True)
    request_path.write_text(
        json.dumps(
            {
                "nonce": "0123456789abcdef",
                "stage": "interrupted",
                "command": f"touch {marker}",
            }
        ),
        encoding="utf-8",
    )

    code = supervisor.main(
        ["reconcile"], host=fake_host, home=tmp_path, now=FIXED_NOW
    )

    assert code == 1
    assert fake_host.restored == []
    assert len(fake_host.sent) == 15
    assert not marker.exists()
    assert not request_path.exists()
    assert (
        request_path.with_name("smoke-request.quarantined.json").exists()
    )
    status = json.loads(
        (
            tmp_path
            / ".local/state/icml-worker-supervisor/status.json"
        ).read_text(encoding="utf-8")
    )
    assert status["state"] == "partial"
    assert status["supervisor_errors"] == ["smoke-request-invalid"]


def test_stale_smoke_request_is_quarantined_without_blocking_production(
    fake_host, tmp_path
):
    request_path = (
        tmp_path
        / ".local/state/icml-worker-supervisor"
        / "smoke-request.json"
    )
    request_path.parent.mkdir(parents=True)
    request_path.write_text(
        json.dumps({"nonce": "0123456789abcdef", "stage": "interrupted"}),
        encoding="utf-8",
    )
    stale_time = time.time() - supervisor.SMOKE_REQUEST_STALE_AFTER - 1
    os.utime(request_path, (stale_time, stale_time))

    code = supervisor.main(
        ["reconcile"], host=fake_host, home=tmp_path, now=FIXED_NOW
    )

    assert code == 1
    assert fake_host.restored == []
    assert len(fake_host.sent) == 15
    assert not request_path.exists()
    status = json.loads(
        (
            tmp_path
            / ".local/state/icml-worker-supervisor/status.json"
        ).read_text(encoding="utf-8")
    )
    assert status["state"] == "partial"
    assert status["supervisor_errors"] == ["smoke-request-stale"]


def test_smoke_restore_host_error_does_not_block_production_or_status(
    fake_host, tmp_path
):
    fake_host.restore_error = True
    request_path = (
        tmp_path
        / ".local/state/icml-worker-supervisor"
        / "smoke-request.json"
    )
    request_path.parent.mkdir(parents=True)
    request_path.write_text(
        json.dumps({"nonce": "0123456789abcdef", "stage": "interrupted"}),
        encoding="utf-8",
    )

    code = supervisor.main(
        ["reconcile"], host=fake_host, home=tmp_path, now=FIXED_NOW
    )

    assert code == 1
    assert len(fake_host.sent) == 15
    status = json.loads(
        (
            tmp_path
            / ".local/state/icml-worker-supervisor/status.json"
        ).read_text(encoding="utf-8")
    )
    assert status["state"] == "partial"
    assert status["supervisor_errors"] == ["smoke-recovery-failed"]


def test_lane_host_error_does_not_block_later_lanes_or_status(
    fake_host, tmp_path
):
    first, *later = supervisor.desired_workers()
    fake_host.health_errors.add(first.session_name)

    code = supervisor.main(
        ["reconcile"], host=fake_host, home=tmp_path, now=FIXED_NOW
    )

    assert code == 1
    assert [session for session, _command in fake_host.sent] == [
        spec.session_name for spec in later
    ]
    status = json.loads(
        (
            tmp_path
            / ".local/state/icml-worker-supervisor/status.json"
        ).read_text(encoding="utf-8")
    )
    assert status["observed_at"] == FIXED_NOW.isoformat()
    assert status["state"] == "partial"
    assert status["workers"][0]["health"] == "error"
    assert status["workers"][0]["last_error"] == "host-command-failure"
    assert status["workers"][-1]["health"] == "started"


def test_smoke_timeout_is_nonzero_and_cleans_only_disposable_session(
    fake_host, tmp_path
):
    fake_host.production_pids = {
        spec.session_name: index
        for index, spec in enumerate(supervisor.desired_workers(), start=100)
    }
    fake_host.restore_succeeds = False

    code = supervisor.main(
        ["smoke-test", "--timeout", "0"],
        host=fake_host,
        home=tmp_path,
        now=FIXED_NOW,
    )

    assert code == 1
    assert fake_host.stopped == ["icml-supervisor-smoke-test"]
    assert not (
        tmp_path
        / ".local/state/icml-worker-supervisor"
        / "smoke-request.json"
    ).exists()


@pytest.mark.parametrize("timeout", ("-1", "nan"))
def test_smoke_rejects_negative_or_nonfinite_timeout(
    fake_host, tmp_path, timeout
):
    code = supervisor.main(
        ["smoke-test", "--timeout", timeout],
        host=fake_host,
        home=tmp_path,
        now=FIXED_NOW,
    )

    assert code == 2
    assert fake_host.stopped == []


def test_smoke_detects_changed_production_pid_and_still_cleans_up(
    fake_host, tmp_path
):
    fake_host.production_pids = {
        spec.session_name: index
        for index, spec in enumerate(supervisor.desired_workers(), start=100)
    }
    fake_host.mutate_production_pid_on_restore = True

    code = supervisor.main(
        ["smoke-test", "--timeout", "1"],
        host=fake_host,
        home=tmp_path,
        now=FIXED_NOW,
    )

    assert code == 1
    assert fake_host.stopped == ["icml-supervisor-smoke-test"]


def test_confirmed_stop_targets_only_desired_worker_sessions(fake_host, tmp_path):
    fake_host.health["unrelated-session"] = supervisor.SessionHealth(
        True, False, "bash", ""
    )

    code = supervisor.main(
        ["stop", "--confirm"],
        host=fake_host,
        home=tmp_path,
        now=FIXED_NOW,
    )

    assert code == 0
    assert fake_host.events[0] == (
        "systemctl --user disable --now icml-worker-supervisor.timer"
    )
    assert fake_host.stopped == [
        spec.session_name for spec in supervisor.desired_workers()
    ]
    assert "unrelated-session" not in fake_host.stopped
    status_path = (
        tmp_path / ".local/state/icml-worker-supervisor/status.json"
    )
    status = json.loads(status_path.read_text(encoding="utf-8"))
    assert status["observed_at"] == FIXED_NOW.isoformat()
    assert status["state"] == "stopped"
    assert {
        worker["health"] for worker in status["workers"]
    } == {"stopped"}

    status_code = supervisor.main(
        ["status"],
        host=fake_host,
        home=tmp_path,
        now=FIXED_NOW + timedelta(hours=1),
    )
    assert status_code == 0


def test_stop_quiesces_service_and_waits_for_inflight_reconcile(
    fake_host, tmp_path
):
    lock_path = (
        tmp_path
        / ".local/state/icml-worker-supervisor"
        / "supervisor.lock"
    )
    fake_host.stop_lock_path = lock_path
    result = []

    with supervisor.exclusive_lock(lock_path):
        stop_thread = threading.Thread(
            target=lambda: result.append(
                supervisor.main(
                    ["stop", "--confirm"], host=fake_host, home=tmp_path
                )
            )
        )
        stop_thread.start()
        service_was_quiesced = fake_host.service_quiesced.wait(timeout=5)
        stopped_while_reconcile_held_lock = list(fake_host.stopped)
        fake_host.inflight_reconcile = True
        fake_host.ensure_session(
            supervisor.desired_workers()[0], "exec agy"
        )
        fake_host.inflight_reconcile = False

    stop_thread.join(timeout=5)

    assert service_was_quiesced
    assert stopped_while_reconcile_held_lock == []
    assert not stop_thread.is_alive()
    assert result == [0]
    assert fake_host.events[:2] == [
        "systemctl --user disable --now icml-worker-supervisor.timer",
        "systemctl --user stop icml-worker-supervisor.service",
    ]
    first_stop = fake_host.events.index("stop agy-paper-owner-01")
    assert fake_host.events.index("recreate agy-paper-owner-01") < first_stop
    assert fake_host.events[first_stop - 1] == "lock-held"
    assert all(
        not event.startswith("recreate ")
        for event in fake_host.events[first_stop:]
    )


def test_status_prints_one_line_for_each_degraded_lane(
    capsys, fake_host, tmp_path
):
    status_path = (
        tmp_path / ".local/state/icml-worker-supervisor" / "status.json"
    )
    status_path.parent.mkdir(parents=True)
    status_path.write_text(
        json.dumps(
            {
                "observed_at": FIXED_NOW.isoformat(),
                "state": "running",
                "supervisor_errors": [],
                "workers": [
                    {
                        "worker_id": "agy-paper-owner-01",
                        "agent": "agy",
                        "health": "backed_off",
                        "last_error": "quota-reached",
                    },
                    {
                        "worker_id": "agy-paper-owner-02",
                        "agent": "agy",
                        "health": "healthy",
                        "last_error": "",
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    code = supervisor.main(
        ["status"], host=fake_host, home=tmp_path, now=FIXED_NOW
    )

    assert code == 0
    output = capsys.readouterr().out
    assert "agy 1/10" in output
    assert (
        "agy-paper-owner-01 backed_off quota-reached" in output
    )
    assert "agy-paper-owner-02 healthy" not in output


def test_cli_reconcile_loads_and_rewrites_runtime_state(fake_host, tmp_path):
    spec = supervisor.desired_workers()[0]
    for other_spec in supervisor.desired_workers()[1:]:
        fake_host.health[other_spec.session_name] = supervisor.SessionHealth(
            True, False, other_spec.agent, ""
        )
    fake_host.health[spec.session_name] = supervisor.SessionHealth(
        True,
        False,
        "bash",
        "Individual quota reached. Resets in 46m20s.",
    )

    first_code = supervisor.main(
        ["reconcile"], host=fake_host, home=tmp_path, now=FIXED_NOW
    )
    fake_host.health[spec.session_name] = supervisor.SessionHealth(
        True, False, "agy", ""
    )
    second_code = supervisor.main(
        ["reconcile"],
        host=fake_host,
        home=tmp_path,
        now=FIXED_NOW + timedelta(seconds=30),
    )

    runtime_path = (
        tmp_path / ".local/state/icml-worker-supervisor" / "runtime.json"
    )
    runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
    assert first_code == 0
    assert second_code == 0
    assert runtime["lanes"][spec.worker_id]["profile_index"] == 1
    assert runtime["lanes"][spec.worker_id]["profile_backoff"] == {
        "gemini-3.1-pro-high": (
            FIXED_NOW + timedelta(minutes=46, seconds=20)
        ).isoformat()
    }


def test_cli_reconcile_returns_zero_when_provider_is_backed_off(
    fake_host, tmp_path
):
    spec = supervisor.desired_workers()[0]
    for other_spec in supervisor.desired_workers()[1:]:
        fake_host.health[other_spec.session_name] = supervisor.SessionHealth(
            True, False, other_spec.agent, ""
        )
    runtime_path = (
        tmp_path / ".local/state/icml-worker-supervisor" / "runtime.json"
    )
    runtime_path.parent.mkdir(parents=True)
    runtime_path.write_text(
        json.dumps(
            {
                "lanes": {
                    spec.worker_id: {
                        "profile_index": 0,
                        "profile_backoff": {
                            profile.name: (
                                FIXED_NOW + timedelta(hours=index + 1)
                            ).isoformat()
                            for index, profile in enumerate(
                                supervisor.agy_profiles()
                            )
                        },
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    code = supervisor.main(
        ["reconcile"], host=fake_host, home=tmp_path, now=FIXED_NOW
    )

    status_path = (
        tmp_path / ".local/state/icml-worker-supervisor" / "status.json"
    )
    status = json.loads(status_path.read_text(encoding="utf-8"))
    assert code == 0
    assert status["workers"][0]["health"] == "backed_off"


def test_reconcile_status_has_utc_time_and_bounded_safe_lane_events(
    fake_host, fixed_now
):
    for spec in supervisor.desired_workers():
        fake_host.health[spec.session_name] = supervisor.SessionHealth(
            True,
            False,
            spec.agent,
            "HF_TOKEN=hf_secret Use the shared paper-owner prompt",
        )
    state = supervisor.RuntimeState.empty()
    result = None
    for index in range(supervisor.EVENT_HISTORY_LIMIT + 5):
        result = supervisor.reconcile(
            fake_host,
            state,
            fixed_now + timedelta(seconds=30 * index),
            Path("/repo"),
        )
        state = result.state

    assert result is not None
    assert result.status["observed_at"] == (
        fixed_now
        + timedelta(seconds=30 * (supervisor.EVENT_HISTORY_LIMIT + 4))
    ).isoformat()
    events = result.status["workers"][0]["events"]
    assert len(events) == supervisor.EVENT_HISTORY_LIMIT
    assert all(
        set(event) == {"observed_at", "health", "error_class"}
        for event in events
    )
    assert all(event["health"] in supervisor.HEALTH_CLASSES for event in events)
    assert all(
        event["error_class"] in supervisor.ERROR_CLASSES
        for event in events
    )
    encoded = json.dumps(result.status)
    assert "HF_TOKEN" not in encoded
    assert "hf_secret" not in encoded
    assert "Use the shared" not in encoded


def test_legacy_runtime_error_text_is_normalized_before_rewrite():
    spec = supervisor.desired_workers()[0]
    state = supervisor._runtime_from_json(
        {
            "lanes": {
                spec.worker_id: {
                    "last_error": (
                        "HF_TOKEN=hf_secret "
                        "Use the shared paper-owner prompt"
                    )
                }
            }
        }
    )

    encoded = json.dumps(supervisor._runtime_to_json(state))

    assert "HF_TOKEN" not in encoded
    assert "hf_secret" not in encoded
    assert "Use the shared" not in encoded
    assert (
        supervisor._runtime_to_json(state)["lanes"][spec.worker_id][
            "last_error"
        ]
        == "unhealthy-session"
    )


def test_status_fails_closed_when_green_snapshot_is_stale(
    capsys, fake_host, tmp_path
):
    status_path = (
        tmp_path / ".local/state/icml-worker-supervisor/status.json"
    )
    status_path.parent.mkdir(parents=True)
    observed_at = FIXED_NOW - supervisor.STATUS_STALE_AFTER - timedelta(seconds=1)
    status_path.write_text(
        json.dumps(
            {
                "observed_at": observed_at.isoformat(),
                "state": "running",
                "supervisor_errors": [],
                "workers": [
                    {
                        "worker_id": spec.worker_id,
                        "agent": spec.agent,
                        "health": "healthy",
                        "last_error": "",
                    }
                    for spec in supervisor.desired_workers()
                ],
            }
        ),
        encoding="utf-8",
    )

    code = supervisor.main(
        ["status"], host=fake_host, home=tmp_path, now=FIXED_NOW
    )

    captured = capsys.readouterr()
    assert code == 1
    assert "status snapshot is stale" in captured.err
    assert "agy 0/10" in captured.out
    assert "codex 0/5" in captured.out
    assert "agy 10/10" not in captured.out


def test_status_before_install_reports_zero_counts_and_absence(
    capsys, fake_host, tmp_path
):
    code = supervisor.main(
        ["status"], host=fake_host, home=tmp_path, now=FIXED_NOW
    )

    captured = capsys.readouterr()
    assert code == 1
    assert "status snapshot is unavailable" in captured.err
    assert "agy 0/10" in captured.out
    assert "codex 0/5" in captured.out


def test_host_adapter_smoke_operations_use_fixed_argument_arrays(monkeypatch):
    calls = []

    def run(argv, *, check, text, capture_output):
        assert isinstance(argv, list)
        assert check is False
        assert text is True
        assert capture_output is True
        calls.append(argv)
        if argv[-1] == "#{pane_pid}":
            return subprocess.CompletedProcess(argv, 0, "321\n", "")
        if argv[-1] == "#{pane_current_command}":
            return subprocess.CompletedProcess(argv, 0, "sleep\n", "")
        return subprocess.CompletedProcess(argv, 0, "", "")

    monkeypatch.setattr(supervisor.subprocess, "run", run)
    host = supervisor.HostAdapter(Path("/repo"))

    assert host.pane_pid("agy-paper-owner-01") == 321
    host.create_disposable_session(
        "icml-supervisor-smoke-test", "/usr/bin/sleep 300"
    )
    host.interrupt_session("icml-supervisor-smoke-test")
    host.restore_disposable_session(
        "icml-supervisor-smoke-test", "exec /usr/bin/sleep 300"
    )
    assert (
        host.session_foreground_command("icml-supervisor-smoke-test")
        == "sleep"
    )
    host.stop_session("icml-supervisor-smoke-test")
    host.systemctl_user(
        "enable", "--now", "icml-worker-supervisor.timer"
    )

    assert calls == [
        [
            "tmux",
            "list-panes",
            "-t",
            "agy-paper-owner-01",
            "-F",
            "#{pane_pid}",
        ],
        [
            "tmux",
            "new-session",
            "-d",
            "-s",
            "icml-supervisor-smoke-test",
            "-c",
            "/repo",
            "exec /usr/bin/sleep 300",
        ],
        [
            "tmux",
            "set-option",
            "-w",
            "-t",
            "icml-supervisor-smoke-test",
            "remain-on-exit",
            "on",
        ],
        [
            "tmux",
            "send-keys",
            "-t",
            "icml-supervisor-smoke-test",
            "C-c",
        ],
        [
            "tmux",
            "respawn-pane",
            "-k",
            "-t",
            "icml-supervisor-smoke-test",
            "exec /usr/bin/sleep 300",
        ],
        [
            "tmux",
            "list-panes",
            "-t",
            "icml-supervisor-smoke-test",
            "-F",
            "#{pane_current_command}",
        ],
        [
            "tmux",
            "kill-session",
            "-t",
            "icml-supervisor-smoke-test",
        ],
        [
            "systemctl",
            "--user",
            "enable",
            "--now",
            "icml-worker-supervisor.timer",
        ],
    ]


def test_disposable_target_survives_interrupt_and_accepts_respawn(monkeypatch):
    session_name = "icml-supervisor-smoke-test"
    state = {
        "exists": False,
        "foreground": "",
        "remain_on_exit": False,
    }

    def run(argv, *, check, text, capture_output):
        assert check is False
        assert text is True
        assert capture_output is True
        if argv[:4] == ["tmux", "new-session", "-d", "-s"]:
            assert argv == [
                "tmux",
                "new-session",
                "-d",
                "-s",
                session_name,
                "-c",
                "/repo",
                "exec /usr/bin/sleep 300",
            ]
            state["exists"] = True
            state["foreground"] = "sleep"
            return subprocess.CompletedProcess(argv, 0, "", "")
        if argv[:3] == ["tmux", "set-option", "-w"]:
            assert argv == [
                "tmux",
                "set-option",
                "-w",
                "-t",
                session_name,
                "remain-on-exit",
                "on",
            ]
            state["remain_on_exit"] = True
            return subprocess.CompletedProcess(argv, 0, "", "")
        if argv[:3] == ["tmux", "send-keys", "-t"]:
            if not state["exists"]:
                return subprocess.CompletedProcess(
                    argv, 1, "", f"can't find session: {session_name}"
                )
            if state["remain_on_exit"]:
                state["foreground"] = ""
            else:
                state["exists"] = False
            return subprocess.CompletedProcess(argv, 0, "", "")
        if argv[:3] == ["tmux", "respawn-pane", "-k"]:
            if not state["exists"]:
                return subprocess.CompletedProcess(
                    argv, 1, "", f"can't find pane: {session_name}"
                )
            state["foreground"] = "sleep"
            return subprocess.CompletedProcess(argv, 0, "", "")
        if argv[-1] == "#{pane_current_command}":
            if not state["exists"]:
                return subprocess.CompletedProcess(
                    argv, 1, "", f"can't find session: {session_name}"
                )
            return subprocess.CompletedProcess(
                argv, 0, f"{state['foreground']}\n", ""
            )
        raise AssertionError(f"unexpected argv: {argv}")

    monkeypatch.setattr(supervisor.subprocess, "run", run)
    host = supervisor.HostAdapter(Path("/repo"))

    host.create_disposable_session(session_name, "/usr/bin/sleep 300")
    host.interrupt_session(session_name)
    host.restore_disposable_session(
        session_name, "exec /usr/bin/sleep 300"
    )

    assert host.session_foreground_command(session_name) == "sleep"


def test_disposable_creation_cleans_up_if_remain_on_exit_cannot_be_set(
    monkeypatch
):
    session_name = "icml-supervisor-smoke-test"
    calls = []

    def run(argv, *, check, text, capture_output):
        calls.append(argv)
        if argv[:3] == ["tmux", "set-option", "-w"]:
            return subprocess.CompletedProcess(
                argv, 1, "", "invalid option: remain-on-exit"
            )
        return subprocess.CompletedProcess(argv, 0, "", "")

    monkeypatch.setattr(supervisor.subprocess, "run", run)

    with pytest.raises(supervisor.HostCommandError):
        supervisor.HostAdapter(Path("/repo")).create_disposable_session(
            session_name, "/usr/bin/sleep 300"
        )

    assert calls[-1] == [
        "tmux",
        "kill-session",
        "-t",
        session_name,
    ]
