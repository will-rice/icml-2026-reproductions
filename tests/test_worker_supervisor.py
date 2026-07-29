import json
from datetime import datetime, timedelta, timezone
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

    def mark_started(self, worker_ids):
        by_worker_id = {spec.worker_id: spec for spec in supervisor.desired_workers()}
        for worker_id in worker_ids:
            spec = by_worker_id[worker_id]
            self.health[spec.session_name] = supervisor.SessionHealth(
                True, False, spec.agent, ""
            )


@pytest.fixture
def fake_host():
    return FakeHost()


@pytest.fixture
def fixed_now():
    return datetime(2026, 7, 29, 12, 0, tzinfo=timezone.utc)


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
    fake_host.health[spec.session_name] = supervisor.SessionHealth(
        True, False, "bash", quota_output
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
