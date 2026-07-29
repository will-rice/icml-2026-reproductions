from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shlex
import secrets
import subprocess
import sys
import tempfile
import time
from typing import Any, Iterator, Literal

AgentName = Literal["agy", "codex"]
HUGGING_FACE_TOKEN_PATTERN = re.compile(r"\bhf_[A-Za-z0-9]{6,}\b")
GITHUB_TOKEN_PATTERN = re.compile(
    r"\b(?:gh[pousr]_[A-Za-z0-9]+|github_pat_[A-Za-z0-9_]+)\b"
)
BEARER_TOKEN_PATTERN = re.compile(r"(Authorization:\s*Bearer\s+)\S+", re.IGNORECASE)
SECRET_ENV_ASSIGNMENT_PATTERN = re.compile(
    r"\b(?:HF_TOKEN|HUGGING_FACE_HUB_TOKEN|GH_TOKEN)\s*=\s*\S+", re.IGNORECASE
)
QUOTA_RE = re.compile(
    r"quota reached.*?Resets in "
    r"(?:(?P<hours>\d+)h)?(?:(?P<minutes>\d+)m)?(?:(?P<seconds>\d+)s)?",
    re.IGNORECASE | re.DOTALL,
)
EXIT_RE = re.compile(r"process exited with code \d+", re.IGNORECASE)
HEALTH_CLASSES = frozenset(
    {"healthy", "started", "backed_off", "proposed", "error", "stopped"}
)
ERROR_CLASSES = frozenset(
    {
        "",
        "quota-reached",
        "ordinary-exit",
        "unhealthy-session",
        "host-command-failure",
    }
)
FAILURE_CLASSES = ERROR_CLASSES - {""}
SUPERVISOR_ERROR_CLASSES = frozenset(
    {
        "smoke-request-invalid",
        "smoke-request-stale",
        "smoke-recovery-failed",
    }
)
EVENT_HISTORY_LIMIT = 20
STATUS_STALE_AFTER = timedelta(seconds=90)
SMOKE_REQUEST_STALE_AFTER = 120.0
TMUX_MISSING_TARGET_PREFIXES = (
    "can't find session:",
    "can't find window:",
    "no server running on ",
)
SMOKE_SESSION = "icml-supervisor-smoke-test"
SMOKE_COMMAND = "/usr/bin/sleep 300"
SMOKE_RESTORE_COMMAND = "exec /usr/bin/sleep 300"
LAUNCH_MARKER_PREFIX = "__ICML_WORKER_LAUNCH__:"
LAUNCH_MARKER_RE = re.compile(
    re.escape(LAUNCH_MARKER_PREFIX) + r"(?P<marker>[A-Za-z0-9_-]{1,128})"
)
TMUX_FALLBACK_SHELL = "/bin/bash"
PROMPT = (
    "Use the shared icml-repro-loop skill directly and keep running its "
    "paper-owner loop. Read and follow "
    "/home/will/.agents/skills/icml-repro-loop/SKILL.md. "
    "Persistent worker ID: {worker_id}."
)
AGY_IDLE_PROMPT_RE = re.compile(
    r"Do you trust the contents of this project\?"
    r"|Accept-edits mode: file edits auto-approved[\s\S]*\? for shortcuts",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class WorkerSpec:
    worker_id: str
    agent: AgentName
    session_name: str


@dataclass(frozen=True)
class ModelProfile:
    name: str
    argv: tuple[str, ...]


@dataclass(frozen=True)
class SessionHealth:
    exists: bool
    pane_dead: bool
    foreground_command: str
    recent_output: str
    launch_marker: str = ""


@dataclass(frozen=True)
class LaneEvent:
    observed_at: datetime
    health: str
    error_class: str


@dataclass(frozen=True)
class LaneState:
    profile_index: int = 0
    profile_backoff: dict[str, datetime] = field(default_factory=dict)
    restart_count: int = 0
    ordinary_failures: int = 0
    next_retry_at: datetime | None = None
    last_error: str = ""
    processed_failure_digest: str = ""
    launch_marker: str = ""
    events: tuple[LaneEvent, ...] = ()


@dataclass(frozen=True)
class RuntimeState:
    lanes: dict[str, LaneState] = field(default_factory=dict)

    @classmethod
    def empty(cls) -> RuntimeState:
        return cls()

    @classmethod
    def with_lane(
        cls,
        worker_id: str,
        *,
        profile_index: int = 0,
        profile_backoff: dict[str, datetime] | None = None,
    ) -> RuntimeState:
        return cls(
            {
                worker_id: LaneState(
                    profile_index=profile_index,
                    profile_backoff=dict(profile_backoff or {}),
                )
            }
        )


@dataclass(frozen=True)
class ReconcileResult:
    state: RuntimeState
    status: dict[str, Any]
    started: tuple[str, ...]
    proposed: tuple[str, ...]
    failures: tuple[str, ...] = ()


class AlreadyRunning(RuntimeError):
    """Raised when a concurrent supervisor already holds the runtime lock."""


class HostCommandError(RuntimeError):
    """Raised when a host command cannot complete an operation."""

    def __init__(self, stderr: str):
        super().__init__(f"host command failed: {sanitize_text(stderr).strip()}")


class SmokeRequestStale(ValueError):
    """Raised when an abandoned smoke request is too old to execute."""


class SmokeRecoveryError(RuntimeError):
    """Raised when a valid smoke request cannot restore its disposable lane."""


class HostAdapter:
    def __init__(self, repo_root: Path):
        self.repo_root = repo_root

    def _run_tmux(self, argv: list[str]) -> subprocess.CompletedProcess[str]:
        return subprocess.run(argv, check=False, text=True, capture_output=True)

    def _checked_command(self, argv: list[str]) -> None:
        result = subprocess.run(argv, check=False, text=True, capture_output=True)
        if result.returncode:
            raise HostCommandError(result.stderr)

    def _has_expected_child(self, spec: WorkerSpec, pane_pid: int) -> bool:
        children_path = Path(
            f"/proc/{pane_pid}/task/{pane_pid}/children"
        )
        try:
            child_pids = children_path.read_text(encoding="utf-8").split()
        except OSError:
            return False
        for child_pid in child_pids:
            try:
                command = Path(f"/proc/{child_pid}/comm").read_text(
                    encoding="utf-8"
                )
            except OSError:
                continue
            if command.strip() == spec.agent:
                return True
        return False

    def session_health(self, spec: WorkerSpec) -> SessionHealth:
        pane_argv = [
            "tmux",
            "list-panes",
            "-t",
            spec.session_name,
            "-F",
            (
                "#{pane_dead}\t#{pane_current_command}\t#{pane_pid}"
                "\t#{pane_start_command}"
            ),
        ]
        pane_result = self._run_tmux(pane_argv)
        if pane_result.returncode:
            if pane_result.stderr.strip().lower().startswith(
                TMUX_MISSING_TARGET_PREFIXES
            ):
                return SessionHealth(False, False, "", "")
            raise HostCommandError(pane_result.stderr)

        try:
            (
                pane_dead,
                foreground_command,
                pane_pid_text,
                pane_start_command,
            ) = pane_result.stdout.splitlines()[0].split("\t", maxsplit=3)
            pane_pid = int(pane_pid_text)
        except (IndexError, ValueError) as error:
            raise HostCommandError("tmux returned malformed pane metadata") from error

        launch_marker = _launch_marker_from_text(pane_start_command)
        if launch_marker and self._has_expected_child(spec, pane_pid):
            foreground_command = spec.agent

        output_argv = [
            "tmux",
            "capture-pane",
            "-pt",
            spec.session_name,
            "-S",
            "-80",
        ]
        output_result = self._run_tmux(output_argv)
        if output_result.returncode:
            raise HostCommandError(output_result.stderr)
        return SessionHealth(
            True,
            pane_dead == "1",
            foreground_command,
            sanitize_text(output_result.stdout),
            launch_marker,
        )

    def ensure_session(self, spec: WorkerSpec, command: str) -> bool:
        health = self.session_health(spec)
        if is_healthy(spec, health):
            return False

        persistent_command = f"{command}; exec {TMUX_FALLBACK_SHELL}"
        if not health.exists:
            mutation_argv = [
                "tmux",
                "new-session",
                "-d",
                "-s",
                spec.session_name,
                "-c",
                str(self.repo_root),
                persistent_command,
            ]
        else:
            mutation_argv = [
                "tmux",
                "respawn-pane",
                "-k",
                "-t",
                spec.session_name,
                "-c",
                str(self.repo_root),
                persistent_command,
            ]
        mutation_result = self._run_tmux(mutation_argv)
        if mutation_result.returncode:
            raise HostCommandError(mutation_result.stderr)
        return True

    def stop_session(self, session_name: str) -> None:
        result = self._run_tmux(["tmux", "kill-session", "-t", session_name])
        stderr = result.stderr.strip().lower()
        if result.returncode and not stderr.startswith(
            TMUX_MISSING_TARGET_PREFIXES
        ):
            raise HostCommandError(result.stderr)

    def systemctl_user(self, *arguments: str) -> None:
        self._checked_command(["systemctl", "--user", *arguments])

    def pane_pid(self, session_name: str) -> int:
        result = self._run_tmux(
            ["tmux", "list-panes", "-t", session_name, "-F", "#{pane_pid}"]
        )
        if result.returncode:
            raise HostCommandError(result.stderr)
        try:
            return int(result.stdout.splitlines()[0])
        except (IndexError, ValueError) as error:
            raise HostCommandError("tmux returned malformed pane PID") from error

    def create_disposable_session(self, session_name: str, command: str) -> None:
        self._checked_command(
            [
                "tmux",
                "new-session",
                "-d",
                "-s",
                session_name,
                "-c",
                str(self.repo_root),
                f"exec {command}",
            ]
        )
        try:
            self._checked_command(
                [
                    "tmux",
                    "set-option",
                    "-w",
                    "-t",
                    session_name,
                    "remain-on-exit",
                    "on",
                ]
            )
        except HostCommandError:
            self.stop_session(session_name)
            raise

    def interrupt_session(self, session_name: str) -> None:
        self._checked_command(["tmux", "send-keys", "-t", session_name, "C-c"])

    def restore_disposable_session(
        self, session_name: str, command: str
    ) -> None:
        self._checked_command(
            ["tmux", "respawn-pane", "-k", "-t", session_name, command]
        )

    def session_foreground_command(self, session_name: str) -> str:
        result = self._run_tmux(
            [
                "tmux",
                "list-panes",
                "-t",
                session_name,
                "-F",
                "#{pane_current_command}",
            ]
        )
        if result.returncode:
            raise HostCommandError(result.stderr)
        try:
            return result.stdout.splitlines()[0]
        except IndexError as error:
            raise HostCommandError(
                "tmux returned malformed foreground command"
            ) from error

    def wait_for_smoke_restore(
        self, request_path: Path, nonce: str, timeout: float
    ) -> bool:
        deadline = time.monotonic() + timeout
        while True:
            request = _load_smoke_request(request_path)
            if request is None or request["nonce"] != nonce:
                raise ValueError("smoke request ownership changed")
            if request["stage"] == "restored":
                return True
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return False
            time.sleep(min(0.05, remaining))


def desired_workers() -> tuple[WorkerSpec, ...]:
    agy = tuple(
        WorkerSpec(f"agy-paper-owner-{i:02d}", "agy", f"agy-paper-owner-{i:02d}")
        for i in range(1, 11)
    )
    codex = tuple(
        WorkerSpec(
            f"codex-paper-owner-{i:02d}", "codex", f"codex-paper-owner-{i:02d}"
        )
        for i in range(1, 6)
    )
    return agy + codex


def is_healthy(spec: WorkerSpec, health: SessionHealth) -> bool:
    return (
        health.exists
        and not health.pane_dead
        and health.foreground_command == spec.agent
        and not (
            spec.agent == "agy"
            and AGY_IDLE_PROMPT_RE.search(health.recent_output)
        )
    )


def runtime_directory() -> Path:
    state_home = Path(
        os.environ.get("XDG_STATE_HOME", str(Path.home() / ".local" / "state"))
    )
    path = state_home / "icml-worker-supervisor"
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    return path


@contextmanager
def exclusive_lock(path: Path, *, wait: bool = False) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    lock_file = path.open("a+")
    try:
        try:
            operation = fcntl.LOCK_EX
            if not wait:
                operation |= fcntl.LOCK_NB
            fcntl.flock(lock_file.fileno(), operation)
        except BlockingIOError as error:
            raise AlreadyRunning("worker supervisor is already running") from error
        yield
    finally:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
        lock_file.close()


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            json.dump(value, temporary, sort_keys=True)
            temporary.write("\n")
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def atomic_write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            temporary.write(value)
            temporary.flush()
            os.fsync(temporary.fileno())
        os.chmod(temporary_path, 0o644)
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def agy_profiles() -> tuple[ModelProfile, ...]:
    return (
        ModelProfile(
            "gemini-3.1-pro-high",
            (
                "agy",
                "--dangerously-skip-permissions",
                "--effort",
                "high",
                "--model",
                "gemini-3.1-pro-high",
                "--mode",
                "accept-edits",
                "--print-timeout",
                "24h",
                "--output-format",
                "stream-json",
            ),
        ),
        ModelProfile(
            "gemini-3.6-flash-high",
            (
                "agy",
                "--dangerously-skip-permissions",
                "--effort",
                "high",
                "--model",
                "gemini-3.6-flash-high",
                "--mode",
                "accept-edits",
                "--print-timeout",
                "24h",
                "--output-format",
                "stream-json",
            ),
        ),
        ModelProfile(
            "claude-sonnet-4-6",
            (
                "agy",
                "--dangerously-skip-permissions",
                "--model",
                "claude-sonnet-4-6",
                "--mode",
                "accept-edits",
                "--print-timeout",
                "24h",
                "--output-format",
                "stream-json",
            ),
        ),
    )


def codex_profile() -> ModelProfile:
    return ModelProfile(
        "gpt-5.5-high",
        (
            "codex",
            "exec",
            "--ignore-user-config",
            "--ephemeral",
            "--json",
            "--sandbox",
            "workspace-write",
            "-c",
            "sandbox_workspace_write.network_access=true",
            "-c",
            'model_reasoning_effort="high"',
            "-m",
            "gpt-5.5",
        ),
    )


def launch_shell_command(
    spec: WorkerSpec, profile: ModelProfile, repo_root: Path
) -> str:
    prompt = PROMPT.format(worker_id=spec.worker_id)
    user_cli_path = (
        'export PATH="$HOME/.local/bin:$HOME/.cargo/bin:'
        '/usr/local/bin:/usr/bin:/bin:$PATH";'
    )
    credentials = (
        'exec /usr/bin/env HF_TOKEN="$(hf auth token)" '
        'GH_TOKEN="$(gh auth token)"'
    )
    if spec.agent == "agy":
        return " ".join(
            (
                user_cli_path,
                credentials,
                'HF_HOME="/tmp/icml-agy-hf-XX"',
                'UV_CACHE_DIR="/tmp/icml-repro-uv-cache"',
                shlex.join((*profile.argv, "--print", prompt)),
            )
        )
    return " ".join(
        (
            user_cli_path,
            credentials,
            shlex.join((*profile.argv, "-C", str(repo_root), prompt)),
        )
    )


def _profile_for(spec: WorkerSpec, lane: LaneState) -> ModelProfile:
    if spec.agent == "codex":
        return codex_profile()
    profiles = agy_profiles()
    return profiles[lane.profile_index % len(profiles)]


def parse_quota_reset(message: str, now: datetime) -> datetime | None:
    match = QUOTA_RE.search(message)
    if match is None:
        return None
    parts = {name: int(value or 0) for name, value in match.groupdict().items()}
    return now + timedelta(**parts)


def next_profile(
    spec: WorkerSpec, lane: LaneState, now: datetime, *, rotate: bool = False
) -> tuple[ModelProfile | None, int | None]:
    if spec.agent == "codex":
        return codex_profile(), 0
    active_profile = _profile_for(spec, lane)
    active_reset = lane.profile_backoff.get(active_profile.name)
    if not rotate and (active_reset is None or active_reset <= now):
        return active_profile, lane.profile_index
    for index, profile in enumerate(agy_profiles()):
        reset_at = lane.profile_backoff.get(profile.name)
        if reset_at is None or reset_at <= now:
            return profile, index
    return None, None


def _earliest_profile_reset(lane: LaneState) -> datetime | None:
    resets = tuple(lane.profile_backoff.values())
    return min(resets) if resets else None


def _ordinary_retry_at(lane: LaneState, worker_id: str, now: datetime) -> datetime:
    delay = min(15 * 2**lane.ordinary_failures, 900)
    jitter = int.from_bytes(hashlib.sha256(worker_id.encode()).digest()[:2], "big") % 11
    return now + timedelta(seconds=delay + jitter)


def _quota_event_digest(match: re.Match[str]) -> str:
    record = " ".join(match.group(0).split())
    return hashlib.sha256(record.encode()).hexdigest()


def _marked_launch_command(command: str, marker: str) -> str:
    marker_line = f"{LAUNCH_MARKER_PREFIX}{marker}"
    return f"printf '%s\\n' {shlex.quote(marker_line)}; {command}"


def _launch_marker_from_text(text: str) -> str:
    match = LAUNCH_MARKER_RE.search(text)
    return "" if match is None else match.group("marker")


def _current_launch_output(lane: LaneState, health: SessionHealth) -> str:
    output = health.recent_output
    if not lane.launch_marker:
        return output
    if health.launch_marker != lane.launch_marker:
        return ""
    marker_line = f"{LAUNCH_MARKER_PREFIX}{lane.launch_marker}"
    _before, found, current = output.rpartition(marker_line)
    return current if found else output


def _classified_error(output: str) -> str:
    if not output:
        return ""
    if output in FAILURE_CLASSES:
        return output
    if QUOTA_RE.search(output):
        return "quota-reached"
    if EXIT_RE.search(output):
        return "ordinary-exit"
    return "unhealthy-session"


def _iso_timestamp(value: datetime | None) -> str | None:
    if value is None:
        return None
    return value.astimezone(timezone.utc).isoformat()


def _append_lane_event(
    lane: LaneState,
    now: datetime,
    health: str,
    error_class: str,
) -> LaneState:
    if health not in HEALTH_CLASSES:
        raise ValueError("invalid lane health class")
    if error_class not in ERROR_CLASSES:
        raise ValueError("invalid lane error class")
    events = (
        *lane.events,
        LaneEvent(
            observed_at=now.astimezone(timezone.utc),
            health=health,
            error_class=error_class,
        ),
    )
    return replace(lane, events=events[-EVENT_HISTORY_LIMIT:])


def _status_entry(
    spec: WorkerSpec, health: str, profile: ModelProfile, lane: LaneState
) -> dict[str, Any]:
    return {
        "worker_id": spec.worker_id,
        "agent": spec.agent,
        "health": health,
        "model": profile.name,
        "restart_count": lane.restart_count,
        "next_retry_at": _iso_timestamp(lane.next_retry_at),
        "last_error": _classified_error(lane.last_error),
        "events": [
            {
                "observed_at": _iso_timestamp(event.observed_at),
                "health": event.health,
                "error_class": event.error_class,
            }
            for event in lane.events
        ],
    }


def reconcile(
    host: HostAdapter,
    state: RuntimeState,
    now: datetime,
    repo_root: Path,
    dry_run: bool = False,
) -> ReconcileResult:
    lanes = dict(state.lanes)
    started: list[str] = []
    proposed: list[str] = []
    failures: list[str] = []
    workers: list[dict[str, Any]] = []
    for spec in desired_workers():
        lane = lanes.get(spec.worker_id, LaneState())
        profile = _profile_for(spec, lane)
        try:
            health = host.session_health(spec)
        except HostCommandError:
            failures.append(spec.worker_id)
            if not dry_run:
                lane = replace(
                    lane,
                    next_retry_at=now + timedelta(seconds=15),
                    last_error="host-command-failure",
                )
                lane = _append_lane_event(
                    lane, now, "error", "host-command-failure"
                )
                lanes[spec.worker_id] = lane
            workers.append(_status_entry(spec, "error", profile, lane))
            continue
        diagnostic_output = _current_launch_output(lane, health)
        if is_healthy(spec, health):
            if not dry_run:
                lane = replace(
                    lane,
                    ordinary_failures=0,
                    next_retry_at=None,
                    processed_failure_digest="",
                )
                lane = _append_lane_event(lane, now, "healthy", "")
                lanes[spec.worker_id] = lane
            workers.append(_status_entry(spec, "healthy", profile, lane))
            continue
        if lane.next_retry_at is not None and now < lane.next_retry_at:
            if not dry_run:
                lane = _append_lane_event(
                    lane,
                    now,
                    "backed_off",
                    _classified_error(lane.last_error),
                )
                lanes[spec.worker_id] = lane
            workers.append(_status_entry(spec, "backed_off", profile, lane))
            continue
        quota_reset = parse_quota_reset(diagnostic_output, now)
        quota_match = QUOTA_RE.search(diagnostic_output)
        quota_digest = _quota_event_digest(quota_match) if quota_match else ""
        new_quota_event = (
            spec.agent == "agy"
            and quota_reset is not None
            and quota_digest != lane.processed_failure_digest
        )
        if new_quota_event:
            profile_backoff = dict(lane.profile_backoff)
            profile_backoff[profile.name] = quota_reset
            lane = replace(
                lane,
                profile_backoff=profile_backoff,
                next_retry_at=None,
                last_error=_classified_error(diagnostic_output),
                processed_failure_digest=quota_digest,
            )
        selected_profile, profile_index = next_profile(
            spec, lane, now, rotate=new_quota_event
        )
        if selected_profile is None:
            earliest_reset = _earliest_profile_reset(lane)
            assert earliest_reset is not None
            if dry_run:
                proposed.append(spec.worker_id)
                workers.append(_status_entry(spec, "proposed", profile, lane))
                continue
            lane = replace(
                lane,
                next_retry_at=earliest_reset,
                last_error=_classified_error(diagnostic_output),
            )
            lane = _append_lane_event(
                lane,
                now,
                "backed_off",
                _classified_error(lane.last_error),
            )
            lanes[spec.worker_id] = lane
            workers.append(_status_entry(spec, "backed_off", profile, lane))
            continue
        profile = selected_profile
        assert profile_index is not None
        command = launch_shell_command(spec, profile, repo_root)
        if dry_run:
            proposed.append(spec.worker_id)
            workers.append(_status_entry(spec, "proposed", profile, lane))
            continue
        launch_marker = secrets.token_hex(16)
        try:
            started_now = host.ensure_session(
                spec, _marked_launch_command(command, launch_marker)
            )
        except HostCommandError:
            failures.append(spec.worker_id)
            lane = replace(
                lane,
                next_retry_at=now + timedelta(seconds=15),
                last_error="host-command-failure",
            )
            lane = _append_lane_event(
                lane, now, "error", "host-command-failure"
            )
            lanes[spec.worker_id] = lane
            workers.append(_status_entry(spec, "error", profile, lane))
            continue
        if not started_now:
            lane = replace(
                lane,
                ordinary_failures=0,
                next_retry_at=None,
                processed_failure_digest="",
            )
            lane = _append_lane_event(lane, now, "healthy", "")
            lanes[spec.worker_id] = lane
            workers.append(_status_entry(spec, "healthy", profile, lane))
            continue
        ordinary_failures = lane.ordinary_failures
        if health.exists and quota_reset is None:
            ordinary_failures += 1
            next_retry_at = _ordinary_retry_at(lane, spec.worker_id, now)
        else:
            next_retry_at = now + timedelta(seconds=15)
        lane = replace(
            lane,
            profile_index=profile_index,
            profile_backoff=dict(lane.profile_backoff),
            restart_count=lane.restart_count + 1,
            ordinary_failures=ordinary_failures,
            next_retry_at=next_retry_at,
            last_error=_classified_error(diagnostic_output),
            processed_failure_digest="",
            launch_marker=launch_marker,
        )
        lane = _append_lane_event(
            lane,
            now,
            "started",
            _classified_error(lane.last_error),
        )
        lanes[spec.worker_id] = lane
        started.append(spec.worker_id)
        workers.append(_status_entry(spec, "started", profile, lane))
    status = {
        "observed_at": _iso_timestamp(now),
        "state": "partial" if failures else "running",
        "supervisor_errors": [],
        "workers": workers,
    }
    return ReconcileResult(
        RuntimeState(lanes),
        status,
        tuple(started),
        tuple(proposed),
        tuple(failures),
    )


def sanitize_text(text: str) -> str:
    clean = SECRET_ENV_ASSIGNMENT_PATTERN.sub("<redacted>", text)
    clean = HUGGING_FACE_TOKEN_PATTERN.sub("<redacted>", clean)
    clean = GITHUB_TOKEN_PATTERN.sub("<redacted>", clean)
    return BEARER_TOKEN_PATTERN.sub(r"\1<redacted>", clean)


def _runtime_paths(home: Path) -> tuple[Path, Path, Path]:
    state_dir = home / ".local/state/icml-worker-supervisor"
    return (
        state_dir / "status.json",
        state_dir / "runtime.json",
        state_dir / "supervisor.lock",
    )


def _smoke_request_path(home: Path) -> Path:
    return home / ".local/state/icml-worker-supervisor/smoke-request.json"


def _runtime_to_json(state: RuntimeState) -> dict[str, dict[str, dict[str, Any]]]:
    return {
        "lanes": {
            worker_id: {
                "profile_index": lane.profile_index,
                "profile_backoff": {
                    name: reset.astimezone(timezone.utc).isoformat()
                    for name, reset in lane.profile_backoff.items()
                },
                "restart_count": lane.restart_count,
                "ordinary_failures": lane.ordinary_failures,
                "next_retry_at": _iso_timestamp(lane.next_retry_at),
                "last_error": lane.last_error,
                "processed_failure_digest": lane.processed_failure_digest,
                "launch_marker": lane.launch_marker,
                "events": [
                    {
                        "observed_at": _iso_timestamp(event.observed_at),
                        "health": event.health,
                        "error_class": event.error_class,
                    }
                    for event in lane.events
                ],
            }
            for worker_id, lane in state.lanes.items()
        }
    }


def _parse_timestamp(value: object) -> datetime | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError("runtime timestamp must be a string or null")
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        raise ValueError("runtime timestamp must include a timezone")
    return parsed.astimezone(timezone.utc)


def _runtime_from_json(value: object) -> RuntimeState:
    if not isinstance(value, dict) or not isinstance(value.get("lanes"), dict):
        raise ValueError("runtime state must contain a lanes object")
    lanes: dict[str, LaneState] = {}
    for worker_id, raw_lane in value["lanes"].items():
        if not isinstance(worker_id, str) or not isinstance(raw_lane, dict):
            raise ValueError("runtime lanes must be objects keyed by worker ID")
        raw_backoff = raw_lane.get("profile_backoff", {})
        if not isinstance(raw_backoff, dict):
            raise ValueError("profile_backoff must be an object")
        raw_events = raw_lane.get("events", [])
        if not isinstance(raw_events, list):
            raise ValueError("lane events must be a list")
        events: list[LaneEvent] = []
        for raw_event in raw_events[-EVENT_HISTORY_LIMIT:]:
            if not isinstance(raw_event, dict):
                raise ValueError("lane events must be objects")
            observed_at = _parse_timestamp(raw_event.get("observed_at"))
            health = raw_event.get("health")
            error_class = raw_event.get("error_class")
            if observed_at is None:
                raise ValueError("lane event observed_at is required")
            if health not in HEALTH_CLASSES:
                raise ValueError("lane event health class is invalid")
            if error_class not in ERROR_CLASSES:
                raise ValueError("lane event error class is invalid")
            events.append(
                LaneEvent(
                    observed_at=observed_at,
                    health=str(health),
                    error_class=str(error_class),
                )
            )
        lanes[worker_id] = LaneState(
            profile_index=int(raw_lane.get("profile_index", 0)),
            profile_backoff={
                str(name): timestamp
                for name, reset in raw_backoff.items()
                if (timestamp := _parse_timestamp(reset)) is not None
            },
            restart_count=int(raw_lane.get("restart_count", 0)),
            ordinary_failures=int(raw_lane.get("ordinary_failures", 0)),
            next_retry_at=_parse_timestamp(raw_lane.get("next_retry_at")),
            last_error=_classified_error(
                str(raw_lane.get("last_error", ""))
            ),
            processed_failure_digest=str(
                raw_lane.get("processed_failure_digest", "")
            ),
            launch_marker=str(raw_lane.get("launch_marker", "")),
            events=tuple(events),
        )
    return RuntimeState(lanes)


def _load_runtime(path: Path) -> RuntimeState:
    if not path.exists():
        return RuntimeState.empty()
    return _runtime_from_json(json.loads(path.read_text(encoding="utf-8")))


def _load_smoke_request(path: Path) -> dict[str, str] | None:
    if not path.exists():
        return None
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or set(value) != {"nonce", "stage"}:
        raise ValueError(
            "smoke request must contain only nonce and stage"
        )
    nonce = value["nonce"]
    stage = value["stage"]
    if not isinstance(nonce, str) or not nonce or len(nonce) > 128:
        raise ValueError("smoke request nonce must be a nonempty string")
    if stage not in {"interrupted", "restored"}:
        raise ValueError("smoke request stage is invalid")
    return {"nonce": nonce, "stage": stage}


def _quarantine_smoke_request(path: Path) -> None:
    if path.exists():
        os.replace(
            path,
            path.with_name("smoke-request.quarantined.json"),
        )


def _restore_pending_smoke_request(host: HostAdapter, path: Path) -> None:
    if (
        path.exists()
        and time.time() - path.stat().st_mtime
        > SMOKE_REQUEST_STALE_AFTER
    ):
        raise SmokeRequestStale("smoke request is stale")
    request = _load_smoke_request(path)
    if request is None or request["stage"] == "restored":
        return
    try:
        host.restore_disposable_session(SMOKE_SESSION, SMOKE_RESTORE_COMMAND)
        if host.session_foreground_command(SMOKE_SESSION) != "sleep":
            raise SmokeRecoveryError("smoke command did not recover")
    except HostCommandError as error:
        raise SmokeRecoveryError("smoke host recovery failed") from error
    atomic_write_json(
        path, {"nonce": request["nonce"], "stage": "restored"}
    )


def _run_reconcile(
    host: HostAdapter,
    home: Path,
    now: datetime,
    repo_root: Path,
    *,
    dry_run: bool = False,
) -> ReconcileResult:
    status_path, runtime_path, lock_path = _runtime_paths(home)
    with exclusive_lock(lock_path):
        smoke_errors: list[str] = []
        if not dry_run:
            smoke_path = _smoke_request_path(home)
            try:
                _restore_pending_smoke_request(host, smoke_path)
            except SmokeRequestStale:
                smoke_errors.append("smoke-request-stale")
            except (json.JSONDecodeError, ValueError):
                smoke_errors.append("smoke-request-invalid")
            except (HostCommandError, OSError, SmokeRecoveryError):
                smoke_errors.append("smoke-recovery-failed")
            if smoke_errors:
                try:
                    _quarantine_smoke_request(smoke_path)
                except OSError:
                    smoke_errors[:] = ["smoke-recovery-failed"]
        state = _load_runtime(runtime_path)
        result = reconcile(host, state, now, repo_root, dry_run=dry_run)
        if not dry_run:
            if smoke_errors:
                status = dict(result.status)
                status["state"] = "partial"
                status["supervisor_errors"] = smoke_errors
                result = replace(
                    result,
                    status=status,
                    failures=(*result.failures, *smoke_errors),
                )
            atomic_write_json(runtime_path, _runtime_to_json(result.state))
            atomic_write_json(status_path, result.status)
    return result


def _print_zero_counts() -> None:
    print("agy 0/10")
    print("codex 0/5")


def _print_status(path: Path, now: datetime) -> int:
    try:
        status = json.loads(path.read_text(encoding="utf-8"))
        workers = status["workers"]
        if not isinstance(workers, list):
            raise TypeError("workers must be a list")
        observed_at = _parse_timestamp(status.get("observed_at"))
        if observed_at is None:
            raise TypeError("observed_at is required")
        state = status.get("state")
        if state not in {"running", "partial", "stopped"}:
            raise TypeError("state is invalid")
        supervisor_errors = status.get("supervisor_errors", [])
        if (
            not isinstance(supervisor_errors, list)
            or any(
                error not in SUPERVISOR_ERROR_CLASSES
                for error in supervisor_errors
            )
        ):
            raise TypeError("supervisor_errors is invalid")
    except (
        FileNotFoundError,
        json.JSONDecodeError,
        KeyError,
        TypeError,
        ValueError,
    ):
        print("status snapshot is unavailable", file=sys.stderr)
        _print_zero_counts()
        return 1

    if state == "stopped":
        _print_zero_counts()
        print(f"supervisor stopped at {_iso_timestamp(observed_at)}")
        return 0

    if now.astimezone(timezone.utc) - observed_at > STATUS_STALE_AFTER:
        print(
            f"status snapshot is stale: observed at {_iso_timestamp(observed_at)}",
            file=sys.stderr,
        )
        _print_zero_counts()
        return 1

    for agent, desired in (("agy", 10), ("codex", 5)):
        healthy = sum(
            worker.get("agent") == agent and worker.get("health") == "healthy"
            for worker in workers
            if isinstance(worker, dict)
        )
        print(f"{agent} {healthy}/{desired}")
    for worker in workers:
        if not isinstance(worker, dict) or worker.get("health") == "healthy":
            continue
        print(
            " ".join(
                (
                    str(worker.get("worker_id", "unknown")),
                    str(worker.get("health", "unknown")),
                    str(worker.get("last_error") or ""),
                )
            ).rstrip()
        )
    for error_class in supervisor_errors:
        print(f"supervisor partial {error_class}")
    return 1 if state == "partial" or supervisor_errors else 0


def _stopped_status(now: datetime) -> dict[str, Any]:
    workers: list[dict[str, Any]] = []
    for spec in desired_workers():
        lane = _append_lane_event(LaneState(), now, "stopped", "")
        workers.append(
            _status_entry(spec, "stopped", _profile_for(spec, lane), lane)
        )
    return {
        "observed_at": _iso_timestamp(now),
        "state": "stopped",
        "supervisor_errors": [],
        "workers": workers,
    }


def _render_systemd_units(repo_root: Path, python: Path) -> dict[str, str]:
    template_dir = repo_root / "ops" / "systemd"
    replacements = {
        "@REPO_ROOT@": str(repo_root),
        "@PYTHON@": str(python),
    }
    rendered: dict[str, str] = {}
    for filename in (
        "icml-worker-supervisor.service",
        "icml-worker-supervisor.timer",
    ):
        content = (template_dir / filename).read_text(encoding="utf-8")
        for placeholder, replacement in replacements.items():
            content = content.replace(placeholder, replacement)
        rendered[filename] = content
    return rendered


def _install(
    host: HostAdapter,
    home: Path,
    now: datetime,
    repo_root: Path,
    python: Path,
) -> int:
    unit_dir = home / ".config/systemd/user"
    for filename, content in _render_systemd_units(repo_root, python).items():
        atomic_write_text(unit_dir / filename, content)
    result = _run_reconcile(host, home, now, repo_root)
    host.systemctl_user("daemon-reload")
    host.systemctl_user("enable", "--now", "icml-worker-supervisor.timer")
    host.systemctl_user("start", "icml-worker-supervisor.service")
    return 1 if result.failures else 0


def _production_pids(host: HostAdapter) -> dict[str, int]:
    return {
        spec.session_name: host.pane_pid(spec.session_name)
        for spec in desired_workers()
    }


def _remove_owned_smoke_request(path: Path, nonce: str) -> None:
    try:
        request = _load_smoke_request(path)
    except (json.JSONDecodeError, ValueError):
        return
    if request is not None and request["nonce"] == nonce:
        path.unlink(missing_ok=True)


def _smoke_test(
    host: HostAdapter,
    home: Path,
    timeout: float,
) -> int:
    if not math.isfinite(timeout) or timeout < 0:
        print("smoke timeout must be finite and nonnegative", file=sys.stderr)
        return 2
    request_path = _smoke_request_path(home)
    if request_path.exists():
        print("smoke request already exists", file=sys.stderr)
        return 1

    nonce = secrets.token_hex(16)
    before = _production_pids(host)
    created = False
    try:
        host.create_disposable_session(SMOKE_SESSION, SMOKE_COMMAND)
        created = True
        host.interrupt_session(SMOKE_SESSION)
        atomic_write_json(
            request_path, {"nonce": nonce, "stage": "interrupted"}
        )

        if not host.wait_for_smoke_restore(request_path, nonce, timeout):
            print("smoke restoration timed out", file=sys.stderr)
            return 1

        if host.session_foreground_command(SMOKE_SESSION) != "sleep":
            print("smoke command was not restored", file=sys.stderr)
            return 1
        if _production_pids(host) != before:
            print("production worker pane PIDs changed", file=sys.stderr)
            return 1
        return 0
    finally:
        if created:
            host.stop_session(SMOKE_SESSION)
        _remove_owned_smoke_request(request_path, nonce)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Maintain direct paper-owner workers")
    commands = parser.add_subparsers(dest="command", required=True)
    reconcile_command = commands.add_parser("reconcile")
    reconcile_command.add_argument("--dry-run", action="store_true")
    commands.add_parser("status")
    commands.add_parser("install")
    smoke = commands.add_parser("smoke-test")
    smoke.add_argument("--timeout", type=float, default=45)
    stop = commands.add_parser("stop")
    stop.add_argument("--confirm", action="store_true")
    return parser


def main(
    argv: list[str] | None = None,
    *,
    host: HostAdapter | None = None,
    home: Path | None = None,
    now: datetime | None = None,
) -> int:
    args = _build_parser().parse_args(argv)
    repo_root = Path(__file__).resolve().parents[1]
    selected_home = Path.home() if home is None else home
    selected_now = datetime.now(timezone.utc) if now is None else now
    selected_host = HostAdapter(repo_root) if host is None else host

    try:
        if args.command == "reconcile":
            result = _run_reconcile(
                selected_host,
                selected_home,
                selected_now,
                repo_root,
                dry_run=args.dry_run,
            )
            if args.dry_run:
                for worker_id in result.proposed:
                    print(f"proposed {worker_id}")
            return 1 if result.failures else 0
        if args.command == "status":
            status_path, _, _ = _runtime_paths(selected_home)
            return _print_status(status_path, selected_now)
        if args.command == "install":
            return _install(
                selected_host,
                selected_home,
                selected_now,
                repo_root,
                Path(sys.executable).resolve(),
            )
        if args.command == "stop":
            if not args.confirm:
                print("stop requires --confirm", file=sys.stderr)
                return 2
            selected_host.systemctl_user(
                "disable", "--now", "icml-worker-supervisor.timer"
            )
            selected_host.systemctl_user(
                "stop", "icml-worker-supervisor.service"
            )
            status_path, _, lock_path = _runtime_paths(selected_home)
            with exclusive_lock(lock_path, wait=True):
                stop_failed = False
                for spec in desired_workers():
                    try:
                        selected_host.stop_session(spec.session_name)
                    except HostCommandError:
                        stop_failed = True
                if not stop_failed:
                    atomic_write_json(
                        status_path, _stopped_status(selected_now)
                    )
            return 1 if stop_failed else 0
        if args.command == "smoke-test":
            return _smoke_test(
                selected_host,
                selected_home,
                args.timeout,
            )
    except (
        AlreadyRunning,
        HostCommandError,
        KeyError,
        OSError,
        TypeError,
        ValueError,
    ) as error:
        print(f"supervisor failed: {sanitize_text(str(error))}", file=sys.stderr)
        return 1
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
