from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import tempfile
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
PROMPT = (
    "Use the shared icml-repro-loop skill directly and keep running its "
    "paper-owner loop. Read and follow "
    "/home/will/.agents/skills/icml-repro-loop/SKILL.md. "
    "Persistent worker ID: {worker_id}."
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


@dataclass(frozen=True)
class LaneState:
    profile_index: int = 0
    profile_backoff: dict[str, datetime] = field(default_factory=dict)
    restart_count: int = 0
    ordinary_failures: int = 0
    next_retry_at: datetime | None = None
    last_error: str = ""


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
    status: dict[str, list[dict[str, str | int | None]]]
    started: tuple[str, ...]
    proposed: tuple[str, ...]


class AlreadyRunning(RuntimeError):
    """Raised when a concurrent supervisor already holds the runtime lock."""


class HostCommandError(RuntimeError):
    """Raised when tmux cannot complete a host operation."""

    def __init__(self, stderr: str):
        super().__init__(f"tmux command failed: {sanitize_text(stderr).strip()}")


class HostAdapter:
    def __init__(self, repo_root: Path):
        self.repo_root = repo_root

    def _run_tmux(self, argv: list[str]) -> subprocess.CompletedProcess[str]:
        return subprocess.run(argv, check=False, text=True, capture_output=True)

    def session_health(self, spec: WorkerSpec) -> SessionHealth:
        pane_argv = [
            "tmux",
            "list-panes",
            "-t",
            spec.session_name,
            "-F",
            "#{pane_dead}\t#{pane_current_command}\t#{pane_pid}",
        ]
        pane_result = self._run_tmux(pane_argv)
        if pane_result.returncode:
            if "can't find session" in pane_result.stderr.lower():
                return SessionHealth(False, False, "", "")
            raise HostCommandError(pane_result.stderr)

        try:
            pane_dead, foreground_command, _pane_pid = pane_result.stdout.splitlines()[
                0
            ].split("\t", maxsplit=2)
        except (IndexError, ValueError) as error:
            raise HostCommandError("tmux returned malformed pane metadata") from error

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
        )

    def ensure_session(self, spec: WorkerSpec, command: str) -> None:
        if not self.session_health(spec).exists:
            create_argv = [
                "tmux",
                "new-session",
                "-d",
                "-s",
                spec.session_name,
                "-c",
                str(self.repo_root),
            ]
            create_result = self._run_tmux(create_argv)
            if create_result.returncode:
                raise HostCommandError(create_result.stderr)

        send_argv = ["tmux", "send-keys", "-t", spec.session_name, command, "C-m"]
        send_result = self._run_tmux(send_argv)
        if send_result.returncode:
            raise HostCommandError(send_result.stderr)


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
    )


def runtime_directory() -> Path:
    state_home = Path(
        os.environ.get("XDG_STATE_HOME", str(Path.home() / ".local" / "state"))
    )
    path = state_home / "icml-worker-supervisor"
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    return path


@contextmanager
def exclusive_lock(path: Path) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    lock_file = path.open("a+")
    try:
        try:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
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
    credentials = 'HF_TOKEN="$(hf auth token)" GH_TOKEN="$(gh auth token)"'
    if spec.agent == "agy":
        return " ".join(
            (
                credentials,
                'HF_HOME="/tmp/icml-agy-hf-XX"',
                'UV_CACHE_DIR="/tmp/icml-repro-uv-cache"',
                shlex.join((*profile.argv, prompt)),
            )
        )
    return " ".join(
        (
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
    spec: WorkerSpec, lane: LaneState, now: datetime
) -> tuple[ModelProfile | None, int | None]:
    if spec.agent == "codex":
        return codex_profile(), 0
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


def _iso_timestamp(value: datetime | None) -> str | None:
    if value is None:
        return None
    return value.astimezone(timezone.utc).isoformat()


def _status_entry(
    spec: WorkerSpec, health: str, profile: ModelProfile, lane: LaneState
) -> dict[str, str | int | None]:
    return {
        "worker_id": spec.worker_id,
        "agent": spec.agent,
        "health": health,
        "model": profile.name,
        "restart_count": lane.restart_count,
        "next_retry_at": _iso_timestamp(lane.next_retry_at),
        "last_error": sanitize_text(lane.last_error),
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
    workers: list[dict[str, str | int | None]] = []
    for spec in desired_workers():
        lane = lanes.get(spec.worker_id, LaneState())
        health = host.session_health(spec)
        profile = _profile_for(spec, lane)
        if is_healthy(spec, health):
            if lane.ordinary_failures:
                lane = LaneState(
                    profile_index=lane.profile_index,
                    profile_backoff=dict(lane.profile_backoff),
                    restart_count=lane.restart_count,
                    next_retry_at=None,
                    last_error=lane.last_error,
                )
                lanes[spec.worker_id] = lane
            workers.append(_status_entry(spec, "healthy", profile, lane))
            continue
        if lane.next_retry_at is not None and now < lane.next_retry_at:
            workers.append(_status_entry(spec, "backed_off", profile, lane))
            continue
        quota_reset = parse_quota_reset(health.recent_output, now)
        if spec.agent == "agy" and quota_reset is not None:
            profile_backoff = dict(lane.profile_backoff)
            profile_backoff[profile.name] = quota_reset
            lane = LaneState(
                profile_index=lane.profile_index,
                profile_backoff=profile_backoff,
                restart_count=lane.restart_count,
                ordinary_failures=lane.ordinary_failures,
                next_retry_at=None,
                last_error=sanitize_text(health.recent_output),
            )
        selected_profile, profile_index = next_profile(spec, lane, now)
        if selected_profile is None:
            earliest_reset = _earliest_profile_reset(lane)
            assert earliest_reset is not None
            if dry_run:
                proposed.append(spec.worker_id)
                workers.append(_status_entry(spec, "proposed", profile, lane))
                continue
            lane = LaneState(
                profile_index=lane.profile_index,
                profile_backoff=dict(lane.profile_backoff),
                restart_count=lane.restart_count,
                ordinary_failures=lane.ordinary_failures,
                next_retry_at=earliest_reset,
                last_error=sanitize_text(health.recent_output),
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
        host.ensure_session(spec, command)
        ordinary_failures = lane.ordinary_failures
        if health.exists and quota_reset is None:
            ordinary_failures += 1
            next_retry_at = _ordinary_retry_at(lane, spec.worker_id, now)
        else:
            next_retry_at = now + timedelta(seconds=15)
        lane = LaneState(
            profile_index=profile_index,
            profile_backoff=dict(lane.profile_backoff),
            restart_count=lane.restart_count + 1,
            ordinary_failures=ordinary_failures,
            next_retry_at=next_retry_at,
            last_error=sanitize_text(health.recent_output),
        )
        lanes[spec.worker_id] = lane
        started.append(spec.worker_id)
        workers.append(_status_entry(spec, "started", profile, lane))
    return ReconcileResult(RuntimeState(lanes), {"workers": workers}, tuple(started), tuple(proposed))


def sanitize_text(text: str) -> str:
    clean = SECRET_ENV_ASSIGNMENT_PATTERN.sub("<redacted>", text)
    clean = HUGGING_FACE_TOKEN_PATTERN.sub("<redacted>", clean)
    clean = GITHUB_TOKEN_PATTERN.sub("<redacted>", clean)
    return BEARER_TOKEN_PATTERN.sub(r"\1<redacted>", clean)
