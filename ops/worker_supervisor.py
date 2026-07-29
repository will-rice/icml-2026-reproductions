from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import fcntl
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


def sanitize_text(text: str) -> str:
    clean = HUGGING_FACE_TOKEN_PATTERN.sub("<redacted>", text)
    clean = GITHUB_TOKEN_PATTERN.sub("<redacted>", clean)
    return BEARER_TOKEN_PATTERN.sub(r"\1<redacted>", clean)
