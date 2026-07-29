from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal
import re
import shlex

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
