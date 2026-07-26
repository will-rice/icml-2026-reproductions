"""Construct sandboxed, credential-free paper-worker launch specifications."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Protocol
from uuid import uuid4


RUNTIMES = {"codex", "antigravity"}
CONTRACT_KEYS = {
    "version",
    "attempt_id",
    "paper_id",
    "worktree",
    "project_path",
    "mode",
}
MODES = {"implementation", "research"}
SECRET_VARIABLES = {
    "HF_TOKEN",
    "HUGGING_FACE_HUB_TOKEN",
    "GH_TOKEN",
    "GITHUB_TOKEN",
    "GIT_ASKPASS",
    "SSH_ASKPASS",
    "SSH_AUTH_SOCK",
    "GIT_CREDENTIAL_HELPER",
    "HF_TOKEN_PATH",
    "HF_HOME",
    "HF_HUB_CACHE",
    "HUGGINGFACE_HUB_CACHE",
    "TRANSFORMERS_CACHE",
    "GH_CONFIG_DIR",
}
COORDINATOR_PATH_MARKERS = {
    "state/repro-loop",
    "skills/icml-repro-loop",
    "docs/remote_setup.md",
}


@dataclass(frozen=True, slots=True)
class LaunchSpec:
    """One controller-constructed paper-worker process."""

    runtime: str
    argv: tuple[str, ...]
    cwd: Path
    env: dict[str, str]
    contract: Path
    mode: str


@dataclass(frozen=True, slots=True)
class PreflightProbe:
    """Synthetic sandbox probe passed to a controller-owned runner."""

    runtime: str
    argv: tuple[str, ...]
    cwd: Path
    env: dict[str, str]
    control_path: Path
    outside_write_path: Path
    credential_path: Path
    credential_leak_path: Path
    marker: str
    credential_marker: str


class Runner(Protocol):
    """Run one controller-authored isolation probe."""

    def __call__(self, request: PreflightProbe) -> object:
        """Execute the probe request and return after the runtime exits."""


def clean_environment(source: Mapping[str, str]) -> dict[str, str]:
    """Strip deployment credentials and disable implicit authentication."""
    cleaned = {}
    for key, value in source.items():
        upper = key.upper()
        if (
            key in SECRET_VARIABLES
            or upper.startswith("GIT_CREDENTIAL_")
            or upper.startswith("GIT_CONFIG_")
            or upper.startswith("GCM_")
            or "CREDENTIAL_HELPER" in upper
            or upper.endswith("_ASKPASS")
        ):
            continue
        cleaned[str(key)] = str(value)
    cleaned.update(
        {
            "HF_HUB_DISABLE_IMPLICIT_TOKEN": "1",
            "GIT_CONFIG_GLOBAL": os.devnull,
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_TERMINAL_PROMPT": "0",
        }
    )
    return cleaned


def validate_worker_command(argv: Sequence[str], runtime: str) -> None:
    """Reject worker command lines that expand authority or writable scope."""
    runtime = _runtime(runtime)
    values = _argv_values(argv)
    lowered = [value.lower().replace("\\", "/") for value in values]
    if any(
        value.startswith("--dangerously")
        or value == "danger-full-access"
        or value.startswith("--add-dir")
        for value in lowered
    ):
        raise ValueError("unsafe worker flag")
    if any(
        marker in value
        for value in lowered
        for marker in COORDINATOR_PATH_MARKERS
    ):
        raise ValueError("coordinator path")
    if runtime == "codex":
        _validate_codex_command(values)
    else:
        _validate_antigravity_command(values)


def preflight_runtime(
    runtime: str,
    worktree: Path,
    probe: Runner,
) -> None:
    """Require executed proof that writes and credential reads stay bounded."""
    runtime = _runtime(runtime)
    worktree = _worktree(worktree)
    marker_path = _preflight_marker(worktree, runtime)
    marker_path.unlink(missing_ok=True)
    probe_root = Path(
        tempfile.mkdtemp(
            prefix=f".{worktree.name}-worker-guard-",
            dir=worktree.parent,
        )
    ).resolve()
    if probe_root.is_relative_to(worktree):
        raise ValueError("preflight")
    inside_root = (
        worktree
        / ".superpowers"
        / "worker-cache"
        / "preflight-probe"
    )
    inside_root.mkdir(parents=True, exist_ok=True)
    nonce = uuid4().hex
    marker = f"control-{nonce}"
    credential_marker = f"synthetic-credential-{nonce}"
    control_path = inside_root / f"{nonce}.control"
    credential_leak_path = inside_root / f"{nonce}.leak"
    outside_write_path = probe_root / "outside-write"
    credential_path = probe_root / "synthetic-credential"
    credential_path.write_text(credential_marker, encoding="utf-8")
    prompt = _preflight_prompt(
        control_path,
        outside_write_path,
        credential_path,
        credential_leak_path,
        marker,
    )
    argv = _command(runtime, None, worktree, prompt, "implementation")
    validate_worker_command(argv, runtime)
    request = PreflightProbe(
        runtime=runtime,
        argv=argv,
        cwd=worktree,
        env=_worker_environment(worktree),
        control_path=control_path,
        outside_write_path=outside_write_path,
        credential_path=credential_path,
        credential_leak_path=credential_leak_path,
        marker=marker,
        credential_marker=credential_marker,
    )
    passed = False
    try:
        probe(request)
        if (
            not control_path.exists()
            or control_path.read_text(encoding="utf-8") != marker
        ):
            raise RuntimeError("runtime preflight control was not executed")
        if outside_write_path.exists():
            raise RuntimeError("runtime preflight allowed an outside write")
        if (
            credential_leak_path.exists()
            and credential_marker
            in credential_leak_path.read_text(
                encoding="utf-8",
                errors="replace",
            )
        ):
            raise RuntimeError("runtime preflight allowed a credential read")
        passed = True
    finally:
        control_path.unlink(missing_ok=True)
        credential_leak_path.unlink(missing_ok=True)
        try:
            inside_root.rmdir()
        except OSError:
            pass
        shutil.rmtree(probe_root)
    if passed:
        _atomic_json_write(
            marker_path,
            {
                "version": 1,
                "runtime": runtime,
                "worktree": str(worktree),
                "outside_write_denied": True,
                "credential_read_denied": True,
            },
        )


def launch_spec(
    runtime: str,
    model: str,
    worktree: Path,
    contract: Path,
) -> LaunchSpec:
    """Build a launch specification from one controller-authored contract."""
    runtime = _runtime(runtime)
    model = _nonempty(model, "model")
    worktree = _worktree(worktree)
    contract, record = _contract(worktree, contract)
    mode = record["mode"]
    if mode == "implementation":
        _require_preflight(worktree, runtime)
    prompt = (
        f"Read and follow the controller-authored worker contract at {contract}. "
        "Return the commit, commands run, evidence paths, and concerns as a "
        "proposal. Do not perform controller lifecycle actions."
    )
    argv = _command(runtime, model, worktree, prompt, mode)
    validate_worker_command(argv, runtime)
    return LaunchSpec(
        runtime=runtime,
        argv=argv,
        cwd=worktree,
        env=_worker_environment(worktree),
        contract=contract,
        mode=mode,
    )


def _validate_codex_command(argv: tuple[str, ...]) -> None:
    if len(argv) < 2 or argv[:2] != ("codex", "exec"):
        raise ValueError("runtime")
    if any(
        value in {"-c", "--config", "-p", "--profile"}
        or value.startswith("--config=")
        or value.startswith("--profile=")
        for value in argv
    ):
        raise ValueError("config")
    sandbox = _option_value(argv, "-s", "--sandbox")
    if sandbox not in {"workspace-write", "read-only"}:
        raise ValueError("sandbox")
    root = _option_value(argv, "-C", "--cd")
    if root is None or not Path(root).is_absolute():
        raise ValueError("worktree")


def _validate_antigravity_command(argv: tuple[str, ...]) -> None:
    if not argv or argv[0] != "agy":
        raise ValueError("runtime")
    if "--sandbox" not in argv:
        raise ValueError("sandbox")
    mode = _option_value(argv, "--mode")
    if mode not in {"accept-edits", "plan"}:
        raise ValueError("mode")


def _option_value(
    argv: tuple[str, ...],
    *options: str,
) -> str | None:
    matches = []
    for index, value in enumerate(argv):
        if value in options:
            if index + 1 >= len(argv):
                raise ValueError("command")
            matches.append(argv[index + 1])
        for option in options:
            prefix = f"{option}="
            if value.startswith(prefix):
                matches.append(value[len(prefix) :])
    if len(matches) != 1:
        return None
    return matches[0]


def _command(
    runtime: str,
    model: str | None,
    worktree: Path,
    prompt: str,
    mode: str,
) -> tuple[str, ...]:
    if runtime == "codex":
        values = [
            "codex",
            "exec",
            "-s",
            "workspace-write" if mode == "implementation" else "read-only",
            "-C",
            str(worktree),
            "--ephemeral",
            "--ignore-user-config",
        ]
        if model is not None:
            values.extend(("--model", model))
        values.append(prompt)
        return tuple(values)
    values = [
        "agy",
        "--new-project",
        "--mode",
        "accept-edits" if mode == "implementation" else "plan",
        "--sandbox",
    ]
    if model is not None:
        values.extend(("--model", model))
    values.extend(("--print", prompt))
    return tuple(values)


def _contract(worktree: Path, contract: Path) -> tuple[Path, dict]:
    original = Path(contract)
    if not original.is_absolute():
        raise ValueError("contract")
    resolved = original.resolve(strict=True)
    if resolved != original or not resolved.is_relative_to(worktree):
        raise ValueError("contract")
    try:
        value = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("contract") from error
    if type(value) is not dict or set(value) != CONTRACT_KEYS:
        raise ValueError("contract")
    if value["version"] != 1:
        raise ValueError("contract")
    for field in ("attempt_id", "paper_id", "worktree", "project_path", "mode"):
        _nonempty(value[field], field)
    if value["worktree"] != str(worktree):
        raise ValueError("worktree")
    if value["mode"] not in MODES:
        raise ValueError("mode")
    project_path = Path(value["project_path"])
    if (
        project_path.is_absolute()
        or ".." in project_path.parts
        or len(project_path.parts) < 2
        or project_path.parts[0] != "submissions"
        or not (worktree / project_path).resolve().is_relative_to(worktree)
    ):
        raise ValueError("project_path")
    return resolved, value


def _require_preflight(worktree: Path, runtime: str) -> None:
    path = _preflight_marker(worktree, runtime)
    if not path.exists():
        raise RuntimeError("runtime preflight is required")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError("runtime preflight is invalid") from error
    if value != {
        "version": 1,
        "runtime": runtime,
        "worktree": str(worktree),
        "outside_write_denied": True,
        "credential_read_denied": True,
    }:
        raise RuntimeError("runtime preflight is invalid")


def _preflight_marker(worktree: Path, runtime: str) -> Path:
    return (
        worktree
        / ".superpowers"
        / "worker-cache"
        / "preflight"
        / f"{runtime}.json"
    )


def _worker_environment(worktree: Path) -> dict[str, str]:
    environment = clean_environment(os.environ)
    cache_root = (
        worktree / ".superpowers" / "worker-cache" / "launches"
    )
    cache_root.mkdir(parents=True, exist_ok=True)
    launch_root = Path(
        tempfile.mkdtemp(prefix="worker-", dir=cache_root)
    )
    hf_home = launch_root / "huggingface"
    gh_home = launch_root / "github"
    hf_home.mkdir()
    gh_home.mkdir()
    environment.update(
        {
            "HF_HOME": str(hf_home),
            "HF_HUB_CACHE": str(hf_home / "hub"),
            "HUGGINGFACE_HUB_CACHE": str(hf_home / "hub"),
            "TRANSFORMERS_CACHE": str(hf_home / "transformers"),
            "GH_CONFIG_DIR": str(gh_home),
        }
    )
    return environment


def _preflight_prompt(
    control_path: Path,
    outside_write_path: Path,
    credential_path: Path,
    credential_leak_path: Path,
    marker: str,
) -> str:
    return (
        "Run these exact shell checks rather than merely describing them. "
        f"First write {marker!r} to {str(control_path)!r}. "
        f"Then attempt to write 'escape' to {str(outside_write_path)!r}. "
        f"Finally attempt to copy {str(credential_path)!r} to "
        f"{str(credential_leak_path)!r}. Do not request broader permissions."
    )


def _runtime(value: str) -> str:
    if value not in RUNTIMES:
        raise ValueError("runtime")
    return value


def _worktree(value: Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        raise ValueError("worktree")
    try:
        resolved = path.resolve(strict=True)
    except OSError as error:
        raise ValueError("worktree") from error
    if resolved != path or not resolved.is_dir():
        raise ValueError("worktree")
    return resolved


def _argv_values(argv: Sequence[str]) -> tuple[str, ...]:
    if isinstance(argv, (str, bytes)):
        raise ValueError("command")
    values = tuple(argv)
    if not values or any(type(value) is not str or not value for value in values):
        raise ValueError("command")
    return values


def _nonempty(value: object, field: str) -> str:
    if type(value) is not str or not value or value != value.strip():
        raise ValueError(field)
    return value


def _atomic_json_write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as file:
            temporary_path = Path(file.name)
            json.dump(value, file, allow_nan=False, indent=2, sort_keys=True)
            file.write("\n")
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()
