"""Construct sandboxed, credential-free paper-worker launch specifications."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import time
from typing import Protocol
from uuid import uuid4

import telemetry


RUNTIMES = {"codex", "antigravity"}
TERMINATION_GRACE_SECONDS = 5
KILL_REAP_SECONDS = 5
CONTRACT_KEYS = {
    "version",
    "attempt_id",
    "paper_id",
    "worktree",
    "project_path",
    "plan_path",
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
    plan: Path
    mode: str
    attempt_id: str
    paper_id: str
    project_path: str


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
    *,
    attempt_id: str,
    paper_id: str,
    project_path: str,
) -> LaunchSpec:
    """Build a launch specification from one controller-authored contract."""
    runtime = _runtime(runtime)
    model = _nonempty(model, "model")
    worktree = _worktree(worktree)
    contract, plan, record = _contract(worktree, contract)
    for field, expected in (
        ("attempt_id", attempt_id),
        ("paper_id", paper_id),
        ("project_path", project_path),
    ):
        if record[field] != _nonempty(expected, field):
            raise ValueError(field)
    mode = record["mode"]
    if mode == "implementation":
        _require_preflight(worktree, runtime)
    prompt = (
        f"Read and follow the controller-authored worker contract at {contract}. "
        f"Then read the approved implementation plan at {plan}. "
        "Execute that plan task by task, including its tests and commits. "
        "Include at least 200 substantive characters of judge-readable "
        "evidence in direct root pages/*.md files under the project path. "
        "Work directly in this session. Do not delegate or spawn nested agents. "
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
        plan=plan,
        mode=mode,
        attempt_id=attempt_id,
        paper_id=paper_id,
        project_path=project_path,
    )


def run_worker(
    paths,
    spec: LaunchSpec,
    *,
    timeout_seconds: int | None = None,
    work_kind: str = "implementation",
    process_factory=subprocess.Popen,
    utc_now=None,
    monotonic_ns=time.monotonic_ns,
    session_id_factory=None,
    git_head=None,
    termination_grace_seconds: int | float = TERMINATION_GRACE_SECONDS,
    kill_reap_seconds: int | float = KILL_REAP_SECONDS,
) -> dict:
    """Run one worker child and durably measure its process boundary."""
    if timeout_seconds is not None and (
        type(timeout_seconds) is not int or timeout_seconds <= 0
    ):
        raise ValueError("timeout_seconds")
    if work_kind not in {"implementation", "correction"}:
        raise ValueError("work_kind")
    termination_grace_seconds = _positive_seconds(
        termination_grace_seconds, "termination_grace_seconds"
    )
    kill_reap_seconds = _positive_seconds(
        kill_reap_seconds, "kill_reap_seconds"
    )
    utc_now = utc_now or _utc_now
    session_id_factory = session_id_factory or (lambda: uuid4().hex)
    git_head = git_head or _git_head
    session_id = session_id_factory()
    contract_sha256 = hashlib.sha256(spec.contract.read_bytes()).hexdigest()
    common = {
        "attempt_id": spec.attempt_id,
        "paper_id": spec.paper_id,
        "project_path": spec.project_path,
        "runtime": spec.runtime,
        "model": _option_value(spec.argv, "--model"),
        "worktree": str(spec.cwd),
        "contract_sha256": contract_sha256,
        "work_kind": work_kind,
    }
    telemetry.append_event(
        paths,
        session_id,
        0,
        "worker-queued",
        {**common, "observed_at": utc_now()},
    )
    git_sha_before = git_head(spec.cwd)
    process = process_factory(spec.argv, cwd=spec.cwd, env=spec.env)
    launch_counter = monotonic_ns()
    try:
        telemetry.append_event(
            paths,
            session_id,
            1,
            "worker-launched",
            {
                **common,
                "observed_at": utc_now(),
                "pid": process.pid,
                "monotonic_ns": launch_counter,
                "git_sha_before": git_sha_before,
            },
        )
    except BaseException:
        try:
            _terminate_and_reap(
                process,
                termination_grace_seconds,
                kill_reap_seconds,
            )
        except BaseException:
            pass
        raise

    interrupted = False
    exit_observed = True
    outcome = "proposal"
    try:
        return_code = process.wait(timeout=timeout_seconds)
        if return_code != 0:
            outcome = "failed"
    except subprocess.TimeoutExpired:
        outcome = "timed_out"
        return_code, exit_observed = _terminate_and_reap(
            process,
            termination_grace_seconds,
            kill_reap_seconds,
        )
    except KeyboardInterrupt:
        interrupted = True
        outcome = "interrupted"
        return_code, exit_observed = _terminate_and_reap(
            process,
            termination_grace_seconds,
            kill_reap_seconds,
        )

    exit_counter = monotonic_ns() if exit_observed else None
    exit_code, signal = _exit_status(return_code)
    git_sha_after = (
        _optional_git_head(git_head, spec.cwd) if exit_observed else None
    )
    telemetry.append_event(
        paths,
        session_id,
        2,
        "worker-exited",
        {
            **common,
            "observed_at": utc_now(),
            "monotonic_ns": exit_counter,
            "git_sha_after": git_sha_after,
            "exit_code": exit_code,
            "signal": signal,
            "downloaded_bytes": _downloaded_bytes(process),
            "outcome": outcome,
        },
    )
    summary = telemetry.summarize_worker_session(
        telemetry.read_session(paths, session_id)
    )
    result = {
        **summary,
        "outcome": outcome,
        "exit_code": exit_code,
        "signal": signal,
        "downloaded_bytes": _downloaded_bytes(process),
    }
    if interrupted:
        raise KeyboardInterrupt
    return result


def _validate_codex_command(argv: tuple[str, ...]) -> None:
    if len(argv) < 2 or argv[:2] != ("codex", "exec"):
        raise ValueError("runtime")
    if any(
        value in {"-p", "--profile"}
        or value.startswith("--profile=")
        or value.startswith("--config=")
        for value in argv
    ):
        raise ValueError("config")
    root = _option_value(argv, "-C", "--cd")
    if root is None or not Path(root).is_absolute():
        raise ValueError("worktree")
    sandbox = _option_value(argv, "-s", "--sandbox")
    configs = tuple(
        argv[index + 1]
        for index, value in enumerate(argv)
        if value in {"-c", "--config"} and index + 1 < len(argv)
    )
    if any(
        value in {"-c", "--config"} and index + 1 >= len(argv)
        for index, value in enumerate(argv)
    ):
        raise ValueError("config")
    if sandbox == "read-only":
        if configs:
            raise ValueError("config")
        return
    if sandbox is not None:
        raise ValueError("sandbox")
    if configs != _codex_permission_config(Path(root)):
        raise ValueError("config")


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
        ]
        if mode == "implementation":
            for config in _codex_permission_config(worktree):
                values.extend(("-c", config))
        else:
            values.extend(("-s", "read-only"))
        values.extend(
            (
                "-C",
                str(worktree),
                "--ephemeral",
                "--ignore-user-config",
            )
        )
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
        "--print-timeout",
        "2h",
    ]
    if model is not None:
        values.extend(("--model", model))
    values.extend(("--print", prompt))
    return tuple(values)


def _codex_permission_config(worktree: Path) -> tuple[str, ...]:
    """Build the one accepted least-privilege Codex implementation profile."""
    return (
        'default_permissions="paper-worker"',
        'permissions.paper-worker.description="Assigned paper worktree only."',
        "permissions.paper-worker.filesystem="
        f'{{":minimal"="read","{worktree}"="write",'
        '":tmpdir"="write",":slash_tmp"="write"}',
        "permissions.paper-worker.network.enabled=true",
        'permissions.paper-worker.network.domains={"*"="allow"}',
    )


def _contract(worktree: Path, contract: Path) -> tuple[Path, Path, dict]:
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
    for field in (
        "attempt_id",
        "paper_id",
        "worktree",
        "project_path",
        "plan_path",
        "mode",
    ):
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
    plan_path = Path(value["plan_path"])
    if (
        plan_path.is_absolute()
        or ".." in plan_path.parts
        or len(plan_path.parts) < 4
        or plan_path.parts[:3] != ("docs", "superpowers", "plans")
        or plan_path.suffix != ".md"
    ):
        raise ValueError("plan_path")
    try:
        plan = (worktree / plan_path).resolve(strict=True)
    except OSError as error:
        raise ValueError("plan_path") from error
    if (
        plan != worktree / plan_path
        or not plan.is_file()
        or not plan.is_relative_to(worktree)
    ):
        raise ValueError("plan_path")
    return resolved, plan, value


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


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _git_head(worktree: Path) -> str:
    return subprocess.run(
        ["git", "-C", str(worktree), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _exit_status(return_code: object) -> tuple[int | None, int | None]:
    if type(return_code) is not int:
        return None, None
    if return_code < 0:
        return None, -return_code
    return return_code, None


def _terminate_and_reap(
    process: object,
    termination_grace_seconds: int | float,
    kill_reap_seconds: int | float,
) -> tuple[object, bool]:
    process.terminate()
    try:
        return process.wait(timeout=termination_grace_seconds), True
    except subprocess.TimeoutExpired:
        process.kill()
        try:
            return process.wait(timeout=kill_reap_seconds), True
        except subprocess.TimeoutExpired:
            return None, False


def _optional_git_head(git_head, worktree: Path) -> str | None:
    try:
        return git_head(worktree)
    except Exception:
        return None


def _positive_seconds(value: object, field: str) -> int | float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value <= 0
    ):
        raise ValueError(field)
    return value


def _downloaded_bytes(process: object) -> int | None:
    value = getattr(process, "downloaded_bytes", None)
    if type(value) is int and value >= 0:
        return value
    return None


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
