"""Trusted controller operations for reproduction attempt authority."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys
import tempfile


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import attestations  # noqa: E402
import attempts  # noqa: E402
import leases  # noqa: E402
import store  # noqa: E402


MANIFEST_KEYS = {
    "worktree",
    "branch",
    "base_sha",
    "project_path",
    "design_path",
    "commands",
}
GIT_SHA_PATTERN = re.compile(r"(?:[0-9a-f]{40}|[0-9a-f]{64})")
ENVIRONMENT_ALLOWLIST = {
    "LANG",
    "LC_ALL",
    "PATH",
    "SSL_CERT_DIR",
    "SSL_CERT_FILE",
    "TMPDIR",
    "UV_CACHE_DIR",
    "VIRTUAL_ENV",
}


@dataclass(frozen=True, slots=True)
class CommandResult:
    """Captured result of one controller-run validation command."""

    argv: tuple[str, ...]
    returncode: int
    stdout: str
    stderr: str


Runner = Callable[[tuple[str, ...], Path], CommandResult]


def clean_validation_environment(isolated_home: Path) -> dict[str, str]:
    """Return a credential-free environment rooted in an empty home."""
    environment = {
        key: value
        for key in sorted(ENVIRONMENT_ALLOWLIST)
        if (value := os.environ.get(key)) is not None
    }
    environment.update(
        {
            "HF_HOME": str(isolated_home.parent / "hf-home"),
            "HF_HUB_DISABLE_IMPLICIT_TOKEN": "1",
            "HOME": str(isolated_home),
            "XDG_CACHE_HOME": str(isolated_home / "cache"),
            "XDG_CONFIG_HOME": str(isolated_home / "config"),
        }
    )
    return environment


def run_command(argv: tuple[str, ...], worktree: Path) -> CommandResult:
    """Run one command at the registered worktree with a sanitized environment."""
    with tempfile.TemporaryDirectory(prefix="icml-repro-validation-") as name:
        isolated_root = Path(name)
        isolated_home = isolated_root / "home"
        isolated_home.mkdir()
        (isolated_root / "hf-home").mkdir()
        result = subprocess.run(
            argv,
            cwd=worktree,
            text=True,
            capture_output=True,
            check=False,
            env=clean_validation_environment(isolated_home),
        )
    return CommandResult(argv, result.returncode, result.stdout, result.stderr)


def validation_now() -> datetime:
    """Return a fresh controller time for the post-validation fence."""
    return datetime.now(timezone.utc)


def attest_validation(
    paths: store.StatePaths,
    attempt_id: str,
    lease: leases.Lease,
    manifest: dict,
    runner: Runner,
    now: datetime,
) -> dict:
    """Run and attest one paper's complete validation under its exact boundary."""
    attempt = attempts.read_attempt(paths, attempt_id)
    _assert_attempt_fence(paths, attempt_id, lease, now)
    worktree, commands = _validate_manifest(attempt, manifest)
    check_results: list[CommandResult] = []

    top_level = _checked(
        runner, ("git", "rev-parse", "--show-toplevel"), worktree, check_results
    ).stdout.strip()
    if Path(top_level).resolve() != worktree:
        raise ValueError("worktree")
    _require_clean(
        _checked(
            runner,
            ("git", "status", "--porcelain"),
            worktree,
            check_results,
        )
    )
    branch = _checked(
        runner, ("git", "branch", "--show-current"), worktree, check_results
    ).stdout.strip()
    if branch != manifest["branch"]:
        raise ValueError("branch")
    source_commit = _git_sha(
        _checked(
            runner, ("git", "rev-parse", "HEAD"), worktree, check_results
        ).stdout.strip(),
        "source commit",
    )
    diff = _checked(
        runner,
        (
            "git",
            "diff",
            "--name-only",
            f"{manifest['base_sha']}...HEAD",
        ),
        worktree,
        check_results,
    )
    _validate_changed_paths(
        diff.stdout, manifest["project_path"], manifest["design_path"]
    )

    environment_results = []
    for argv in (
        ("git", "--version"),
        ("uv", "--version"),
        (sys.executable, "--version"),
    ):
        result = runner(argv, worktree)
        environment_results.append(result)
        if result.argv != argv or result.returncode != 0:
            raise ValueError("environment version")

    command_results = []
    for index, argv in enumerate(commands, start=1):
        result = runner(argv, worktree)
        command_results.append(result)
        if result.argv != argv or result.returncode != 0:
            raise ValueError(f"validation command {index}")

    _require_clean(
        _checked(
            runner,
            ("git", "status", "--porcelain"),
            worktree,
            check_results,
        )
    )
    final_branch = _checked(
        runner, ("git", "branch", "--show-current"), worktree, check_results
    ).stdout.strip()
    if final_branch != branch:
        raise ValueError("branch")
    final_commit = _git_sha(
        _checked(
            runner, ("git", "rev-parse", "HEAD"), worktree, check_results
        ).stdout.strip(),
        "source commit",
    )
    if final_commit != source_commit:
        raise ValueError("source commit")
    source_tree = _git_sha(
        _checked(
            runner,
            ("git", "rev-parse", "HEAD^{tree}"),
            worktree,
            check_results,
        ).stdout.strip(),
        "tree hash",
    )
    completed_at = validation_now()
    _assert_attempt_fence(paths, attempt_id, lease, completed_at)

    payload = {
        "worktree": str(worktree),
        "branch": branch,
        "base_sha": manifest["base_sha"],
        "project_path": manifest["project_path"],
        "design_path": manifest["design_path"],
        "commands": [_result_record(result) for result in command_results],
        "checks": [_result_record(result) for result in check_results],
        "environment": [
            _result_record(result) for result in environment_results
        ],
        "source_tree": source_tree,
        "environment_sha256": _sha256_json(
            clean_validation_environment(
                Path("/isolated-validation/home")
            )
        ),
    }
    attempt_number = attempt.get("improvement_attempts", 0) + 1
    record = {
        "kind": "validation",
        "attempt_id": attempt_id,
        "attempt_number": attempt_number,
        "observed_at": _aware_timestamp(completed_at),
        "source_commit": source_commit,
        "payload_sha256": _sha256_json(payload),
        **payload,
    }
    attestation_id = attestations.persist(paths, record)
    return attempts.transition_attested(
        paths,
        attempt_id,
        "validated",
        attestation_id,
        {},
        lease,
        completed_at,
    )


def _validate_manifest(
    attempt: dict, manifest: object
) -> tuple[Path, tuple[tuple[str, ...], ...]]:
    if type(manifest) is not dict or set(manifest) != MANIFEST_KEYS:
        raise ValueError("manifest")
    for field in (
        "worktree",
        "branch",
        "base_sha",
        "project_path",
        "design_path",
    ):
        _nonempty_string(manifest[field], field)
    worktree_value = Path(manifest["worktree"])
    if not worktree_value.is_absolute():
        raise ValueError("worktree")
    try:
        worktree = worktree_value.resolve(strict=True)
    except OSError as error:
        raise ValueError("worktree") from error
    if worktree != worktree_value or not worktree.is_dir():
        raise ValueError("worktree")
    if "\n" in manifest["branch"] or "\r" in manifest["branch"]:
        raise ValueError("branch")
    _git_sha(manifest["base_sha"], "base_sha")
    project_path = _relative_path(manifest["project_path"], "project_path")
    design_path = _relative_path(manifest["design_path"], "design_path")
    if attempt.get("project_path") != project_path:
        raise ValueError("project_path")
    design = attempt.get("design")
    review = attempt.get("design_review")
    if type(design) is not dict or design.get("path") != design_path:
        raise ValueError("design_path")
    if type(review) is not dict or review.get("decision") != "approved":
        raise ValueError("design_review")
    if attempt.get("phase") not in {"implementing", "improving"}:
        raise ValueError("phase")
    commands = _validation_commands(manifest["commands"], project_path)
    return worktree, commands


def _validation_commands(
    value: object, project_path: str
) -> tuple[tuple[str, ...], ...]:
    if type(value) is not list or len(value) != 5:
        raise ValueError("commands")
    commands = []
    for command in value:
        if (
            type(command) is not list
            or not command
            or any(
                type(argument) is not str
                or not argument
                or "\x00" in argument
                for argument in command
            )
        ):
            raise ValueError("commands")
        commands.append(tuple(command))
    evidence, paper_pytest, root_pytest, skill_validation, pre_commit = commands
    if (
        evidence[:3] == ("uv", "run", "pytest")
        or not _references_project(evidence, project_path)
    ):
        raise ValueError("commands")
    if (
        paper_pytest[:3] != ("uv", "run", "pytest")
        or not _references_project(paper_pytest, project_path)
        or any(
            _pytest_bypass(argument)
            for argument in paper_pytest[3:]
        )
    ):
        raise ValueError("commands")
    if root_pytest != ("uv", "run", "pytest", "-q"):
        raise ValueError("commands")
    if (
        len(skill_validation) != 4
        or skill_validation[:2] != ("uv", "run")
        or Path(skill_validation[2]).name != "quick_validate.py"
        or skill_validation[3] != "skills/icml-repro-loop"
    ):
        raise ValueError("commands")
    if pre_commit != ("uv", "run", "pre-commit", "run", "-a"):
        raise ValueError("commands")
    return tuple(commands)


def _references_project(argv: tuple[str, ...], project_path: str) -> bool:
    return any(
        argument == project_path or argument.startswith(f"{project_path}/")
        for argument in argv
    )


def _pytest_bypass(argument: str) -> bool:
    exact = {
        "-k",
        "-m",
        "--collect-only",
        "--ff",
        "--lf",
        "--last-failed",
        "-x",
    }
    prefixes = (
        "-k",
        "-m",
        "--collect-only",
        "--deselect",
        "--ff",
        "--ignore",
        "--lf",
        "--maxfail",
    )
    return argument in exact or argument.startswith(prefixes)


def _validate_changed_paths(
    output: str, project_path: str, design_path: str
) -> None:
    project = PurePosixPath(project_path)
    for line in output.splitlines():
        if not line:
            continue
        changed = PurePosixPath(_relative_path(line, "changed path"))
        if changed != PurePosixPath(design_path) and not changed.is_relative_to(
            project
        ):
            raise ValueError("changed path")


def _checked(
    runner: Runner,
    argv: tuple[str, ...],
    worktree: Path,
    records: list[CommandResult],
) -> CommandResult:
    result = runner(argv, worktree)
    records.append(result)
    if result.argv != argv or result.returncode != 0:
        raise ValueError(f"git check: {' '.join(argv)}")
    return result


def _require_clean(result: CommandResult) -> None:
    if result.stdout.strip():
        raise ValueError("clean worktree")


def _assert_attempt_fence(
    paths: store.StatePaths,
    attempt_id: str,
    lease: leases.Lease,
    observed_at: datetime,
) -> None:
    if (
        lease.resource != f"attempt:{attempt_id}"
        or lease.attempt_id != attempt_id
    ):
        raise leases.StaleFence(f"attempt:{attempt_id}")
    leases.assert_fence(paths, lease, observed_at)


def _result_record(result: CommandResult) -> dict:
    return {
        "argv": list(result.argv),
        "returncode": result.returncode,
        "stdout_sha256": hashlib.sha256(result.stdout.encode("utf-8")).hexdigest(),
        "stderr_sha256": hashlib.sha256(result.stderr.encode("utf-8")).hexdigest(),
    }


def _relative_path(value: object, field: str) -> str:
    value = _nonempty_string(value, field)
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or path.as_posix() != value
        or any(part in {".", ".."} for part in path.parts)
    ):
        raise ValueError(field)
    return value


def _git_sha(value: object, field: str) -> str:
    value = _nonempty_string(value, field)
    if GIT_SHA_PATTERN.fullmatch(value) is None:
        raise ValueError(field)
    return value


def _nonempty_string(value: object, field: str) -> str:
    if type(value) is not str or not value or value != value.strip():
        raise ValueError(field)
    return value


def _aware_timestamp(value: datetime) -> str:
    if (
        not isinstance(value, datetime)
        or value.tzinfo is None
        or value.utcoffset() is None
    ):
        raise ValueError("observed_at")
    return value.isoformat()


def _sha256_json(value: object) -> str:
    encoded = json.dumps(
        value, allow_nan=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
