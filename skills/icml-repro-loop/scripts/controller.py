"""Trusted controller operations for reproduction attempt authority."""

import copy
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
import publication_policy  # noqa: E402
import refresh  # noqa: E402
import scheduler  # noqa: E402
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
    "PRE_COMMIT_HOME",
    "SSL_CERT_DIR",
    "SSL_CERT_FILE",
    "UV_CACHE_DIR",
    "UV_OFFLINE",
}



@dataclass(frozen=True, slots=True)
class CommandResult:
    """Captured result of one controller-run validation command."""

    argv: tuple[str, ...]
    returncode: int
    stdout: str
    stderr: str


Runner = Callable[[tuple[str, ...], Path], CommandResult]


def clean_validation_environment(
    isolated_home: Path, worktree: Path | None = None
) -> dict[str, str]:
    """Return a credential-free environment rooted in an empty home."""
    environment = {
        key: value
        for key in sorted(ENVIRONMENT_ALLOWLIST)
        if (value := os.environ.get(key)) is not None
    }
    inherited_virtual_env = os.environ.get("VIRTUAL_ENV")
    if inherited_virtual_env and "PATH" in environment:
        virtual_environment = Path(inherited_virtual_env)
        unsafe_entries = {
            (virtual_environment / "bin").resolve(),
            (virtual_environment / "Scripts").resolve(),
        }
        environment["PATH"] = os.pathsep.join(
            entry
            for entry in environment["PATH"].split(os.pathsep)
            if Path(entry).resolve() not in unsafe_entries
        )
    if worktree and (worktree / ".venv").exists():
        venv_path = worktree / ".venv"
        environment["VIRTUAL_ENV"] = str(venv_path)
        bin_dir = str(venv_path / "bin")
        if "PATH" in environment:
            environment["PATH"] = bin_dir + os.pathsep + environment["PATH"]
        else:
            environment["PATH"] = bin_dir
    environment.update(
        {
            "HF_HOME": str(isolated_home.parent / "hf-home"),
            "HF_HUB_DISABLE_IMPLICIT_TOKEN": "1",
            "HOME": str(isolated_home),
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTEST_ADDOPTS": "-p no:cacheprovider --ignore=submissions",
            "TMPDIR": str(isolated_home.parent / "tmp"),
            "UV_NO_SYNC": "1",
            "UV_OFFLINE": "1",
            "UV_SYSTEM_PYTHON": "1",
            "UV_PROJECT_ENVIRONMENT": (
                str(worktree / ".venv")
                if worktree and (worktree / ".venv").exists()
                else str(isolated_home.parent / "uv-project-environment")
            ),
            "XDG_CACHE_HOME": str(isolated_home / "cache"),
            "XDG_CONFIG_HOME": str(isolated_home / "config"),
        }
    )
    if "PRE_COMMIT_HOME" not in environment:
        default_precommit = Path("/home/will/.cache/pre-commit")
        if default_precommit.exists():
            environment["PRE_COMMIT_HOME"] = str(default_precommit)
    return environment


def run_command(argv: tuple[str, ...], worktree: Path) -> CommandResult:
    """Run one command at the registered worktree with a sanitized environment."""
    with tempfile.TemporaryDirectory(
        prefix="icml-repro-validation-",
        dir="/tmp",
    ) as name:
        isolated_root = Path(name)
        isolated_home = isolated_root / "home"
        isolated_home.mkdir()
        (isolated_root / "hf-home").mkdir()
        (isolated_root / "tmp").mkdir()
        result = subprocess.run(
            argv,
            cwd=worktree,
            text=True,
            capture_output=True,
            check=False,
            env=clean_validation_environment(isolated_home, worktree),
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
    transition_updates = (
        {"project_path": manifest["project_path"]}
        if attempt.get("project_path") is None
        else {}
    )
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
    _require_clean(
        _checked(
            runner,
            _project_status_command(manifest["project_path"]),
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
    _require_clean(
        _checked(
            runner,
            _project_status_command(manifest["project_path"]),
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
        "source_tree_sha256": _source_tree_sha256(
            worktree / manifest["project_path"]
        ),
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
        transition_updates,
        lease,
        completed_at,
    )


def publish_and_attest_deployment(
    paths: store.StatePaths,
    attempt_id: str,
    lease: leases.Lease,
    space_id: str,
    source_dir: Path,
    client,
    now: datetime,
) -> dict:
    """Publish one validated source tree and attest its exact live Space."""
    attempt = attempts.read_attempt(paths, attempt_id)
    _assert_attempt_fence(paths, attempt_id, lease, now)
    if attempt.get("phase") != "validated":
        raise ValueError("phase")
    validation = _authoritative_attestation(
        paths, attempt, "validation", "validated"
    )
    owner = publication_policy.space_owner(space_id)
    if owner not in publication_policy.ALLOWED_SPACE_OWNERS:
        raise ValueError("owner")

    worktree = Path(validation["worktree"])
    expected_source = _validated_project_source(
        worktree,
        validation["project_path"],
        "source_dir",
    )
    actual_source = _resolved_directory(Path(source_dir), "source_dir")
    if actual_source != expected_source:
        raise ValueError("source_dir")
    _require_current_validated_tree(worktree, actual_source, validation)
    _require_scoring_pages(actual_source)

    try:
        client.create_repo(
            repo_id=space_id,
            repo_type="space",
            space_sdk="gradio",
            exist_ok=True,
        )
    except Exception:
        if not client.repo_exists(repo_id=space_id, repo_type="space"):
            raise
    upload = client.upload_folder(
        repo_id=space_id,
        folder_path=actual_source,
        repo_type="space",
        commit_message=f"Publish validated {validation['source_commit']}",
    )
    upload_sha = _nonempty_string(
        _hub_attribute(upload, "oid"), "upload revision"
    )
    info = client.space_info(repo_id=space_id, files_metadata=True)
    live_space_id = _hub_attribute(info, "id")
    if live_space_id != space_id:
        raise ValueError("space")
    live_sha = _nonempty_string(
        _hub_attribute(info, "sha"), "space revision"
    )
    if live_sha != upload_sha:
        raise ValueError("revision")
    live_owner = publication_policy.space_owner(live_space_id)
    if live_owner not in publication_policy.ALLOWED_SPACE_OWNERS:
        raise ValueError("owner")
    tags = _hub_attribute(info, "tags")
    if (
        type(tags) not in {list, tuple}
        or any(type(tag) is not str or not tag for tag in tags)
    ):
        raise ValueError("tag")
    normalized_tags = sorted(set(tags))
    required_tags = {"icml2026-repro", f"paper-{attempt['paper_id']}"}
    if not required_tags.issubset(normalized_tags):
        raise ValueError("tag")
    runtime = _hub_attribute(info, "runtime")
    stage = _hub_attribute(runtime, "stage")
    if stage != "RUNNING":
        raise ValueError("runtime")
    completed_at = validation_now()
    _assert_attempt_fence(paths, attempt_id, lease, completed_at)

    payload = {
        "space_id": live_space_id,
        "space_sha": live_sha,
        "owner": live_owner,
        "tags": normalized_tags,
        "runtime_stage": stage,
        "validation_attestation_id": validation["attestation_id"],
        "source_tree_sha256": validation["source_tree_sha256"],
    }
    record = {
        "kind": "deployment",
        "attempt_id": attempt_id,
        "attempt_number": validation["attempt_number"],
        "observed_at": _aware_timestamp(completed_at),
        "source_commit": validation["source_commit"],
        "payload_sha256": _sha256_json(payload),
        **payload,
    }
    attestation_id = attestations.persist(paths, record)
    return attempts.transition_attested(
        paths,
        attempt_id,
        "deployed",
        attestation_id,
        {"space_id": live_space_id, "deployed_sha": live_sha},
        lease,
        completed_at,
    )


def _require_scoring_pages(source_dir: Path) -> None:
    """Require the exact root Markdown surface consumed by the official judge.

    The judge reads only pages/*.md. Every judged summary-only logbook has
    scored zero, so the surface must be plural pages carrying concrete
    numeric results, not one manifest of assertions.
    """
    pages = source_dir / "pages"
    if not pages.is_dir() or pages.is_symlink():
        raise ValueError("scoring pages")
    markdown = sorted(
        path
        for path in pages.glob("*.md")
        if path.is_file() and not path.is_symlink()
    )
    if len(markdown) < 2:
        raise ValueError("scoring pages")
    try:
        texts = [path.read_text(encoding="utf-8") for path in markdown]
    except (OSError, UnicodeError) as error:
        raise ValueError("scoring pages") from error
    substantive_characters = sum(len(text.strip()) for text in texts)
    if substantive_characters < 200:
        raise ValueError("scoring pages")
    numeric_lines = sum(
        1
        for text in texts
        for line in text.splitlines()
        if any(character.isdigit() for character in line)
    )
    if numeric_lines < 15:
        raise ValueError("scoring pages")


def attest_submission(
    paths: store.StatePaths,
    attempt_id: str,
    lease: leases.Lease,
    snapshot_id: str,
    now: datetime,
) -> dict:
    """Attest one exact tagged-Space queue observation from a live snapshot."""
    attempt = attempts.read_attempt(paths, attempt_id)
    _assert_attempt_fence(paths, attempt_id, lease, now)
    if attempt.get("phase") != "deployed":
        raise ValueError("phase")
    deployment = _authoritative_attestation(
        paths, attempt, "deployment", "deployed"
    )
    snapshot = refresh.read_snapshot(paths, snapshot_id)
    fetched_at = _parsed_timestamp(snapshot["fetched_at"], "snapshot")
    deployed_at = _parsed_timestamp(deployment["observed_at"], "deployment")
    if fetched_at <= deployed_at or fetched_at > now:
        raise ValueError("snapshot")

    paper_id = attempt["paper_id"]
    space_id = deployment["space_id"]
    space_sha = deployment["space_sha"]

    exact_spaces = [
        space for space in snapshot["spaces"] if space["space_id"] == space_id
    ]
    if len(exact_spaces) != 1:
        raise ValueError("space")
    exact_space = exact_spaces[0]
    if exact_space["revision"] != space_sha:
        raise ValueError("revision")
    if exact_space["paper_ids"] != [paper_id]:
        raise ValueError("paper")
    owner = publication_policy.space_owner(space_id)
    canonical_spaces = [
        space
        for space in snapshot["spaces"]
        if publication_policy.space_owner(space["space_id"]) == owner
        and paper_id in space["paper_ids"]
    ]
    if len(canonical_spaces) != 1:
        raise ValueError("duplicate")

    tagged = [
        record
        for record in snapshot["tagged_spaces"]
        if record.get("paper_id") == paper_id
        and record.get("space_id") == space_id
        and record.get("revision") == space_sha
    ]
    if len(tagged) != 1:
        raise ValueError("paper association")
    queued = [
        record
        for record in snapshot["queued_submissions"]
        if record.get("paper_id") == paper_id
        and record.get("space_id") == space_id
    ]
    if len(queued) == 1:
        queue = queued[0]
        if queue.get("revision") != space_sha:
            raise ValueError("revision")
        if queue.get("status") != "pending":
            raise ValueError("queue")
        queue_status = queue["status"]
    elif not queued:
        # No live queue entry: the exact deployed revision is acceptable only
        # when the official verdict feed already judged this exact Space SHA.
        official = _exact_official_verdict(snapshot, paper_id, space_id)
        if official.get("sha") != space_sha:
            raise ValueError("queue")
        queue_status = "judged"
    else:
        raise ValueError("queue")
    verdict_revision = _snapshot_verdict_revision(snapshot)

    payload = {
        "snapshot_id": snapshot_id,
        "verdict_revision": verdict_revision,
        "space_id": space_id,
        "space_sha": space_sha,
        "paper_id": paper_id,
        "queue_status": queue_status,
        "deployment_attestation_id": deployment["attestation_id"],
    }
    record = {
        "kind": "submission",
        "attempt_id": attempt_id,
        "attempt_number": deployment["attempt_number"],
        "observed_at": _aware_timestamp(now),
        "source_commit": deployment["source_commit"],
        "payload_sha256": _sha256_json(payload),
        **payload,
    }
    attestation_id = attestations.persist(paths, record)
    return attempts.transition_attested(
        paths,
        attempt_id,
        "submitted",
        attestation_id,
        {},
        lease,
        now,
    )


def sync_verdict(
    paths: store.StatePaths,
    attempt_id: str,
    lease: leases.Lease,
    snapshot_id: str,
    now: datetime,
    improvement_reason: str | None = None,
) -> dict:
    """Import one exact official verdict and complete or correct atomically."""
    attempt = attempts.read_attempt(paths, attempt_id)
    _assert_attempt_fence(paths, attempt_id, lease, now)
    if attempt.get("phase") != "judging":
        raise ValueError("phase")
    submission = _authoritative_attestation(
        paths, attempt, "submission", "submitted"
    )
    authority = _authoritative_attestation(
        paths, attempt, "authority-audit", "judging"
    )
    snapshot = scheduler.read_fresh_snapshot(paths, snapshot_id, now)
    verdict_revision = _snapshot_verdict_revision(snapshot)
    official = _exact_official_verdict(
        snapshot,
        attempt["paper_id"],
        submission["space_id"],
    )
    if official.get("source_revision") != verdict_revision:
        raise ValueError("source_revision")
    if official.get("paper_id") != attempt["paper_id"]:
        raise ValueError("paper")
    if official.get("space_id") != submission["space_id"]:
        raise ValueError("space")
    if official.get("sha") != submission["space_sha"]:
        raise ValueError("sha")
    judged_at = _parsed_timestamp(official.get("judged_at"), "judged_at")
    submitted_at = _parsed_timestamp(
        submission["observed_at"], "submission"
    )
    fetched_at = _parsed_timestamp(snapshot["fetched_at"], "snapshot")
    if judged_at > fetched_at or fetched_at > now:
        raise ValueError("judged_at")
    if submission.get("queue_status") != "judged" and judged_at <= submitted_at:
        # A pending-queue submission must be judged after it was observed;
        # a judged-queue submission imports a verdict that already existed,
        # bound to the exact deployed SHA by the equality checks above.
        raise ValueError("judged_at")
    normalized = _normalize_official_claims(attempt, official)

    judgment_path = paths.judgment(attempt_id)
    judgment = store.read_json(judgment_path)
    scheduler.validate_judgment_record(judgment)
    if (
        judgment["attempt_id"] != attempt_id
        or judgment["paper_id"] != attempt["paper_id"]
        or judgment["space_id"] != submission["space_id"]
        or judgment["submitted_sha"] != submission["space_sha"]
        or judgment["attempt_number"] != authority["attempt_number"]
        or judgment["raw_verdict"] is not None
    ):
        raise ValueError("judgment")
    latest_event = (
        _parsed_timestamp(judgment["polls"][-1]["at"], "polls")
        if judgment["polls"]
        else _parsed_timestamp(judgment["created_at"], "created_at")
    )
    if now < latest_event:
        raise ValueError("now")

    payload = {
        "snapshot_id": snapshot_id,
        "verdict_revision": verdict_revision,
        "submission_attestation_id": submission["attestation_id"],
        "authority_attestation_id": authority["attestation_id"],
        "space_id": submission["space_id"],
        "space_sha": submission["space_sha"],
        "paper_id": attempt["paper_id"],
        "judged_at": official["judged_at"],
        "claims": normalized["claims"],
    }
    record = attestations.prepare({
        "kind": "verdict",
        "attempt_id": attempt_id,
        "attempt_number": authority["attempt_number"],
        "observed_at": _aware_timestamp(now),
        "source_commit": submission["source_commit"],
        "payload_sha256": _sha256_json(payload),
        **payload,
    })
    attestation_id = record["attestation_id"]
    finalized = copy.deepcopy(judgment)
    finalized["raw_verdict"] = copy.deepcopy(official)
    finalized["normalized_verdict"] = copy.deepcopy(normalized)
    finalized["source_revision"] = verdict_revision
    finalized["verdict_at"] = _aware_timestamp(now)
    finalized["updated_at"] = finalized["verdict_at"]
    scheduler.validate_judgment_record(finalized)
    if improvement_reason is None:
        phase = "complete"
        transition_updates = {
            "verdict": normalized,
            "verdict_source_revision": verdict_revision,
            "verdict_at": finalized["verdict_at"],
        }
    else:
        if type(improvement_reason) is not str or not improvement_reason.strip():
            raise ValueError("improvement_reason")
        phase = "improving"
        improvement_attempt = attempt.get("improvement_attempts", 0) + 1
        transition_updates = {
            "improvement_attempts": improvement_attempt,
            "improvement_reason": improvement_reason,
            "verdicts": [
                *copy.deepcopy(attempt.get("verdicts", [])),
                {
                    **copy.deepcopy(normalized),
                    "improvement_attempt": improvement_attempt,
                    "improvement_reason": improvement_reason,
                },
            ],
        }
    return attempts.transition_attested(
        paths,
        attempt_id,
        phase,
        attestation_id,
        transition_updates,
        lease,
        now,
        transaction_targets=[
            (
                judgment_path,
                finalized,
                scheduler.validate_judgment_record,
            )
        ],
        attestation_record=record,
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
    registered_project_path = attempt.get("project_path")
    if registered_project_path is None:
        slug = _nonempty_string(attempt.get("slug"), "slug")
        registered_project_path = f"submissions/{slug}"
    if registered_project_path != project_path:
        raise ValueError("project_path")
    _validated_project_source(worktree, project_path, "project_path")
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
    paper_pytest_arguments = _paper_pytest_arguments(
        paper_pytest, project_path
    )
    if paper_pytest_arguments is None or any(
        _pytest_bypass(argument)
        for argument in paper_pytest_arguments
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


def _paper_pytest_arguments(
    command: tuple[str, ...], project_path: str
) -> tuple[str, ...] | None:
    expected = (
        "uv",
        "run",
        "--project",
        project_path,
        "python",
        "-m",
        "pytest",
        f"{project_path}/tests",
        "-q",
    )
    if command != expected:
        return None
    return command[7:]


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


def _project_status_command(project_path: str) -> tuple[str, ...]:
    return (
        "git",
        "status",
        "--porcelain",
        "--ignored",
        "--untracked-files=all",
        "--",
        project_path,
    )


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


def _source_tree_sha256(source_dir: Path) -> str:
    """Hash the relative paths and bytes in one validated upload tree."""
    source_dir = _resolved_directory(source_dir, "source_dir")
    entries = []
    for path in sorted(source_dir.rglob("*")):
        if path.is_symlink():
            raise ValueError("source_dir")
        if path.is_file():
            entries.append(
                {
                    "path": path.relative_to(source_dir).as_posix(),
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
            )
    return _sha256_json(entries)


def _resolved_directory(path: Path, field: str) -> Path:
    try:
        resolved = path.resolve(strict=True)
    except OSError as error:
        raise ValueError(field) from error
    if path != resolved or not resolved.is_dir():
        raise ValueError(field)
    return resolved


def _validated_project_source(
    worktree: Path,
    project_path: str,
    field: str,
) -> Path:
    source = _resolved_directory(worktree / project_path, field)
    if not source.is_relative_to(worktree):
        raise ValueError(field)
    return source


def _authoritative_attestation(
    paths: store.StatePaths,
    attempt: dict,
    kind: str,
    phase: str,
) -> dict:
    attempt_number = attempt.get("improvement_attempts", 0) + 1
    path = paths.attestation(kind, attempt["attempt_id"], attempt_number)
    if not path.exists():
        raise ValueError(kind)
    record = store.read_json(path)
    attestations.validate_target(paths, path, record)
    transitions = [
        transition
        for transition in attempt.get("transitions", [])
        if transition.get("to") == phase
    ]
    if (
        record.get("kind") != kind
        or record.get("attempt_id") != attempt["attempt_id"]
        or record.get("attempt_number") != attempt_number
        or not transitions
        or transitions[-1].get("attestation_id")
        != record.get("attestation_id")
    ):
        raise ValueError(kind)
    return record


def _require_current_validated_tree(
    worktree: Path, source_dir: Path, validation: dict
) -> None:
    if _git_output(worktree, "status", "--porcelain"):
        raise ValueError("source tree")
    project_status = _project_status_command(validation["project_path"])
    if _git_output(worktree, *project_status[1:]):
        raise ValueError("source tree")
    if _git_output(worktree, "branch", "--show-current") != validation["branch"]:
        raise ValueError("source tree")
    if _git_output(worktree, "rev-parse", "HEAD") != validation["source_commit"]:
        raise ValueError("source commit")
    if (
        _git_output(worktree, "rev-parse", "HEAD^{tree}")
        != validation["source_tree"]
    ):
        raise ValueError("source tree")
    if _source_tree_sha256(source_dir) != validation["source_tree_sha256"]:
        raise ValueError("source tree")


def _git_output(worktree: Path, *arguments: str) -> str:
    result = run_command(("git", *arguments), worktree)
    if result.returncode != 0:
        raise ValueError("source tree")
    return result.stdout.strip()


def _hub_attribute(value: object, field: str):
    result = value.get(field) if isinstance(value, dict) else getattr(
        value, field, None
    )
    if result is None:
        raise ValueError(field)
    return result


def _parsed_timestamp(value: object, field: str) -> datetime:
    value = _nonempty_string(value, field)
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError(field) from error
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(field)
    return parsed


def _snapshot_verdict_revision(snapshot: dict) -> str:
    sources = snapshot.get("sources")
    if type(sources) is not dict:
        raise ValueError("verdict revision")
    verdicts = sources.get("verdicts")
    if type(verdicts) is not dict:
        raise ValueError("verdict revision")
    return _nonempty_string(verdicts.get("revision"), "verdict revision")


def _exact_official_verdict(
    snapshot: dict, paper_id: str, space_id: str
) -> dict:
    same_space = [
        verdict
        for verdict in snapshot["verdicts"]
        if verdict.get("space_id") == space_id
    ]
    if same_space and not any(
        verdict.get("paper_id") == paper_id for verdict in same_space
    ):
        raise ValueError("paper")
    same_paper = [
        verdict
        for verdict in snapshot["verdicts"]
        if verdict.get("paper_id") == paper_id
    ]
    if same_paper and not any(
        verdict.get("space_id") == space_id for verdict in same_paper
    ):
        raise ValueError("space")
    exact = [
        verdict
        for verdict in same_space
        if verdict.get("paper_id") == paper_id
    ]
    if len(exact) != 1:
        raise ValueError("official_verdict")
    return copy.deepcopy(exact[0])


def _normalize_official_claims(attempt: dict, official: dict) -> dict:
    target_claims = attempt.get("target_claims")
    bindings = attempt.get("claim_bindings")
    official_claims = official.get("claims")
    if (
        type(target_claims) is not list
        or type(bindings) is not list
        or len(bindings) != len(target_claims)
        or type(official_claims) is not list
    ):
        raise ValueError("claim")
    claims_by_text = {}
    for claim in official_claims:
        if (
            type(claim) is not dict
            or set(claim) != {"claim", "verdict", "evidence"}
            or type(claim.get("claim")) is not str
            or not claim["claim"]
            or claim["claim"] in claims_by_text
        ):
            raise ValueError("claim")
        claims_by_text[claim["claim"]] = claim
    normalized = []
    for target_claim, binding in zip(target_claims, bindings, strict=True):
        if (
            type(binding) is not dict
            or set(binding)
            != {
                "target_claim",
                "challenge_claim",
                "challenge_claim_sha256",
            }
            or binding["target_claim"] != target_claim
            or type(binding["challenge_claim"]) is not str
            or not binding["challenge_claim"]
            or binding["challenge_claim_sha256"]
            != hashlib.sha256(
                binding["challenge_claim"].encode("utf-8")
            ).hexdigest()
        ):
            raise ValueError("claim")
        claim = claims_by_text.get(binding["challenge_claim"])
        if claim is None:
            raise ValueError("claim")
        status = claim["verdict"]
        evidence = claim["evidence"]
        if (
            status not in scheduler.OFFICIAL_VERDICT_STATUSES
            or type(evidence) is not str
        ):
            raise ValueError("verdict")
        normalized.append(
            {
                "target_claim": target_claim,
                "claim": binding["challenge_claim"],
                "status": status,
                "evidence": evidence,
            }
        )
    return {"claims": normalized}
