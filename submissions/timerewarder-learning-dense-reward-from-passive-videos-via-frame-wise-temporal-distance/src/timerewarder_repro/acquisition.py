"""Acquire selected upstream files as verified inert bytes."""

import hashlib
import json
import os
import re
import subprocess
import tempfile
from pathlib import Path, PurePosixPath


GIT_REVISION = re.compile(r"[0-9a-f]{40}")
PAPER_REVISION = re.compile(r"arxiv:[0-9]{4}\.[0-9]{5}v[1-9][0-9]*")


def acquire_inert_sources(
    manifest_path: Path, output_root: Path, receipt_path: Path
) -> dict[str, object]:
    """Extract manifest-selected Git blobs without executing upstream code."""
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    validate_manifest(manifest)
    output_paths = resolved_source_paths(output_root, manifest)
    if output_root.exists() and (
        not output_root.is_dir() or any(output_root.iterdir())
    ):
        raise ValueError("acquisition requires an empty output root")
    commands: list[dict[str, object]] = []
    observed_sources: list[dict[str, object]] = []
    repositories = grouped_sources(manifest)

    output_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="timerewarder-acquisition-") as temporary:
        for index, ((url, revision), sources) in enumerate(repositories.items()):
            checkout = Path(temporary) / f"repository-{index}"
            run_git(
                ["git", "clone", "--no-checkout", url, str(checkout)],
                commands,
                checkout,
            )
            run_git(
                ["git", "-C", str(checkout), "checkout", "--detach", revision],
                commands,
                checkout,
            )
            head = (
                run_git(
                    ["git", "-C", str(checkout), "rev-parse", "HEAD"],
                    commands,
                    checkout,
                )
                .stdout.decode()
                .strip()
            )
            if head != revision:
                raise ValueError(
                    "checked-out HEAD does not match full immutable revision"
                )
            status = run_git(
                ["git", "-C", str(checkout), "status", "--porcelain"],
                commands,
                checkout,
            ).stdout
            if status:
                raise ValueError("checkout is not clean")
            run_git(["git", "-C", str(checkout), "fsck", "--full"], commands, checkout)

            for expected in sources:
                upstream_path = expected.get("upstream_path", expected["path"])
                blob = (
                    run_git(
                        [
                            "git",
                            "-C",
                            str(checkout),
                            "rev-parse",
                            f"{revision}:{upstream_path}",
                        ],
                        commands,
                        checkout,
                    )
                    .stdout.decode()
                    .strip()
                )
                if blob != expected["git_blob"]:
                    raise ValueError(f"git_blob mismatch: {expected['path']}")
                payload = run_git(
                    [
                        "git",
                        "-C",
                        str(checkout),
                        "show",
                        f"{revision}:{upstream_path}",
                    ],
                    commands,
                    checkout,
                ).stdout
                observed = expected | {
                    "sha256": hashlib.sha256(payload).hexdigest(),
                    "byte_size": len(payload),
                }
                for field in ("sha256", "byte_size"):
                    if observed[field] != expected[field]:
                        raise ValueError(f"{field} mismatch: {expected['path']}")
                write_atomic(output_paths[expected["path"]], payload)
                observed_sources.append(observed)

    receipt: dict[str, object] = {
        "commands": commands,
        "sources": observed_sources,
    }
    write_atomic(
        receipt_path,
        (json.dumps(receipt, indent=2, sort_keys=True) + "\n").encode("utf-8"),
    )
    return receipt


def verify_acquisition(
    manifest_path: Path, receipt_path: Path, source_root: Path
) -> tuple[dict[str, object], ...]:
    """Verify source bytes against both manifest and acquisition receipt."""
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    validate_manifest(manifest)
    source_paths = resolved_source_paths(source_root, manifest)
    if not receipt_path.is_file():
        raise ValueError("acquisition receipt is required")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if set(receipt) != {"commands", "sources"}:
        raise ValueError("acquisition command receipt schema mismatch")
    if not isinstance(receipt["commands"], list) or any(
        not isinstance(item, dict)
        or set(item) != {"command", "status"}
        or not isinstance(item["command"], list)
        or not all(isinstance(argument, str) for argument in item["command"])
        or type(item["status"]) is not int
        for item in receipt["commands"]
    ):
        raise ValueError("acquisition command receipt schema mismatch")
    if receipt["commands"] != expected_command_receipts(manifest):
        raise ValueError("acquisition command receipt mismatch")
    expected_paths = [source["path"] for source in manifest["sources"]]
    observed_paths = [source.get("path") for source in receipt["sources"]]
    if observed_paths != expected_paths:
        raise ValueError("receipt source paths mismatch")
    if not source_root.is_dir():
        raise ValueError("source inventory mismatch")
    inventory_files = set()
    inventory_directories = set()
    for path in source_root.rglob("*"):
        relative = path.relative_to(source_root).as_posix()
        if path.is_symlink():
            raise ValueError("source inventory mismatch")
        if path.is_file():
            inventory_files.add(relative)
        elif path.is_dir():
            inventory_directories.add(relative)
        else:
            raise ValueError("source inventory mismatch")
    expected_directories = {
        parent.as_posix()
        for expected_path in expected_paths
        for parent in PurePosixPath(expected_path).parents
        if parent.as_posix() != "."
    }
    if (
        inventory_files != set(expected_paths)
        or inventory_directories != expected_directories
    ):
        raise ValueError("source inventory mismatch")
    observed_by_path = {item["path"]: item for item in receipt["sources"]}
    verified = []
    for expected in manifest["sources"]:
        observed = observed_by_path.get(expected["path"])
        if observed is None:
            raise ValueError(f"source identity mismatch: {expected['path']}")
        path = source_paths[expected["path"]]
        if path.stat().st_mode & 0o111:
            raise ValueError("source inventory mismatch")
        payload = path.read_bytes()
        actual = expected | {
            "sha256": hashlib.sha256(payload).hexdigest(),
            "byte_size": len(payload),
        }
        for field in expected:
            if observed.get(field) != actual[field]:
                raise ValueError(f"{field} mismatch: {expected['path']}")
        if set(observed) != set(actual):
            raise ValueError(f"source identity mismatch: {expected['path']}")
        verified.append(observed)
    return tuple(verified)


def validate_manifest(manifest: dict[str, object]) -> None:
    """Require immutable revisions for every upstream trust domain."""
    for section in ("model", "dataset"):
        revision = manifest[section]["revision"]
        if not GIT_REVISION.fullmatch(revision):
            raise ValueError("full immutable revision is required")
    if not PAPER_REVISION.fullmatch(manifest["paper"]["revision"]):
        raise ValueError("full immutable revision is required")
    for source in manifest["sources"]:
        if not GIT_REVISION.fullmatch(source["revision"]):
            raise ValueError("full immutable revision is required")
    validate_source_paths(manifest)


def validate_source_paths(manifest: dict[str, object]) -> None:
    """Require unique normalized relative POSIX file paths without collisions."""
    paths = []
    for source in manifest["sources"]:
        path = source["path"]
        if (
            not isinstance(path, str)
            or not path
            or "\\" in path
            or PurePosixPath(path).is_absolute()
            or any(part in {"", ".", ".."} for part in path.split("/"))
            or PurePosixPath(path).as_posix() != path
        ):
            raise ValueError("manifest source path must be normalized relative POSIX")
        paths.append(tuple(path.split("/")))
    for index, path in enumerate(paths):
        if any(
            index != other_index and other[: len(path)] == path
            for other_index, other in enumerate(paths)
        ):
            raise ValueError("manifest source path collision")


def resolved_source_paths(root: Path, manifest: dict[str, object]) -> dict[str, Path]:
    """Resolve source destinations and require containment beneath root."""
    resolved_root = root.resolve()
    resolved = {}
    for source in manifest["sources"]:
        path = (root / source["path"]).resolve()
        if not path.is_relative_to(resolved_root):
            raise ValueError("manifest source path escapes output root")
        resolved[source["path"]] = path
    return resolved


def grouped_sources(
    manifest: dict[str, object],
) -> dict[tuple[str, str], list[dict[str, object]]]:
    """Group selected files by immutable repository checkout."""
    repositories: dict[tuple[str, str], list[dict[str, object]]] = {}
    for source in manifest["sources"]:
        repositories.setdefault((source["url"], source["revision"]), []).append(source)
    return repositories


def expected_command_receipts(manifest: dict[str, object]) -> list[dict[str, object]]:
    """Build the only accepted successful Git command receipt sequence."""
    receipts = []
    checkout = "<checkout>"
    for (url, revision), sources in grouped_sources(manifest).items():
        commands = [
            ["git", "clone", "--no-checkout", url, checkout],
            ["git", "-C", checkout, "checkout", "--detach", revision],
            ["git", "-C", checkout, "rev-parse", "HEAD"],
            ["git", "-C", checkout, "status", "--porcelain"],
            ["git", "-C", checkout, "fsck", "--full"],
        ]
        for source in sources:
            upstream_path = source.get("upstream_path", source["path"])
            commands.extend(
                [
                    ["git", "-C", checkout, "rev-parse", f"{revision}:{upstream_path}"],
                    ["git", "-C", checkout, "show", f"{revision}:{upstream_path}"],
                ]
            )
        receipts.extend({"command": command, "status": 0} for command in commands)
    return receipts


def run_git(
    command: list[str], commands: list[dict[str, object]], checkout: Path
) -> subprocess.CompletedProcess[bytes]:
    """Run and record one non-interactive Git command."""
    result = subprocess.run(command, check=True, capture_output=True)
    recorded = ["<checkout>" if item == str(checkout) else item for item in command]
    commands.append({"command": recorded, "status": result.returncode})
    return result


def write_atomic(path: Path, payload: bytes) -> None:
    """Atomically replace a file with payload."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise
