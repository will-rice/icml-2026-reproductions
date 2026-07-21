"""Command-line generation of the portable NAPE reproduction bundle."""

import errno
import json
import logging
import os
import platform
import secrets
import stat
from pathlib import Path
from tempfile import TemporaryDirectory

from icml_2026_repro.audit import (
    DATASET_REPOSITORY,
    DATASET_REVISION,
    GITHUB_REPOSITORY,
    GITHUB_REVISION,
)
from icml_2026_repro.benchmark_evidence import build_benchmark_evidence
from icml_2026_repro.future_adaptation_evidence import build_future_adaptation_evidence
from icml_2026_repro.predictability_evidence import build_predictability_evidence

BUNDLE_ARTIFACT_NAMES = (
    "claim_1_benchmark.json",
    "claim_2_predictability.json",
    "claim_3_future_adaptation.json",
    "claims_4_6_status.json",
    "environment.json",
    "README.md",
)
LEGACY_ARTIFACT_NAMES = (
    "claim_1_audit.json",
    "claim_2_trace.json",
)
STAGING_FILE_PREFIX = ".nape-repro-staging-"
STAGING_CREATE_ATTEMPTS = 10
DARWIN_ROOT_ALIASES = {
    "tmp": ("private/tmp", ("private", "tmp")),
    "var": ("private/var", ("private", "var")),
}


def main() -> None:
    """Generate the default reproduction bundle."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    bundle_path = build_bundle()
    logging.info("Wrote reproduction bundle to %s", bundle_path)


def build_bundle(output_dir: Path = Path("repro_bundle")) -> Path:
    """Build portable JSON evidence and operator documentation."""
    output_directory_fd = _open_output_directory(output_dir)
    try:
        legacy_artifact_names = _validate_artifact_destinations(output_directory_fd)
        claim_1 = build_benchmark_evidence()
        claim_2 = build_predictability_evidence()
        with TemporaryDirectory(prefix="nape-repro-internal-") as recorder_directory:
            claim_3 = build_future_adaptation_evidence(Path(recorder_directory))

        _write_json(output_directory_fd, "claim_1_benchmark.json", claim_1)
        _write_json(output_directory_fd, "claim_2_predictability.json", claim_2)
        _write_json(output_directory_fd, "claim_3_future_adaptation.json", claim_3)
        _write_json(output_directory_fd, "claims_4_6_status.json", _unreplicated_model_claims())
        _write_json(output_directory_fd, "environment.json", _environment_manifest())
        _write_artifact(output_directory_fd, "README.md", _bundle_readme())
        _remove_legacy_artifacts(output_directory_fd, legacy_artifact_names)
    finally:
        os.close(output_directory_fd)
    return output_dir


def _open_output_directory(output_dir: Path) -> int:
    """Open or create ``output_dir`` without following any symlink component."""
    directory_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
    anchor, components = _output_directory_location(output_dir)
    opened_fds: list[int] = []
    try:
        try:
            current_fd = os.open(anchor, directory_flags)
        except OSError as error:
            raise RuntimeError(
                f"could not open output directory anchor: {error.strerror or error}"
            ) from error
        opened_fds.append(current_fd)

        for component in components:
            if component in ("", "."):
                continue
            next_fd = _open_output_directory_component(
                current_fd,
                component,
                directory_flags,
            )
            opened_fds.append(next_fd)
            current_fd = next_fd

        return opened_fds.pop()
    finally:
        for opened_fd in reversed(opened_fds):
            os.close(opened_fd)


def _output_directory_location(output_dir: Path) -> tuple[str, tuple[str, ...]]:
    """Normalize only verified root-owned macOS aliases, never caller path symlinks."""
    anchor = output_dir.anchor or "."
    components = output_dir.parts[1:] if output_dir.anchor else output_dir.parts
    if anchor != "/" or platform.system() != "Darwin" or not components:
        return anchor, components

    trusted_components = _trusted_darwin_root_alias(components[0])
    if trusted_components is None:
        return anchor, components
    return anchor, (*trusted_components, *components[1:])


def _trusted_darwin_root_alias(component: str) -> tuple[str, ...] | None:
    """Return an exact trusted Darwin alias target after ownership and mode checks."""
    alias = DARWIN_ROOT_ALIASES.get(component)
    if alias is None:
        return None
    expected_link, target_components = alias
    alias_path = Path("/") / component
    try:
        alias_stat = alias_path.lstat()
        alias_parent_stat = alias_path.parent.stat(follow_symlinks=False)
        target_parent_stat = Path("/private").stat(follow_symlinks=False)
        link_target = str(alias_path.readlink())
    except OSError:
        return None

    untrusted_write_bits = stat.S_IWGRP | stat.S_IWOTH
    if (
        not stat.S_ISLNK(alias_stat.st_mode)
        or alias_stat.st_uid != 0
        or alias_stat.st_gid != 0
        or stat.S_IMODE(alias_stat.st_mode) & untrusted_write_bits
        or link_target != expected_link
    ):
        return None
    for parent_stat in (alias_parent_stat, target_parent_stat):
        if (
            not stat.S_ISDIR(parent_stat.st_mode)
            or parent_stat.st_uid != 0
            or parent_stat.st_gid != 0
            or stat.S_IMODE(parent_stat.st_mode) & untrusted_write_bits
        ):
            return None
    return target_components


def _open_output_directory_component(
    parent_fd: int,
    component: str,
    directory_flags: int,
) -> int:
    """Open or create one no-follow output directory component."""
    try:
        return os.open(component, directory_flags, dir_fd=parent_fd)
    except FileNotFoundError:
        try:
            os.mkdir(component, dir_fd=parent_fd)
        except FileExistsError:
            pass
        except OSError as error:
            raise RuntimeError(
                f"could not create output directory component {component!r}: "
                f"{error.strerror or error}"
            ) from error
    except OSError as error:
        if error.errno in (errno.ELOOP, errno.ENOTDIR):
            raise RuntimeError(
                "output directory must not be a symlink and must contain only "
                f"directories: {component!r}"
            ) from error
        raise RuntimeError(
            f"could not open output directory component {component!r}: {error.strerror or error}"
        ) from error

    try:
        return os.open(component, directory_flags, dir_fd=parent_fd)
    except OSError as error:
        if error.errno in (errno.ELOOP, errno.ENOTDIR):
            raise RuntimeError(
                "output directory must not be a symlink and must contain only "
                f"directories: {component!r}"
            ) from error
        raise RuntimeError(
            f"could not open output directory component {component!r}: {error.strerror or error}"
        ) from error


def _validate_artifact_destinations(output_directory_fd: int) -> tuple[str, ...]:
    """Validate the complete reused-directory inventory before generation."""
    try:
        directory_entries = set(os.listdir(output_directory_fd))
    except OSError as error:
        raise RuntimeError(
            f"could not inspect output directory: {error.strerror or error}"
        ) from error

    expected_entries = set(BUNDLE_ARTIFACT_NAMES) | set(LEGACY_ARTIFACT_NAMES)
    unexpected_entries = sorted(directory_entries - expected_entries)
    if unexpected_entries:
        raise RuntimeError(f"unexpected output directory entry: {unexpected_entries[0]}")

    legacy_artifact_names: list[str] = []
    for artifact_name in sorted(directory_entries):
        try:
            artifact_stat = os.stat(
                artifact_name,
                dir_fd=output_directory_fd,
                follow_symlinks=False,
            )
        except FileNotFoundError:
            continue
        except OSError as error:
            raise RuntimeError(
                f"could not inspect {artifact_name}: {error.strerror or error}"
            ) from error
        if not stat.S_ISREG(artifact_stat.st_mode):
            raise RuntimeError(f"{artifact_name} must be a regular file")
        if artifact_name in LEGACY_ARTIFACT_NAMES:
            legacy_artifact_names.append(artifact_name)
    return tuple(legacy_artifact_names)


def _remove_legacy_artifacts(
    output_directory_fd: int, legacy_artifact_names: tuple[str, ...]
) -> None:
    """Remove validated legacy files after all replacement artifacts succeed."""
    for artifact_name in legacy_artifact_names:
        try:
            artifact_stat = os.stat(
                artifact_name,
                dir_fd=output_directory_fd,
                follow_symlinks=False,
            )
        except FileNotFoundError:
            continue
        except OSError as error:
            raise RuntimeError(
                f"could not inspect {artifact_name}: {error.strerror or error}"
            ) from error
        if not stat.S_ISREG(artifact_stat.st_mode):
            raise RuntimeError(f"{artifact_name} must be a regular file")

    for artifact_name in legacy_artifact_names:
        try:
            os.unlink(artifact_name, dir_fd=output_directory_fd)
        except FileNotFoundError:
            continue
        except OSError as error:
            raise RuntimeError(
                f"could not remove legacy artifact {artifact_name}: {error.strerror or error}"
            ) from error


def _write_json(output_directory_fd: int, artifact_name: str, value: object) -> None:
    """Write sorted, indented JSON with a stable trailing newline."""
    _write_artifact(
        output_directory_fd,
        artifact_name,
        json.dumps(value, indent=2, sort_keys=True) + "\n",
    )


def _write_artifact(output_directory_fd: int, artifact_name: str, contents: str) -> None:
    """Atomically replace an artifact without opening its existing inode."""
    staging_name, staging_fd = _create_staging_file(output_directory_fd, artifact_name)
    try:
        try:
            remaining = memoryview(contents.encode("utf-8"))
            while remaining:
                try:
                    bytes_written = os.write(staging_fd, remaining)
                except OSError as error:
                    raise RuntimeError(
                        f"could not write {artifact_name}: {error.strerror or error}"
                    ) from error
                if bytes_written == 0:
                    raise RuntimeError(f"could not write {artifact_name}: zero-byte write")
                remaining = remaining[bytes_written:]
        finally:
            try:
                os.close(staging_fd)
            except OSError as error:
                raise RuntimeError(
                    f"could not close staging file for {artifact_name}: {error.strerror or error}"
                ) from error

        try:
            os.replace(
                staging_name,
                artifact_name,
                src_dir_fd=output_directory_fd,
                dst_dir_fd=output_directory_fd,
            )
        except OSError as error:
            raise RuntimeError(
                f"could not replace {artifact_name}: {error.strerror or error}"
            ) from error
    except BaseException:
        try:
            os.unlink(staging_name, dir_fd=output_directory_fd)
        except FileNotFoundError:
            pass
        except OSError as error:
            raise RuntimeError(
                f"could not remove staging file for {artifact_name}: {error.strerror or error}"
            ) from error
        raise


def _create_staging_file(output_directory_fd: int, artifact_name: str) -> tuple[str, int]:
    """Create a collision-resistant private staging file with bounded retries."""
    staging_flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC
    for _ in range(STAGING_CREATE_ATTEMPTS):
        staging_name = f"{STAGING_FILE_PREFIX}{secrets.token_hex(16)}"
        try:
            staging_fd = os.open(
                staging_name,
                staging_flags,
                0o644,
                dir_fd=output_directory_fd,
            )
        except FileExistsError:
            continue
        except OSError as error:
            raise RuntimeError(
                f"could not create staging file for {artifact_name}: {error.strerror or error}"
            ) from error
        return staging_name, staging_fd
    raise RuntimeError(
        f"could not create staging file for {artifact_name}: "
        f"{STAGING_CREATE_ATTEMPTS} name collisions"
    )


def _environment_manifest() -> dict[str, object]:
    """Return portable runtime, source, and rerun metadata."""
    return {
        "commands": [
            "uv sync",
            "uv run nape-repro",
            "uv run pytest tests/test_cli.py -q",
            (
                "uv run pytest tests/test_benchmark_evidence.py "
                "tests/test_predictability_evidence.py "
                "tests/test_future_adaptation_evidence.py -q"
            ),
            "uv run pytest -q",
            "uv run pytest external/NAPE/tests -q",
            "uv run pre-commit run -a",
            "uv run ty check src tests",
            "git diff --check",
        ],
        "pinned_revisions": {
            "dataset": {
                "repository": DATASET_REPOSITORY,
                "revision": DATASET_REVISION,
            },
            "github": {
                "repository": GITHUB_REPOSITORY,
                "revision": GITHUB_REVISION,
            },
        },
        "platform": {
            "machine": platform.machine(),
            "release": platform.release(),
            "system": platform.system(),
        },
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "schema": "icml-2026-repro-bundle/v2",
    }


def _unreplicated_model_claims() -> dict[str, dict[str, str]]:
    """Return explicit status for claims requiring unavailable model outputs."""
    return {
        "claim_4": {
            "status": "not replicated",
            "reason": "Named model outputs and paid API budget were not available.",
        },
        "claim_5": {
            "status": "not replicated",
            "reason": "Named model outputs and paid API budget were not available.",
        },
        "claim_6": {
            "status": "not replicated",
            "reason": "Named model outputs and paid API budget were not available.",
        },
    }


def _bundle_readme() -> str:
    """Return the portable bundle artifact guide."""
    return """# NAPE Reproduction Evidence

This bundle is generated by `uv run nape-repro`.

- `claim_1_benchmark.json` recomputes benchmark counts, sequence-length
  statistics, and construction-artifact coverage from the pinned NAPE release.
- `claim_2_predictability.json` audits released oracle outputs and recomputes
  their aggregate predictability coverage without rerunning paid oracle calls.
- `claim_3_future_adaptation.json` exercises one deterministic official-evaluator
  adaptation case per each of the 52 released trajectories and one residual fixture.
- `claims_4_6_status.json` records why Claims 4-6 were not replicated.
- `environment.json` records the Python/platform metadata, pinned revisions, and
  exact commands used to rerun the evidence.

Claim 1 reproduces 52 trajectories and 11,907 operations, sequence lengths of
35-821 operations, a paper-rounded mean of 229, and a median of 164. One
trajectory is one released JSON file and one operation is one string in its
`operations` array.

Claim 2 recomputes 68.04% weighted property coverage from released oracle
outputs. The original paid frontier-model oracle calls were not rerun.

Claim 3 observes removal in 50/52 executed release cases, inverse insertion in
52/52 cases, and target preservation in 52/52 cases. Each release case is one
deterministic adaptation after the first operation, not a full per-action model
rollout. The release sweep did not exercise residual correction, so one residual
fixture separately demonstrates that mechanism while preserving the target state.

Claims 4-6 were not replicated because named model outputs and paid API budget
were not available.

This reproduction makes zero paid API calls. The canonical Space is:
https://huggingface.co/spaces/wrice/repro-a-benchmark-and-framework-for-evaluating-next-action-predictions-in-spreadsheets
"""
