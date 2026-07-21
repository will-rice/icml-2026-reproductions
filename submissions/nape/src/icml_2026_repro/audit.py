"""Audit released trajectory files and linked dataset evidence."""

import json
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

from huggingface_hub import hf_hub_download

GITHUB_REPOSITORY = "Tej-55/NAPE"
DATASET_REPOSITORY = "Tej-a55/napeval"
GITHUB_REVISION = "ac0d10e4dc345f982a5665a2c4bdb6b752d663f2"
DATASET_REVISION = "c7e28fe9b08ee2c0bfc429519cf100197b7e018c"
DATASET_FILENAME = "data/test.jsonl"
CLAIMED_TRAJECTORIES = 58
CLAIMED_ACTIONS = 13_000
PAPER_REPORTED_TRAJECTORIES = 52
PAPER_REPORTED_ACTIONS = 11_907
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class ReleaseCounts:
    """Counts of trajectories and actions in a released source revision."""

    source: str
    revision: str
    trajectories: int
    actions: int


def audit_trajectory_directory(path: Path, revision: str) -> ReleaseCounts:
    """Count trajectories and operations in JSON files under ``path``."""
    trajectory_paths = sorted(path.glob("*.json"))
    action_count = 0
    for trajectory_path in trajectory_paths:
        trajectory = json.loads(trajectory_path.read_text(encoding="utf-8"))
        operations = trajectory.get("operations")
        if not isinstance(operations, list) or not all(
            isinstance(operation, str) for operation in operations
        ):
            raise ValueError(f"{trajectory_path.name}: operations must be a list of strings")
        action_count += len(operations)
    return ReleaseCounts(
        source=f"github:{GITHUB_REPOSITORY}",
        revision=revision,
        trajectories=len(trajectory_paths),
        actions=action_count,
    )


def audit_jsonl(path: Path, revision: str) -> ReleaseCounts:
    """Count rows and operations in a JSONL dataset release."""
    action_count = 0
    trajectory_count = 0
    with path.open(encoding="utf-8") as dataset_file:
        for line_number, line in enumerate(dataset_file, start=1):
            if not line.strip():
                raise ValueError(f"{path.name}:{line_number}: blank JSONL row")
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path.name}:{line_number}: invalid JSON: {error.msg}") from error
            if not isinstance(row, dict):
                raise ValueError(f"{path.name}:{line_number}: row must be a JSON object")
            name = row.get("name")
            operations = row.get("operations")
            num_operations = row.get("num_operations")
            if not isinstance(name, str):
                raise ValueError(f"{path.name}:{line_number}: name must be a string")
            if not isinstance(operations, list) or not all(
                isinstance(operation, str) for operation in operations
            ):
                raise ValueError(f"{path.name}:{line_number}: operations must be a list of strings")
            if isinstance(num_operations, bool) or not isinstance(num_operations, int):
                raise ValueError(f"{path.name}:{line_number}: num_operations must be an integer")
            if num_operations != len(operations):
                raise ValueError(
                    f"{path.name}:{line_number}: num_operations does not match operations"
                )
            trajectory_count += 1
            action_count += len(operations)
    return ReleaseCounts(
        source=f"dataset:{DATASET_REPOSITORY}",
        revision=revision,
        trajectories=trajectory_count,
        actions=action_count,
    )


def read_git_head(repository_path: Path) -> str:
    """Read a checkout's exact Git HEAD or raise an explicit error."""
    try:
        result = subprocess.run(
            ["git", "-C", str(repository_path), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise RuntimeError(f"could not read Git HEAD for {repository_path}: {error}") from error
    return result.stdout.strip()


def read_git_worktree_status(repository_path: Path) -> str:
    """Read all tracked, staged, and untracked changes in a Git worktree."""
    try:
        result = subprocess.run(
            [
                "git",
                "-C",
                str(repository_path),
                "status",
                "--porcelain=v1",
                "--untracked-files=all",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise RuntimeError(f"could not read Git status for {repository_path}: {error}") from error
    return result.stdout.strip()


def read_git_trajectory_status(repository_path: Path) -> str:
    """Read tracked and untracked changes under the trajectory directory."""
    try:
        result = subprocess.run(
            [
                "git",
                "-C",
                str(repository_path),
                "status",
                "--porcelain=v1",
                "--untracked-files=all",
                "--",
                "data/trajectories",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise RuntimeError(
            f"could not read Git status for {repository_path}/data/trajectories: {error}"
        ) from error
    return result.stdout.strip()


def build_challenge_card_claim_1_audit() -> dict[str, object]:
    """Audit the legacy challenge-card Claim 1 wording against two releases."""
    nape_path = REPOSITORY_ROOT / "external" / "NAPE"
    github_head = read_git_head(nape_path)
    if github_head != GITHUB_REVISION:
        raise ValueError(
            f"NAPE checkout is at {github_head}, expected pinned revision {GITHUB_REVISION}"
        )
    trajectory_status = read_git_trajectory_status(nape_path)
    if trajectory_status:
        changed_paths = ", ".join(trajectory_status.splitlines())
        raise ValueError(f"NAPE trajectory contents are dirty: {changed_paths}")

    github_counts = audit_trajectory_directory(
        nape_path / "data" / "trajectories", revision=GITHUB_REVISION
    )
    try:
        dataset_path = hf_hub_download(
            repo_id=DATASET_REPOSITORY,
            filename=DATASET_FILENAME,
            revision=DATASET_REVISION,
            repo_type="dataset",
        )
    except Exception as error:
        raise RuntimeError(
            f"could not download {DATASET_REPOSITORY}/{DATASET_FILENAME} at "
            f"revision {DATASET_REVISION}: {error}"
        ) from error
    dataset_counts = audit_jsonl(Path(dataset_path), revision=DATASET_REVISION)

    if (github_counts.trajectories, github_counts.actions) != (
        dataset_counts.trajectories,
        dataset_counts.actions,
    ):
        raise ValueError(
            "pinned GitHub and dataset counts disagree: "
            f"{github_counts.trajectories}/{github_counts.actions} versus "
            f"{dataset_counts.trajectories}/{dataset_counts.actions}"
        )

    observed = {
        "trajectories": github_counts.trajectories,
        "actions": github_counts.actions,
    }
    claimed = {"trajectories": CLAIMED_TRAJECTORIES, "actions": CLAIMED_ACTIONS}
    paper_reported = {
        "trajectories": PAPER_REPORTED_TRAJECTORIES,
        "actions": PAPER_REPORTED_ACTIONS,
    }
    verdict = "verified" if observed == claimed else "falsified"
    return {
        "claim": (
            "Challenge card: Benchmark generates 58 symbolic action sequences consisting "
            "of 13K actions from publicly available spreadsheets."
        ),
        "claim_source": "ICML 2026 Agent Reproducibility Challenge card",
        "counting_definition": "Count trajectory rows/files and sum their operations arrays.",
        "claimed": claimed,
        "paper_reported": paper_reported,
        "observed": observed,
        "sources": [asdict(github_counts), asdict(dataset_counts)],
        "interpretation": (
            "Both pinned release artifacts agree with the paper's report of 52 trajectories "
            "and 11,907 steps/operations and falsify the challenge-card wording."
        ),
        "verdict": verdict,
    }
