import re
import subprocess
from pathlib import Path

import yaml


def test_live_coordinator_state_is_not_git_tracked() -> None:
    """Tracked state/ files get reverted by git resets, corrupting live
    coordination (lost claims, stale leases, duplicate paper assignment)."""
    tracked = subprocess.run(
        ["git", "ls-files", "--", "state"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    assert tracked == []


def test_pre_commit_excludes_byte_exact_upstream_evidence() -> None:
    config = yaml.safe_load(Path(".pre-commit-config.yaml").read_text())

    for path in (
        "submissions/example/evidence/inputs/upstream/source.py",
        "submissions/example/artifacts/source/repository/source.py",
        "submissions/example/inputs/upstream/source.json",
    ):
        assert re.search(config["exclude"], path)
