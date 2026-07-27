import re
from pathlib import Path

import yaml


def test_pre_commit_excludes_byte_exact_upstream_evidence() -> None:
    config = yaml.safe_load(Path(".pre-commit-config.yaml").read_text())

    for path in (
        "submissions/example/evidence/inputs/upstream/source.py",
        "submissions/example/artifacts/source/repository/source.py",
        "submissions/example/inputs/upstream/source.json",
    ):
        assert re.search(config["exclude"], path)
