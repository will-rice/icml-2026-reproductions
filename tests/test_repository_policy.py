import re
from pathlib import Path

import yaml


def test_pre_commit_excludes_byte_exact_upstream_evidence() -> None:
    config = yaml.safe_load(Path(".pre-commit-config.yaml").read_text())

    assert re.search(
        config["exclude"],
        "submissions/example/evidence/inputs/upstream/source.py",
    )
