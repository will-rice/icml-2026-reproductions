from pathlib import Path
import pytest

from lrr_repro.cli import main
from lrr_repro.evidence import build_worker_proposal


def test_offline_audits_are_byte_identical(project_root, cache_dir, tmp_path):
    schema_path = project_root / "schema/evidence-v1.schema.json"
    first = tmp_path / "a.json"
    second = tmp_path / "b.json"

    argv_1 = [
        "audit",
        "--project-root",
        str(project_root),
        "--cache-dir",
        str(cache_dir),
        "--schema",
        str(schema_path),
        "--output",
        str(first),
    ]
    argv_2 = [
        "audit",
        "--project-root",
        str(project_root),
        "--cache-dir",
        str(cache_dir),
        "--schema",
        str(schema_path),
        "--output",
        str(second),
    ]

    assert main(argv_1) == 0
    assert main(argv_2) == 0
    assert first.read_bytes() == second.read_bytes()


def test_worker_proposal_has_no_external_mutation():
    proposal = build_worker_proposal(b"{}\n", "a" * 40, "b" * 40)
    assert proposal["requested_action"] == "controller_validation"
    assert proposal["external_mutations"] == []
