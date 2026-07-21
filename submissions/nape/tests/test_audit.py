import json
import subprocess

import pytest

from icml_2026_repro import audit as audit_module
from icml_2026_repro.audit import (
    audit_jsonl,
    audit_trajectory_directory,
    build_challenge_card_claim_1_audit,
)


def test_audit_trajectory_directory_counts_files_and_operations(tmp_path):
    for name, operations in {"a": ["one", "two"], "b": ["three"]}.items():
        (tmp_path / f"{name}.json").write_text(
            json.dumps({"name": name, "operations": operations}), encoding="utf-8"
        )

    result = audit_trajectory_directory(tmp_path, revision="abc123")

    assert result.trajectories == 2
    assert result.actions == 3
    assert result.revision == "abc123"


def test_audit_trajectory_directory_rejects_missing_operations(tmp_path):
    (tmp_path / "bad.json").write_text(json.dumps({"name": "bad"}), encoding="utf-8")

    with pytest.raises(ValueError, match="operations"):
        audit_trajectory_directory(tmp_path, revision="abc123")


def test_audit_jsonl_counts_rows_and_operations(tmp_path):
    dataset_path = tmp_path / "test.jsonl"
    rows = [
        {"name": "a", "operations": ["one", "two"], "num_operations": 2},
        {"name": "b", "operations": ["three"], "num_operations": 1},
    ]
    dataset_path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")

    result = audit_jsonl(dataset_path, revision="def456")

    assert result.source == "dataset:Tej-a55/napeval"
    assert result.trajectories == 2
    assert result.actions == 3
    assert result.revision == "def456"


def test_audit_jsonl_rejects_inconsistent_num_operations(tmp_path):
    dataset_path = tmp_path / "test.jsonl"
    dataset_path.write_text(
        json.dumps({"name": "bad", "operations": ["one"], "num_operations": 2}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="num_operations"):
        audit_jsonl(dataset_path, revision="def456")


@pytest.mark.parametrize(
    "row",
    [
        {"operations": ["one"], "num_operations": 1},
        {"name": 1, "operations": ["one"], "num_operations": 1},
    ],
)
def test_audit_jsonl_requires_string_name(tmp_path, row):
    dataset_path = tmp_path / "test.jsonl"
    dataset_path.write_text(json.dumps(row), encoding="utf-8")

    with pytest.raises(ValueError, match="name must be a string"):
        audit_jsonl(dataset_path, revision="def456")


def test_challenge_card_claim_1_audit_rejects_dirty_trajectory_contents(tmp_path, monkeypatch):
    repository_root = tmp_path / "repository"
    nape_path = repository_root / "external" / "NAPE"
    trajectory_path = nape_path / "data" / "trajectories"
    trajectory_path.mkdir(parents=True)
    tracked_path = trajectory_path / "tracked.json"
    tracked_path.write_text(json.dumps({"operations": ["one"]}), encoding="utf-8")
    subprocess.run(["git", "init", str(nape_path)], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(nape_path), "add", str(tracked_path)], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(nape_path),
            "-c",
            "user.name=Test User",
            "-c",
            "user.email=test@example.com",
            "commit",
            "-m",
            "test fixture",
        ],
        check=True,
        capture_output=True,
    )
    tracked_path.write_text(json.dumps({"operations": ["changed"]}), encoding="utf-8")
    (trajectory_path / "untracked.json").write_text(
        json.dumps({"operations": ["untracked"]}), encoding="utf-8"
    )

    monkeypatch.setattr(audit_module, "REPOSITORY_ROOT", repository_root)
    monkeypatch.setattr(audit_module, "read_git_head", lambda _: audit_module.GITHUB_REVISION)

    with pytest.raises(ValueError, match="tracked.json.*untracked.json"):
        build_challenge_card_claim_1_audit()


def test_challenge_card_claim_1_audit_falsifies_mismatched_claim(tmp_path, monkeypatch):
    repository_root = tmp_path / "repository"
    trajectory_path = repository_root / "external" / "NAPE" / "data" / "trajectories"
    trajectory_path.mkdir(parents=True)
    for index in range(52):
        operations = ["operation"] * (11907 - 51) if index == 0 else ["operation"]
        (trajectory_path / f"{index:04d}.json").write_text(
            json.dumps({"operations": operations}), encoding="utf-8"
        )

    dataset_path = tmp_path / "test.jsonl"
    dataset_path.write_text(
        "\n".join(
            json.dumps(
                {
                    "name": str(index),
                    "operations": ["operation"] * (11907 - 51) if index == 0 else ["operation"],
                    "num_operations": 11907 - 51 if index == 0 else 1,
                }
            )
            for index in range(52)
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(audit_module, "REPOSITORY_ROOT", repository_root)
    monkeypatch.setattr(audit_module, "read_git_head", lambda _: audit_module.GITHUB_REVISION)
    monkeypatch.setattr(audit_module, "read_git_trajectory_status", lambda _: "")
    monkeypatch.setattr(audit_module, "hf_hub_download", lambda **_: str(dataset_path))

    evidence = build_challenge_card_claim_1_audit()

    assert evidence["claimed"] == {"trajectories": 58, "actions": 13000}
    assert evidence["observed"] == {"trajectories": 52, "actions": 11907}
    assert evidence["paper_reported"] == {"trajectories": 52, "actions": 11907}
    assert evidence["claim_source"] == "ICML 2026 Agent Reproducibility Challenge card"
    assert evidence["interpretation"] == (
        "Both pinned release artifacts agree with the paper's report of 52 trajectories "
        "and 11,907 steps/operations and falsify the challenge-card wording."
    )
    assert evidence["verdict"] == "falsified"
