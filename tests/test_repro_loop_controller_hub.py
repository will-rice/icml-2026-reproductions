"""Controller-owned Space publication and live-submission observation tests."""

from __future__ import annotations

import argparse
import copy
from datetime import datetime, timedelta, timezone
import hashlib
import importlib
import json
import os
from pathlib import Path
import subprocess
from types import SimpleNamespace
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "skills" / "icml-repro-loop" / "scripts"
STATE = SCRIPTS / "state.py"
sys.path.insert(0, str(SCRIPTS))

import attestations
import attempts
import controller
import leases
import refresh
import store
import state
import telemetry


NOW = datetime(2026, 7, 25, 18, 0, tzinfo=timezone.utc)
SPACE_SHA = "space-sha-a"
PAPER_ID = "paper-a"
SPACE_ID = "wrice/repro-paper-a"


class RecordedHubClient:
    """Record exact publication calls and return controlled live Space data."""

    def __init__(self, info: SimpleNamespace) -> None:
        self.info = info
        self.calls: list[tuple] = []

    def create_repo(self, **kwargs):
        self.calls.append(("create_repo", kwargs))
        return SimpleNamespace(repo_id=kwargs["repo_id"])

    def upload_folder(self, **kwargs):
        self.calls.append(("upload_folder", kwargs))
        return SimpleNamespace(oid=SPACE_SHA)

    def space_info(self, repo_id, *, files_metadata):
        self.calls.append(("space_info", repo_id, files_metadata))
        return self.info


def canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value, allow_nan=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def git(worktree: Path, *argv: str) -> str:
    return subprocess.run(
        ("git", *argv),
        cwd=worktree,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def validation_record(
    attempt_id: str,
    worktree: Path,
    source_commit: str,
    source_tree: str,
    source_tree_sha256: str,
) -> dict:
    payload = {
        "worktree": str(worktree),
        "branch": "attempt-paper-a",
        "base_sha": source_commit,
        "project_path": "submissions/paper-a",
        "design_path": "docs/designs/paper-a.md",
        "commands": [],
        "checks": [],
        "environment": [],
        "source_tree": source_tree,
        "source_tree_sha256": source_tree_sha256,
        "environment_sha256": "e" * 64,
    }
    return {
        "kind": "validation",
        "attempt_id": attempt_id,
        "attempt_number": 1,
        "observed_at": (NOW - timedelta(minutes=5)).isoformat(),
        "source_commit": source_commit,
        "payload_sha256": canonical_sha256(payload),
        **payload,
    }


@pytest.fixture
def hub_case(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(controller, "validation_now", lambda: NOW)
    paths = store.StatePaths(tmp_path / "repro-loop.json")
    store.atomic_json_write(paths.index, store.new_index(), store.validate_index)
    lease = leases.acquire_lease(
        paths,
        "attempt:a1",
        "controller-1",
        "a1",
        NOW,
        timedelta(hours=2),
    )
    attempts.create_attempt(
        paths,
        "a1",
        {
            "paper_id": PAPER_ID,
            "title": "Paper A",
            "slug": "paper-a",
            "project_path": "submissions/paper-a",
            "upstream_revision": "upstream-a",
            "target_claims": ["claim-a-one", "claim-a-two"],
            "estimated_api_cost_usd": 0.0,
        },
        lease,
        "selection-snapshot",
        NOW,
    )
    attempts.transition_attempt(paths, "a1", "design-pending", lease, NOW)
    attempts.record_design(
        paths, "a1", lease, "author-a", "docs/designs/paper-a.md", NOW
    )
    attempts.record_design_review(
        paths, "a1", lease, "reviewer-a", "approved", NOW
    )

    worktree = tmp_path / "paper-worktree"
    source_dir = worktree / "submissions" / "paper-a"
    source_dir.mkdir(parents=True)
    (source_dir / "README.md").write_text(
        "---\ntitle: Paper A\ntags:\n- icml2026-repro\n- paper-paper-a\n---\n",
        encoding="utf-8",
    )
    (source_dir / "app.py").write_text(
        "print('paper a')\n", encoding="utf-8"
    )
    pages = source_dir / "pages"
    pages.mkdir()
    (pages / "reproduction.md").write_text(
        "Reproduced evidence. " * 20
        + "".join(f"\n| metric-{n} | {n}.{n} |" for n in range(16)),
        encoding="utf-8",
    )
    (pages / "results.md").write_text(
        "".join(f"claim {n}: measured {n}.0e-{n}\n" for n in range(16)),
        encoding="utf-8",
    )
    git(worktree, "init", "-b", "attempt-paper-a")
    git(worktree, "config", "user.name", "Test Controller")
    git(worktree, "config", "user.email", "controller@example.test")
    git(worktree, "add", ".")
    git(worktree, "commit", "-m", "validated source")
    source_commit = git(worktree, "rev-parse", "HEAD")
    source_tree = git(worktree, "rev-parse", "HEAD^{tree}")
    source_tree_sha256 = controller._source_tree_sha256(source_dir)
    record = validation_record(
        "a1",
        worktree,
        source_commit,
        source_tree,
        source_tree_sha256,
    )
    validation_id = attestations.persist(paths, record)
    attempts.transition_attested(
        paths, "a1", "validated", validation_id, {}, lease, NOW
    )
    info = SimpleNamespace(
        id=SPACE_ID,
        sha=SPACE_SHA,
        tags=["gradio", "icml2026-repro", "paper-paper-a"],
        runtime=SimpleNamespace(stage="RUNNING"),
    )
    return {
        "paths": paths,
        "lease": lease,
        "worktree": worktree,
        "source_dir": source_dir,
        "source_commit": source_commit,
        "source_tree": source_tree,
        "source_tree_sha256": source_tree_sha256,
        "validation_id": validation_id,
        "client": RecordedHubClient(info),
    }


def deploy(case: dict) -> dict:
    return controller.publish_and_attest_deployment(
        case["paths"],
        "a1",
        case["lease"],
        SPACE_ID,
        case["source_dir"],
        case["client"],
        NOW,
    )


@pytest.mark.parametrize(
    "relative_path,content",
    [
        ("space/pages/reproduction.md", "nested evidence " * 20),
        ("pages/reproduction.md", "x" * 199),
        ("pages/reproduction.txt", "valid evidence " * 20),
        ("pages/summary-only.md", "single page 1.0 2.0 " * 30),
        ("pages/no-numbers.md", "asserted reproduced verified " * 30),
    ],
)
def test_scoring_pages_require_direct_substantive_markdown(
    tmp_path: Path, relative_path: str, content: str
):
    source_dir = tmp_path / "source"
    page = source_dir / relative_path
    page.parent.mkdir(parents=True)
    page.write_text(content, encoding="utf-8")

    with pytest.raises(ValueError, match="scoring pages"):
        controller._require_scoring_pages(source_dir)


def snapshot_payload(
    fetched_at: datetime,
    *,
    spaces: list[dict] | None = None,
    tagged_spaces: list[dict] | None = None,
    queued_submissions: list[dict] | None = None,
    verdicts: list[dict] | None = None,
) -> dict:
    exact_space = {
        "paper_ids": [PAPER_ID],
        "space_id": SPACE_ID,
        "revision": SPACE_SHA,
        "tags": ["gradio", "icml2026-repro", "paper-paper-a"],
    }
    return {
        "fetched_at": fetched_at.isoformat(),
        "source_revision": "f" * 64,
        "sources": {
            "challenge": {"revision": "challenge-rev"},
            "verdicts": {"revision": "verdict-rev"},
        },
        "assessments": None,
        "candidates": [],
        "queued_submissions": (
            [
                {
                    "paper_id": PAPER_ID,
                    "space_id": SPACE_ID,
                    "revision": SPACE_SHA,
                    "status": "pending",
                }
            ]
            if queued_submissions is None
            else queued_submissions
        ),
        "tagged_spaces": (
            [
                {
                    "paper_id": PAPER_ID,
                    "space_id": SPACE_ID,
                    "revision": SPACE_SHA,
                }
            ]
            if tagged_spaces is None
            else tagged_spaces
        ),
        "verdicts": [] if verdicts is None else verdicts,
        "spaces": [exact_space] if spaces is None else spaces,
    }


def persist_snapshot(case: dict, payload: dict) -> str:
    return refresh.persist_snapshot(case["paths"], payload)


def test_publication_uploads_only_validated_tree_and_attests_exact_live_space(
    hub_case,
):
    transitioned = deploy(hub_case)

    record = attestations.read(
        hub_case["paths"], transitioned["transitions"][-1]["attestation_id"]
    )
    assert transitioned["phase"] == "deployed"
    assert transitioned["space_id"] == SPACE_ID
    assert transitioned["deployed_sha"] == SPACE_SHA
    assert record == {
        "attestation_id": record["attestation_id"],
        "kind": "deployment",
        "attempt_id": "a1",
        "attempt_number": 1,
        "observed_at": NOW.isoformat(),
        "source_commit": hub_case["source_commit"],
        "payload_sha256": record["payload_sha256"],
        "space_id": SPACE_ID,
        "space_sha": SPACE_SHA,
        "owner": "wrice",
        "tags": ["gradio", "icml2026-repro", "paper-paper-a"],
        "runtime_stage": "RUNNING",
        "validation_attestation_id": hub_case["validation_id"],
        "source_tree_sha256": hub_case["source_tree_sha256"],
    }
    assert record["payload_sha256"] == canonical_sha256(
        {
            key: record[key]
            for key in (
                "space_id",
                "space_sha",
                "owner",
                "tags",
                "runtime_stage",
                "validation_attestation_id",
                "source_tree_sha256",
            )
        }
    )
    assert hub_case["client"].calls == [
        (
            "create_repo",
            {
                "repo_id": SPACE_ID,
                "repo_type": "space",
                "space_sdk": "gradio",
                "exist_ok": True,
            },
        ),
        (
            "upload_folder",
            {
                "repo_id": SPACE_ID,
                "folder_path": hub_case["source_dir"],
                "repo_type": "space",
                "commit_message": (
                    f"Publish validated {hub_case['source_commit']}"
                ),
            },
        ),
        ("space_info", SPACE_ID, True),
    ]


@pytest.mark.parametrize(
    ("space_id", "tags", "stage", "sha", "error"),
    [
        (
            "other/repro-paper-a",
            ["icml2026-repro", "paper-paper-a"],
            "RUNNING",
            SPACE_SHA,
            "owner",
        ),
        (
            SPACE_ID,
            ["paper-paper-a"],
            "RUNNING",
            SPACE_SHA,
            "tag",
        ),
        (
            SPACE_ID,
            ["icml2026-repro"],
            "RUNNING",
            SPACE_SHA,
            "tag",
        ),
        (
            SPACE_ID,
            ["icml2026-repro", "paper-paper-a"],
            "CONFIG_ERROR",
            SPACE_SHA,
            "runtime",
        ),
        (
            SPACE_ID,
            ["icml2026-repro", "paper-paper-a"],
            "APP_STARTING",
            SPACE_SHA,
            "runtime",
        ),
        (
            SPACE_ID,
            ["icml2026-repro", "paper-paper-a"],
            "RUNNING",
            "wrong-space-sha",
            "revision",
        ),
    ],
)
def test_publication_rejects_noncanonical_live_space(
    hub_case, space_id, tags, stage, sha, error
):
    hub_case["client"].info = SimpleNamespace(
        id=space_id,
        sha=sha,
        tags=tags,
        runtime=SimpleNamespace(stage=stage),
    )

    with pytest.raises(ValueError, match=error):
        controller.publish_and_attest_deployment(
            hub_case["paths"],
            "a1",
            hub_case["lease"],
            space_id,
            hub_case["source_dir"],
            hub_case["client"],
            NOW,
        )

    assert attempts.read_attempt(hub_case["paths"], "a1")["phase"] == "validated"
    assert not hub_case["paths"].attestation("deployment", "a1").exists()


def test_publication_requires_authoritative_validation_attestation(hub_case):
    hub_case["paths"].attestation("validation", "a1").unlink()

    with pytest.raises(ValueError, match="validation"):
        deploy(hub_case)

    assert hub_case["client"].calls == []


def test_publication_checks_scoring_pages_before_hub_mutation(
    hub_case, monkeypatch
):
    def reject_pages(source_dir):
        raise ValueError("scoring pages")

    monkeypatch.setattr(controller, "_require_scoring_pages", reject_pages)

    with pytest.raises(ValueError, match="scoring pages"):
        deploy(hub_case)

    assert hub_case["client"].calls == []


def test_publication_rechecks_fence_after_hub_mutation(hub_case, monkeypatch):
    monkeypatch.setattr(
        controller,
        "validation_now",
        lambda: NOW + timedelta(hours=3),
    )

    with pytest.raises(leases.StaleFence):
        deploy(hub_case)

    assert attempts.read_attempt(hub_case["paths"], "a1")["phase"] == "validated"
    assert not hub_case["paths"].attestation("deployment", "a1").exists()
    assert [call[0] for call in hub_case["client"].calls] == [
        "create_repo",
        "upload_folder",
        "space_info",
    ]


def test_publication_rejects_source_directory_outside_validated_project(hub_case):
    other = hub_case["worktree"] / "submissions" / "other"
    other.mkdir()

    with pytest.raises(ValueError, match="source_dir"):
        controller.publish_and_attest_deployment(
            hub_case["paths"],
            "a1",
            hub_case["lease"],
            SPACE_ID,
            other,
            hub_case["client"],
            NOW,
        )

    assert hub_case["client"].calls == []


def replace_validation_source_hash(case: dict, source_tree_sha256: str) -> None:
    slot = case["paths"].attestation("validation", "a1")
    replacement = validation_record(
        "a1",
        case["worktree"],
        case["source_commit"],
        case["source_tree"],
        source_tree_sha256,
    )
    replacement_id = attestations.persist(case["paths"], replacement)
    replacement_record = attestations.read(case["paths"], replacement_id)
    slot.unlink()
    store.atomic_json_write(
        slot,
        replacement_record,
        lambda record: attestations.validate_target(
            case["paths"], slot, record
        ),
    )
    attempt = attempts.read_attempt(case["paths"], "a1")
    attempt["transitions"][-1]["attestation_id"] = replacement_id
    store.atomic_json_write(
        case["paths"].attempt("a1"), attempt, store.validate_attempt
    )


def test_publication_rejects_prevalidated_ignored_file_inside_project(hub_case):
    exclude = hub_case["worktree"] / ".git" / "info" / "exclude"
    exclude.write_text(
        "submissions/paper-a/ignored.bin\n",
        encoding="utf-8",
    )
    ignored = hub_case["source_dir"] / "ignored.bin"
    ignored.write_bytes(b"must not upload")
    replace_validation_source_hash(
        hub_case,
        controller._source_tree_sha256(hub_case["source_dir"]),
    )

    with pytest.raises(ValueError, match="source tree"):
        deploy(hub_case)

    assert hub_case["client"].calls == []


def test_publication_allows_ignored_file_outside_project(hub_case):
    exclude = hub_case["worktree"] / ".git" / "info" / "exclude"
    exclude.write_text("root-ignored.bin\n", encoding="utf-8")
    (hub_case["worktree"] / "root-ignored.bin").write_bytes(b"outside project")

    transitioned = deploy(hub_case)

    assert transitioned["phase"] == "deployed"
    upload = [
        call for call in hub_case["client"].calls if call[0] == "upload_folder"
    ]
    assert upload[0][1]["folder_path"] == hub_case["source_dir"]


def test_publication_rejects_validated_project_root_symlink_to_outside(
    hub_case, tmp_path: Path
):
    source_dir = hub_case["source_dir"]
    outside = tmp_path / "outside-publish-source"
    source_dir.rename(outside)
    source_dir.symlink_to(outside, target_is_directory=True)
    git(hub_case["worktree"], "add", "-A")
    git(hub_case["worktree"], "commit", "-m", "replace project with symlink")
    source_commit = git(hub_case["worktree"], "rev-parse", "HEAD")
    source_tree = git(hub_case["worktree"], "rev-parse", "HEAD^{tree}")
    source_tree_sha256 = canonical_sha256(
        [
            {
                "path": path.relative_to(outside).as_posix(),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
            for path in sorted(outside.rglob("*"))
            if path.is_file()
        ]
    )
    slot = hub_case["paths"].attestation("validation", "a1")
    replacement = validation_record(
        "a1",
        hub_case["worktree"],
        source_commit,
        source_tree,
        source_tree_sha256,
    )
    replacement_id = attestations.persist(hub_case["paths"], replacement)
    replacement_record = attestations.read(
        hub_case["paths"], replacement_id
    )
    slot.unlink()
    store.atomic_json_write(
        slot,
        replacement_record,
        lambda record: attestations.validate_target(
            hub_case["paths"], slot, record
        ),
    )
    attempt = attempts.read_attempt(hub_case["paths"], "a1")
    attempt["transitions"][-1]["attestation_id"] = replacement_id
    store.atomic_json_write(
        hub_case["paths"].attempt("a1"), attempt, store.validate_attempt
    )

    with pytest.raises(ValueError, match="source_dir"):
        deploy(hub_case)

    assert hub_case["client"].calls == []


def test_publication_rejects_validation_tree_hash_mismatch(hub_case):
    slot = hub_case["paths"].attestation("validation", "a1")
    replacement = validation_record(
        "a1",
        hub_case["worktree"],
        hub_case["source_commit"],
        "9" * 40,
        hub_case["source_tree_sha256"],
    )
    replacement_id = attestations.persist(hub_case["paths"], replacement)
    replacement_record = attestations.read(
        hub_case["paths"], replacement_id
    )
    slot.unlink()
    store.atomic_json_write(
        slot,
        replacement_record,
        lambda record: attestations.validate_target(
            hub_case["paths"], slot, record
        ),
    )
    attempt = attempts.read_attempt(hub_case["paths"], "a1")
    attempt["transitions"][-1]["attestation_id"] = replacement_id
    store.atomic_json_write(
        hub_case["paths"].attempt("a1"), attempt, store.validate_attempt
    )

    with pytest.raises(ValueError, match="source tree"):
        deploy(hub_case)

    assert hub_case["client"].calls == []


def test_submission_attests_exact_newer_snapshot_without_synthetic_id(hub_case):
    deploy(hub_case)
    observed_at = NOW + timedelta(minutes=5)
    snapshot_id = persist_snapshot(
        hub_case, snapshot_payload(observed_at)
    )

    transitioned = controller.attest_submission(
        hub_case["paths"],
        "a1",
        hub_case["lease"],
        snapshot_id,
        observed_at,
    )

    record = attestations.read(
        hub_case["paths"], transitioned["transitions"][-1]["attestation_id"]
    )
    assert transitioned["phase"] == "submitted"
    assert record["snapshot_id"] == snapshot_id
    assert record["verdict_revision"] == "verdict-rev"
    assert record["space_id"] == SPACE_ID
    assert record["space_sha"] == SPACE_SHA
    assert record["paper_id"] == PAPER_ID
    assert record["queue_status"] == "pending"
    assert "submission_id" not in record


def invalid_submission_payload(case: dict, variant: str) -> dict:
    payload = snapshot_payload(NOW + timedelta(minutes=5))
    if variant == "older":
        payload["fetched_at"] = (NOW - timedelta(seconds=1)).isoformat()
    elif variant == "space":
        payload["spaces"][0]["space_id"] = "wrice/repro-other"
        payload["tagged_spaces"][0]["space_id"] = "wrice/repro-other"
        payload["queued_submissions"][0]["space_id"] = "wrice/repro-other"
    elif variant == "revision":
        payload["spaces"][0]["revision"] = "wrong-space-sha"
        payload["tagged_spaces"][0]["revision"] = "wrong-space-sha"
        payload["queued_submissions"][0]["revision"] = "wrong-space-sha"
    elif variant == "paper":
        payload["spaces"][0]["paper_ids"] = []
        payload["tagged_spaces"] = []
        payload["queued_submissions"] = []
    elif variant == "duplicate":
        duplicate = copy.deepcopy(payload["spaces"][0])
        duplicate["space_id"] = "wrice/repro-paper-a-copy"
        payload["spaces"].append(duplicate)
    elif variant == "conflicting-verdict":
        payload["verdicts"] = [
            {
                "paper_id": PAPER_ID,
                "space_id": "wrice/repro-paper-a-old",
                "source_revision": "verdict-rev",
            }
        ]
    else:
        raise AssertionError(variant)
    return payload


@pytest.mark.parametrize(
    ("variant", "error"),
    [
        ("older", "snapshot"),
        ("space", "space"),
        ("revision", "revision"),
        ("paper", "paper"),
        ("duplicate", "duplicate"),
    ],
)
def test_submission_rejects_nonexact_live_observation(
    hub_case, variant, error
):
    deploy(hub_case)
    payload = invalid_submission_payload(hub_case, variant)
    snapshot_id = persist_snapshot(hub_case, payload)
    observed_at = NOW + timedelta(minutes=5)

    with pytest.raises(ValueError, match=error):
        controller.attest_submission(
            hub_case["paths"],
            "a1",
            hub_case["lease"],
            snapshot_id,
            observed_at,
        )

    assert attempts.read_attempt(hub_case["paths"], "a1")["phase"] == "deployed"
    assert not hub_case["paths"].attestation("submission", "a1").exists()


def test_submission_allows_other_owner_verdict_for_same_paper(hub_case):
    deploy(hub_case)
    observed_at = NOW + timedelta(minutes=5)
    payload = invalid_submission_payload(hub_case, "conflicting-verdict")
    snapshot_id = persist_snapshot(hub_case, payload)

    transitioned = controller.attest_submission(
        hub_case["paths"],
        "a1",
        hub_case["lease"],
        snapshot_id,
        observed_at,
    )

    assert transitioned["phase"] == "submitted"


def test_state_cli_exposes_controller_hub_commands():
    result = subprocess.run(
        [sys.executable, str(STATE), "--help"],
        check=False,
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": str(SCRIPTS)},
    )

    assert result.returncode == 0
    assert "publish-deployment" in result.stdout
    assert "attest-submission" in result.stdout


def test_publication_policy_validates_allowlisted_space_owners():
    publication_policy = importlib.import_module("publication_policy")

    assert "wrice" in publication_policy.ALLOWED_SPACE_OWNERS
    assert publication_policy.space_owner("wrice/repro-paper") == "wrice"
    for invalid in (
        None,
        "",
        "wrice",
        "/repro-paper",
        "wrice/",
        "wrice/repro-paper/extra",
    ):
        with pytest.raises(ValueError, match="space_id"):
            publication_policy.space_owner(invalid)


def test_state_measures_injected_deployment_operation(hub_case):
    result = {"transitions": [{"attestation_id": "b" * 64}]}
    arguments = argparse.Namespace(
        command="publish-deployment",
        path=hub_case["paths"].index,
        attempt_id="a1",
        owner=hub_case["lease"].owner,
        fencing_token=hub_case["lease"].fencing_token,
        space_id=SPACE_ID,
        source_dir=Path("unused-injected-source"),
        now=NOW.isoformat(),
    )

    returned = state._run_v6_command(
        arguments,
        deployment_operation=lambda: result,
        utc_now=iter(
            ["2026-07-27T01:00:00+00:00", "2026-07-27T01:00:02+00:00"]
        ).__next__,
        monotonic_ns=iter([3_000_000_000, 5_000_000_000]).__next__,
        session_id_factory=lambda: "deployment-stage",
    )

    assert returned is result
    assert hub_case["client"].calls == []
    finished = telemetry.read_session(
        hub_case["paths"], "deployment-stage"
    )[-1]
    assert finished["elapsed_seconds"] == 2.0
    assert finished["attestation_id"] == "b" * 64


def test_state_observes_submission_only_after_injected_success(hub_case):
    result = {"transitions": [{"attestation_id": "c" * 64}]}
    snapshot_id = "d" * 64
    arguments = argparse.Namespace(
        command="attest-submission",
        path=hub_case["paths"].index,
        attempt_id="a1",
        owner=hub_case["lease"].owner,
        fencing_token=hub_case["lease"].fencing_token,
        snapshot_id=snapshot_id,
        now=NOW.isoformat(),
    )

    returned = state._run_v6_command(
        arguments,
        submission_operation=lambda: result,
        utc_now=lambda: "2026-07-27T01:00:00+00:00",
        session_id_factory=lambda: "submission-observation",
    )

    assert returned is result
    assert telemetry.read_session(
        hub_case["paths"], "submission-observation"
    ) == [
        {
            "version": 1,
            "session_id": "submission-observation",
            "sequence": 0,
            "event": "observation",
            "name": "submission-observed",
            "attempt_id": "a1",
            "snapshot_id": snapshot_id,
            "attestation_id": "c" * 64,
            "observed_at": "2026-07-27T01:00:00+00:00",
        }
    ]


def test_state_does_not_observe_failed_submission(hub_case):
    arguments = argparse.Namespace(
        command="attest-submission",
        path=hub_case["paths"].index,
        attempt_id="a1",
        owner=hub_case["lease"].owner,
        fencing_token=hub_case["lease"].fencing_token,
        snapshot_id="d" * 64,
        now=NOW.isoformat(),
    )

    def fail():
        raise RuntimeError("authoritative failure")

    with pytest.raises(RuntimeError, match="authoritative failure"):
        state._run_v6_command(
            arguments,
            submission_operation=fail,
            utc_now=lambda: "2026-07-27T01:00:00+00:00",
            session_id_factory=lambda: "failed-submission",
        )

    assert telemetry.read_session(hub_case["paths"], "failed-submission") == []


def test_state_returns_submission_result_when_observation_write_fails(
    hub_case, monkeypatch
):
    result = {"transitions": [{"attestation_id": "c" * 64}]}
    arguments = argparse.Namespace(
        command="attest-submission",
        path=hub_case["paths"].index,
        attempt_id="a1",
        owner=hub_case["lease"].owner,
        fencing_token=hub_case["lease"].fencing_token,
        snapshot_id="d" * 64,
        now=NOW.isoformat(),
    )

    def fail_observation(*_args, **_kwargs):
        raise OSError("telemetry-observation-sensitive-text")

    active_telemetry = sys.modules["telemetry"]
    monkeypatch.setattr(active_telemetry, "append_event", fail_observation)

    returned = state._run_v6_command(
        arguments,
        submission_operation=lambda: result,
        utc_now=lambda: "2026-07-27T01:00:00+00:00",
        session_id_factory=lambda: "submission-observation-failed",
    )

    assert returned is result
    assert active_telemetry.read_session(
        hub_case["paths"], "submission-observation-failed"
    ) == []


def already_judged_verdict(sha: str) -> dict:
    return {
        "paper_id": PAPER_ID,
        "space_id": SPACE_ID,
        "sha": sha,
        "source_revision": "verdict-rev",
        "judged_at": (NOW + timedelta(minutes=1)).isoformat(),
    }


def test_submission_attests_already_judged_space_without_queue_entry(hub_case):
    """A deployed Space whose exact SHA is already judged submits as judged."""
    deploy(hub_case)
    observed_at = NOW + timedelta(minutes=5)
    snapshot_id = persist_snapshot(
        hub_case,
        snapshot_payload(
            observed_at,
            queued_submissions=[],
            verdicts=[already_judged_verdict(SPACE_SHA)],
        ),
    )

    transitioned = controller.attest_submission(
        hub_case["paths"],
        "a1",
        hub_case["lease"],
        snapshot_id,
        observed_at,
    )

    record = attestations.read(
        hub_case["paths"], transitioned["transitions"][-1]["attestation_id"]
    )
    assert transitioned["phase"] == "submitted"
    assert record["queue_status"] == "judged"
    assert record["space_sha"] == SPACE_SHA


def test_submission_rejects_missing_queue_without_exact_verdict(hub_case):
    deploy(hub_case)
    observed_at = NOW + timedelta(minutes=5)
    snapshot_id = persist_snapshot(
        hub_case,
        snapshot_payload(observed_at, queued_submissions=[], verdicts=[]),
    )

    with pytest.raises(ValueError, match="official_verdict"):
        controller.attest_submission(
            hub_case["paths"],
            "a1",
            hub_case["lease"],
            snapshot_id,
            observed_at,
        )


def test_submission_rejects_judged_verdict_at_different_sha(hub_case):
    deploy(hub_case)
    observed_at = NOW + timedelta(minutes=5)
    snapshot_id = persist_snapshot(
        hub_case,
        snapshot_payload(
            observed_at,
            queued_submissions=[],
            verdicts=[already_judged_verdict("stale-space-sha")],
        ),
    )

    with pytest.raises(ValueError, match="queue"):
        controller.attest_submission(
            hub_case["paths"],
            "a1",
            hub_case["lease"],
            snapshot_id,
            observed_at,
        )
