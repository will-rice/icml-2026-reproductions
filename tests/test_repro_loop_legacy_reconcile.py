"""Tests for one-time reconciliation of a migrated schema-v3 attempt."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "skills" / "icml-repro-loop" / "scripts"
FIXTURE = ROOT / "tests" / "fixtures" / "repro-loop-v3-eeg.json"
sys.path.insert(0, str(SCRIPTS))

import attempts
import leases
import migrate_v6
import refresh
import store


NOW = datetime(2026, 7, 25, 21, 0, tzinfo=timezone.utc)
SOURCE_STATE_SHA256 = (
    "f9fb0c976243de61b8fe90441e100c6bc88f341a50adb5326ffd12c8d7e99354"
)
EEG_ATTEMPT_ID = "e20658d7-250a-5b0c-a015-be453c43e9fc"
DESIGN_PATH = "docs/superpowers/plans/2026-07-24-eeg-fm-bench.md"
APPROVAL_REF = "git:1d2c4c74700b66068d43443f7c8e742d4dc797a5"


def migrated_eeg(tmp_path: Path) -> tuple[store.StatePaths, leases.Lease]:
    source = json.loads(FIXTURE.read_text(encoding="utf-8"))
    paths = store.StatePaths(tmp_path / "repro-loop.json")
    store.atomic_json_write(paths.index, source, migrate_v6.legacy_state.validate_state)
    migrate_v6.apply_v6_migration(paths, migrate_v6.plan_v6_migration(source))
    lease = leases.claim_attempt(
        paths,
        EEG_ATTEMPT_ID,
        "controller-reconcile",
        0,
        NOW,
    )
    return paths, lease


def eeg_candidate() -> dict:
    target_claims = [
        "fourteen-dataset-ten-paradigm-curation",
        "standardized-preprocessing-reproducibility",
        "three-strategy-evaluation-harness",
    ]
    challenge_claims = [
        "EEG-FM-Bench curates fourteen datasets across ten EEG paradigms.",
        "EEG-FM-Bench defines a standardized preprocessing pipeline.",
        "EEG-FM-Bench evaluates three fine-tuning strategies.",
    ]
    return {
        "paper_id": "vGeNaFHdET",
        "title": (
            "EEG-FM-Bench: A Comprehensive Benchmark for the Systematic "
            "Evaluation and Diagnostic Analyses of EEG Foundation Models"
        ),
        "slug": "eeg-fm-bench",
        "upstream_revision": (
            "arxiv:2508.17742v3+github:xw1216/EEG-FM-Bench"
            "@325398d7d057ecc1216fb3510d70c16eb60337cc"
        ),
        "target_claims": target_claims,
        "claim_bindings": [
            {
                "target_claim": target_claim,
                "challenge_claim": challenge_claim,
                "challenge_claim_sha256": hashlib.sha256(
                    challenge_claim.encode("utf-8")
                ).hexdigest(),
            }
            for target_claim, challenge_claim in zip(
                target_claims, challenge_claims, strict=True
            )
        ],
        "live_claims": [
            {"text": claim, "status": "unverified"} for claim in challenge_claims
        ],
        "estimated_api_cost_usd": 0.0,
        "score": 18,
        "artifact_access": True,
        "cpu_only": True,
        "safety_blocker": None,
        "licensing_blocker": None,
    }


def write_eeg_snapshot(
    paths: store.StatePaths,
    *,
    candidate: dict | None = None,
    assessments: dict | None = None,
    tagged_spaces: list[dict] | None = None,
) -> tuple[str, dict]:
    candidate = eeg_candidate() if candidate is None else candidate
    payload = {
        "fetched_at": NOW.isoformat(),
        "source_revision": "challenge-revision",
        "sources": {
            "challenge": {"revision": "challenge-revision"},
            "verdicts": {"revision": "verdict-revision"},
        },
        "assessments": assessments,
        "candidates": [candidate],
        "queued_submissions": [],
        "tagged_spaces": tagged_spaces or [],
        "verdicts": [],
        "spaces": [],
    }
    return refresh.persist_snapshot(paths, payload), candidate


def assessed_snapshot(paths: store.StatePaths) -> tuple[str, dict]:
    return write_eeg_snapshot(
        paths,
        assessments={
            "content_sha256": "a" * 64,
            "document": {"challenge_revision": "challenge-revision"},
        },
    )


def test_reconcile_migrated_eeg_binds_fresh_claims_and_design(tmp_path: Path):
    paths, lease = migrated_eeg(tmp_path)
    snapshot_id, candidate = assessed_snapshot(paths)

    updated = attempts.reconcile_legacy_attempt(
        paths,
        EEG_ATTEMPT_ID,
        lease,
        snapshot_id,
        design_author="eeg-design-author",
        design_path=DESIGN_PATH,
        reviewer="user-approved-design",
        approval_ref=APPROVAL_REF,
        now=NOW,
    )

    assert updated["phase"] == "implementing"
    assert updated["snapshot_id"] == snapshot_id
    assert updated["target_claims"] == candidate["target_claims"]
    assert updated["claim_bindings"] == candidate["claim_bindings"]
    assert updated["live_claims"] == candidate["live_claims"]
    assert updated["design"] == {
        "author": "eeg-design-author",
        "path": DESIGN_PATH,
        "recorded_at": NOW.isoformat(),
    }
    assert updated["design_review"] == {
        "reviewer": "user-approved-design",
        "decision": "approved",
        "reviewed_at": NOW.isoformat(),
    }
    assert updated["legacy_reconciliation"] == {
        "source_state_sha256": SOURCE_STATE_SHA256,
        "snapshot_id": snapshot_id,
        "approval_ref": APPROVAL_REF,
        "reconciled_at": NOW.isoformat(),
    }
    assert store.read_json(paths.attempt(EEG_ATTEMPT_ID)) == updated
    backup = paths.root / "v3-backups" / f"{SOURCE_STATE_SHA256}.json"
    assert hashlib.sha256(backup.read_bytes()).hexdigest() == SOURCE_STATE_SHA256


def test_reconcile_migrated_eeg_is_available_through_controller_cli(
    tmp_path: Path,
):
    paths, lease = migrated_eeg(tmp_path)
    snapshot_id, _candidate = assessed_snapshot(paths)

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPTS / "state.py"),
            "reconcile-legacy-attempt",
            str(paths.index),
            "--attempt-id",
            EEG_ATTEMPT_ID,
            "--owner",
            lease.owner,
            "--fencing-token",
            str(lease.fencing_token),
            "--snapshot-id",
            snapshot_id,
            "--design-author",
            "eeg-design-author",
            "--design-path",
            DESIGN_PATH,
            "--reviewer",
            "user-approved-design",
            "--approval-ref",
            APPROVAL_REF,
            "--now",
            NOW.isoformat(),
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    reconciled = json.loads(result.stdout)
    assert reconciled["attempt_id"] == EEG_ATTEMPT_ID
    assert reconciled["snapshot_id"] == snapshot_id
    assert reconciled["legacy_reconciliation"]["approval_ref"] == APPROVAL_REF


def test_generic_transition_cannot_rewrite_reconciled_claim_authority(
    tmp_path: Path,
):
    paths, lease = migrated_eeg(tmp_path)
    snapshot_id, _candidate = assessed_snapshot(paths)
    reconciled = attempts.reconcile_legacy_attempt(
        paths,
        EEG_ATTEMPT_ID,
        lease,
        snapshot_id,
        design_author="eeg-design-author",
        design_path=DESIGN_PATH,
        reviewer="user-approved-design",
        approval_ref=APPROVAL_REF,
        now=NOW,
    )

    with pytest.raises(ValueError, match="claim_bindings"):
        attempts.transition_attempt(
            paths,
            EEG_ATTEMPT_ID,
            "blocked",
            lease,
            NOW,
            blocker="pause",
            claim_bindings=[],
        )

    assert attempts.read_attempt(paths, EEG_ATTEMPT_ID) == reconciled


def test_reconcile_rejects_raw_snapshot_without_mutating_attempt(tmp_path: Path):
    paths, lease = migrated_eeg(tmp_path)
    snapshot_id, _candidate = write_eeg_snapshot(paths)
    original = attempts.read_attempt(paths, EEG_ATTEMPT_ID)

    with pytest.raises(ValueError, match="assessments"):
        attempts.reconcile_legacy_attempt(
            paths,
            EEG_ATTEMPT_ID,
            lease,
            snapshot_id,
            design_author="eeg-design-author",
            design_path=DESIGN_PATH,
            reviewer="user-approved-design",
            approval_ref=APPROVAL_REF,
            now=NOW,
        )

    assert attempts.read_attempt(paths, EEG_ATTEMPT_ID) == original


def test_reconcile_rejects_changed_upstream_identity(tmp_path: Path):
    paths, lease = migrated_eeg(tmp_path)
    candidate = eeg_candidate()
    candidate["upstream_revision"] = "unreviewed-upstream"
    snapshot_id, _candidate = write_eeg_snapshot(
        paths,
        candidate=candidate,
        assessments={
            "content_sha256": "a" * 64,
            "document": {"challenge_revision": "challenge-revision"},
        },
    )
    original = attempts.read_attempt(paths, EEG_ATTEMPT_ID)

    with pytest.raises(ValueError, match="upstream_revision"):
        attempts.reconcile_legacy_attempt(
            paths,
            EEG_ATTEMPT_ID,
            lease,
            snapshot_id,
            design_author="eeg-design-author",
            design_path=DESIGN_PATH,
            reviewer="user-approved-design",
            approval_ref=APPROVAL_REF,
            now=NOW,
        )

    assert attempts.read_attempt(paths, EEG_ATTEMPT_ID) == original


def test_reconcile_rejects_live_external_claim_for_same_paper(tmp_path: Path):
    paths, lease = migrated_eeg(tmp_path)
    snapshot_id, _candidate = write_eeg_snapshot(
        paths,
        assessments={
            "content_sha256": "a" * 64,
            "document": {"challenge_revision": "challenge-revision"},
        },
        tagged_spaces=[
            {
                "paper_id": "vGeNaFHdET",
                "space_id": "other/repro-eeg",
                "revision": "space-sha",
            }
        ],
    )
    original = attempts.read_attempt(paths, EEG_ATTEMPT_ID)

    with pytest.raises(ValueError, match="paper_id"):
        attempts.reconcile_legacy_attempt(
            paths,
            EEG_ATTEMPT_ID,
            lease,
            snapshot_id,
            design_author="eeg-design-author",
            design_path=DESIGN_PATH,
            reviewer="user-approved-design",
            approval_ref=APPROVAL_REF,
            now=NOW,
        )

    assert attempts.read_attempt(paths, EEG_ATTEMPT_ID) == original


def test_reconcile_rejects_tampered_migration_backup(tmp_path: Path):
    paths, lease = migrated_eeg(tmp_path)
    snapshot_id, _candidate = assessed_snapshot(paths)
    backup = paths.root / "v3-backups" / f"{SOURCE_STATE_SHA256}.json"
    backup.write_text("{}\n", encoding="utf-8")
    original = attempts.read_attempt(paths, EEG_ATTEMPT_ID)

    with pytest.raises(ValueError, match="source_state_sha256"):
        attempts.reconcile_legacy_attempt(
            paths,
            EEG_ATTEMPT_ID,
            lease,
            snapshot_id,
            design_author="eeg-design-author",
            design_path=DESIGN_PATH,
            reviewer="user-approved-design",
            approval_ref=APPROVAL_REF,
            now=NOW,
        )

    assert attempts.read_attempt(paths, EEG_ATTEMPT_ID) == original
