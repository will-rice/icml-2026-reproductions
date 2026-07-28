"""Bounded paper-pool admission and independently persisted judgments."""

from __future__ import annotations

import copy
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import hashlib
from itertools import chain
import json
import math
from pathlib import Path
import sys
from uuid import uuid4


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import attestations  # noqa: E402
import attempts  # noqa: E402
import leases  # noqa: E402
import publication_policy  # noqa: E402
import score_rate  # noqa: E402
import state  # noqa: E402
import store  # noqa: E402


ADMISSION_LEASE_TTL = timedelta(minutes=5)
SNAPSHOT_KEYS = {
    "snapshot_id",
    "fetched_at",
    "source_revision",
    "candidates",
    "queued_submissions",
    "tagged_spaces",
    "verdicts",
}
LIVE_SNAPSHOT_KEYS = SNAPSHOT_KEYS | {"sources", "assessments", "spaces"}
JUDGMENT_KEYS = {
    "attempt_id",
    "paper_id",
    "space_id",
    "submitted_sha",
    "attempt_number",
    "target_claims",
    "poll_limit",
    "poll_deadline",
    "polls",
    "raw_verdict",
    "normalized_verdict",
    "source_revision",
    "verdict_at",
    "created_at",
    "updated_at",
}
OFFICIAL_VERDICT_STATUSES = {
    "verified",
    "falsified",
    "toy",
    "inconclusive",
}


@dataclass(frozen=True, slots=True)
class SchedulerAssignment:
    """One admitted paper and the fences needed by its authoritative writer."""

    attempt_id: str
    paper_id: str
    candidate_lease: leases.Lease
    writer_lease: leases.Lease


@dataclass(frozen=True, slots=True)
class SchedulerReport:
    """The successful assignments created by one bounded scheduler pass."""

    assignments: tuple[SchedulerAssignment, ...]

    @property
    def created_attempt_ids(self) -> tuple[str, ...]:
        return tuple(assignment.attempt_id for assignment in self.assignments)

    @property
    def paper_ids(self) -> tuple[str, ...]:
        return tuple(assignment.paper_id for assignment in self.assignments)


class NoEligiblePaper(RuntimeError):
    """Raised when one persistent owner has no paper it can claim."""


class OwnerBusy(RuntimeError):
    """Raised when one persistent owner already has a live paper lease."""


@dataclass(frozen=True, slots=True)
class PaperOwnerAssignment:
    """One fenced assignment to a persistent paper owner."""

    attempt_id: str
    paper_id: str
    writer_lease: leases.Lease
    reclaimed: bool


def scheduler_pass(
    paths: store.StatePaths,
    snapshot_id: str,
    now: datetime,
    adopt_space_id: str | None = None,
) -> SchedulerReport:
    """Admit ranked eligible candidates without exceeding runnable capacity."""
    observed_at = _datetime(now)
    snapshot = read_fresh_snapshot(paths, snapshot_id, observed_at)
    leases.expire_stale_leases(paths, observed_at)
    attempts.recover_transactions(paths)
    index = store.read_json(paths.index)
    store.validate_index(index)
    vacancies = index["max_runnable_attempts"] - len(
        attempts.runnable_attempt_ids(paths)
    )
    claimed = _claimed_paper_ids(paths, index, snapshot, observed_at)
    if adopt_space_id is not None:
        claimed.discard(
            _adoptable_owned_tagged_paper(index, snapshot, adopt_space_id)
        )
    candidates = rank_eligible_candidates(snapshot, claimed)
    return _admit_up_to(
        paths,
        candidates,
        max(0, vacancies),
        snapshot_id,
        observed_at,
    )


def claim_next(
    paths: store.StatePaths,
    snapshot_id: str,
    owner: str,
    now: datetime,
    reclaim_attempt_id: str | None = None,
) -> PaperOwnerAssignment:
    """Atomically give one persistent owner one new or reclaimable attempt."""
    observed_at = _datetime(now)
    snapshot = read_fresh_snapshot(paths, snapshot_id, observed_at)
    _require_assessed_snapshot(snapshot)
    leases.expire_stale_leases(paths, observed_at)
    attempts.recover_transactions(paths)
    with leases.hold_owner_claim(paths, owner):
        _require_owner_available(paths, owner, observed_at)

        if reclaim_attempt_id is not None:
            attempt = attempts.read_attempt(paths, reclaim_attempt_id)
            if attempt["phase"] != "blocked":
                raise ValueError("phase")
            _assessment_for_paper(snapshot, attempt["paper_id"])
            prior = _attempt_lease(paths, reclaim_attempt_id)
            expected = 0 if prior is None else prior.fencing_token
            writer = leases._claim_attempt_locked(
                paths,
                reclaim_attempt_id,
                owner,
                expected,
                observed_at,
            )
            return PaperOwnerAssignment(
                reclaim_attempt_id,
                attempt["paper_id"],
                writer,
                True,
            )

        index = store.read_json(paths.index)
        store.validate_index(index)
        claimed = _claimed_paper_ids(paths, index, snapshot, observed_at)
        candidates = [
            candidate
            for candidate in rank_eligible_candidates(snapshot, claimed)
            if _candidate_matches_assessment(snapshot, candidate)
        ]
        report = _admit_up_to(
            paths,
            candidates,
            1,
            snapshot_id,
            observed_at,
            owner=owner,
        )
        if not report.assignments:
            raise NoEligiblePaper("no eligible paper")
        assignment = report.assignments[0]
        return PaperOwnerAssignment(
            assignment.attempt_id,
            assignment.paper_id,
            assignment.writer_lease,
            False,
        )


def _require_assessed_snapshot(snapshot: dict) -> None:
    import refresh

    assessments = snapshot.get("assessments")
    if (
        type(assessments) is not dict
        or set(assessments) != refresh.PERSISTED_ASSESSMENT_KEYS
        or type(assessments["matched_paper_ids"]) is not list
    ):
        raise ValueError("assessments")
    for paper_id in assessments["matched_paper_ids"]:
        _assessment_for_paper(snapshot, paper_id)


def _assessment_for_paper(snapshot: dict, paper_id: str) -> dict:
    import refresh

    return refresh.assessment_record_for_snapshot(snapshot, paper_id)


def _candidate_matches_assessment(snapshot: dict, candidate: dict) -> bool:
    try:
        assessment = _assessment_for_paper(snapshot, candidate["paper_id"])
    except ValueError:
        return False
    return all(
        candidate.get(field) == assessment[field]
        for field in assessment
    )


def read_fresh_snapshot(
    paths: store.StatePaths, snapshot_id: str, now: datetime
) -> dict:
    """Read one referenced immutable snapshot within the admission window."""
    store.validate_id(snapshot_id)
    observed_at = _datetime(now)
    index = store.read_json(paths.index)
    store.validate_index(index)
    reference = index["snapshots"].get(snapshot_id)
    if reference is None:
        raise ValueError("snapshot_id")
    path = paths.index.parent / reference
    expected = paths.root / "snapshots" / f"{snapshot_id}.json"
    if path != expected:
        raise ValueError("snapshot")
    snapshot = store.read_json(path)
    _validate_snapshot(snapshot)
    if snapshot["snapshot_id"] != snapshot_id:
        raise ValueError("snapshot_id")
    import refresh

    payload = {key: value for key, value in snapshot.items() if key != "snapshot_id"}
    if refresh.canonical_snapshot_id(payload) != snapshot_id:
        raise ValueError("snapshot_id")
    fetched_at = _parse(snapshot["fetched_at"], "fetched_at")
    if fetched_at > observed_at or observed_at - fetched_at > ADMISSION_LEASE_TTL:
        raise ValueError("fetched_at")
    return snapshot


def rank_eligible_candidates(
    snapshot: dict, claimed_paper_ids: set[str] | None = None
) -> list[dict]:
    """Return score-rate-assessed admission candidates in deterministic order."""
    claimed = set() if claimed_paper_ids is None else claimed_paper_ids
    eligible = _eligible_candidates(
        snapshot,
        claimed_paper_ids=claimed,
        require_score_rate=True,
    )
    return sorted(
        eligible,
        key=score_rate.ranking_key,
    )


def legacy_reconciliation_candidates(snapshot: dict) -> list[dict]:
    """Return eligible legacy candidates without making a new admission."""
    return _eligible_candidates(
        snapshot,
        claimed_paper_ids=set(),
        require_score_rate=False,
    )


def _eligible_candidates(
    snapshot: dict,
    *,
    claimed_paper_ids: set[str],
    require_score_rate: bool,
) -> list[dict]:
    _validate_snapshot(snapshot)
    eligible = []
    for candidate in snapshot["candidates"]:
        paper_id = _identity(candidate.get("paper_id"), "paper_id")
        legacy_score = candidate.get("score")
        estimated_cost = candidate.get("estimated_api_cost_usd")
        if (
            paper_id in claimed_paper_ids
            or type(candidate.get("target_claims")) is not list
            or len(candidate["target_claims"]) < 2
            or not _has_current_claim_bindings(candidate)
            or type(candidate.get("upstream_revision")) is not str
            or not candidate["upstream_revision"]
            or candidate.get("artifact_access") is not True
            or candidate.get("cpu_only") is not True
            or candidate.get("safety_blocker") is not None
            or candidate.get("licensing_blocker") is not None
            or (
                not require_score_rate
                and (
                    type(legacy_score) not in {int, float}
                    or not math.isfinite(legacy_score)
                )
            )
            or type(estimated_cost) not in {int, float}
            or not math.isfinite(estimated_cost)
            or estimated_cost < 0
            or estimated_cost > 10
        ):
            continue
        if require_score_rate:
            try:
                score_rate.validate_envelope(
                    candidate.get("score_rate"), candidate.get("live_claims")
                )
            except ValueError:
                continue
        state.validate_target_claims(candidate["target_claims"])
        eligible.append(copy.deepcopy(candidate))
    return eligible


def _has_current_claim_bindings(candidate: dict) -> bool:
    """Require admission inputs to preserve their live challenge claim identity."""
    import refresh

    target_claims = candidate.get("target_claims")
    if type(target_claims) is not list or any(
        type(claim) is not str or not claim for claim in target_claims
    ):
        return False
    return refresh._valid_claim_bindings(
        candidate.get("claim_bindings"), target_claims, candidate.get("live_claims")
    )


def watch_attempt(
    paths: store.StatePaths,
    attempt_id: str,
    lease: leases.Lease,
    poll_limit: int,
    poll_deadline: datetime,
    now: datetime,
) -> dict:
    """Atomically enter judging with a bounded authority attestation."""
    observed_at = _datetime(now)
    deadline = _datetime(poll_deadline)
    if type(poll_limit) is not int or poll_limit <= 0:
        raise ValueError("poll_limit")
    if deadline < observed_at:
        raise ValueError("poll_deadline")
    attempt = attempts.read_attempt(paths, attempt_id)
    if attempt["phase"] != "submitted":
        raise ValueError("phase")
    target_claims = attempt.get("target_claims")
    state.validate_target_claims(target_claims)
    space_id = _identity(attempt.get("space_id"), "space_id")
    submitted_sha = _identity(attempt.get("deployed_sha"), "deployed_sha")
    improvement_attempts = attempt.get("improvement_attempts", 0)
    if (
        type(improvement_attempts) is not int
        or improvement_attempts < 0
    ):
        raise ValueError("improvement_attempts")
    attempt_number = improvement_attempts + 1
    submission = _authoritative_attestation(
        paths, attempt, "submission", "submitted"
    )
    timestamp = _timestamp(observed_at)
    judgment = {
        "attempt_id": attempt_id,
        "paper_id": attempt["paper_id"],
        "space_id": space_id,
        "submitted_sha": submitted_sha,
        "attempt_number": attempt_number,
        "target_claims": copy.deepcopy(target_claims),
        "poll_limit": poll_limit,
        "poll_deadline": _timestamp(deadline),
        "polls": [],
        "raw_verdict": None,
        "normalized_verdict": None,
        "source_revision": None,
        "verdict_at": None,
        "created_at": timestamp,
        "updated_at": timestamp,
    }
    authority_payload = {
        "submission_attestation_id": submission["attestation_id"],
        "poll_limit": poll_limit,
        "poll_deadline": judgment["poll_deadline"],
        "space_id": space_id,
        "space_sha": submitted_sha,
    }
    authority_record = {
        "kind": "authority-audit",
        "attempt_id": attempt_id,
        "attempt_number": attempt_number,
        "observed_at": timestamp,
        "source_commit": submission["source_commit"],
        "payload_sha256": _sha256_json(authority_payload),
        **authority_payload,
    }
    authority_id = attestations.persist(paths, authority_record)
    path = paths.judgment(attempt_id)
    transaction_targets = []
    if path.exists():
        previous = store.read_json(path)
        validate_judgment_record(previous)
        _require_judgment(previous, attempt_id)
        if previous["attempt_number"] != attempt_number - 1:
            raise ValueError("judgment")
        archive_path = paths.judgment_archive(
            attempt_id, previous["attempt_number"]
        )
        if archive_path.exists() and store.read_json(archive_path) != previous:
            raise ValueError("judgment")
        transaction_targets.append(
            (archive_path, previous, validate_judgment_record)
        )
    transaction_targets.append((path, judgment, validate_judgment_record))
    identity_leases = []
    identity_resources = sorted(
        (
            ("space_id", f"judgment-space:{space_id}"),
            ("submitted_sha", f"submitted-sha:{submitted_sha}"),
        ),
        key=lambda item: item[1],
    )
    try:
        for field, resource in identity_resources:
            try:
                identity_leases.append(
                    leases.acquire_lease(
                        paths,
                        resource,
                        f"judgment-{attempt_id}",
                        attempt_id,
                        observed_at,
                        ADMISSION_LEASE_TTL,
                    )
                )
            except leases.LeaseBusy as error:
                raise ValueError(field) from error
        with ExitStack() as identity_fences:
            for identity_lease in identity_leases:
                identity_fences.enter_context(
                    leases.hold_fence(paths, identity_lease, observed_at)
                )
            _assert_unique_submission(paths, attempt_id, space_id, submitted_sha)
            attempts.transition_attested(
                paths,
                attempt_id,
                "judging",
                authority_id,
                {},
                lease,
                observed_at,
                transaction_targets=transaction_targets,
            )
    finally:
        for identity_lease in reversed(identity_leases):
            try:
                leases.release_lease(paths, identity_lease, observed_at)
            except leases.StaleFence:
                # A successor can acquire only after the held write completes.
                pass
    return store.read_json(path)


def record_poll(
    paths: store.StatePaths,
    attempt_id: str,
    lease: leases.Lease,
    status: str,
    now: datetime,
) -> dict:
    """Append one observation within a judgment's finite poll budget."""
    status = _identity(status, "status")
    observed_at = _datetime(now)
    path = paths.judgment(attempt_id)
    with _hold_attempt_fence(paths, attempt_id, lease, observed_at):
        with store.locked_json(path, validate_judgment_record) as judgment:
            _require_judgment(judgment, attempt_id)
            if judgment["raw_verdict"] is not None:
                raise ValueError("verdict")
            if len(judgment["polls"]) >= judgment["poll_limit"]:
                raise ValueError("poll_limit")
            if observed_at > _parse(judgment["poll_deadline"], "poll_deadline"):
                raise ValueError("poll_deadline")
            if observed_at < _parse(judgment["created_at"], "created_at"):
                raise ValueError("now")
            timestamp = _timestamp(observed_at)
            judgment["polls"].append({"at": timestamp, "status": status})
            judgment["updated_at"] = timestamp
    return store.read_json(path)


def validate_judgment_record(judgment: dict) -> None:
    """Validate the complete independent judgment schema."""
    store.validate_judgment(judgment)
    if set(judgment) != JUDGMENT_KEYS:
        raise ValueError("judgment")
    for field in ("attempt_id", "paper_id", "space_id", "submitted_sha"):
        _identity(judgment[field], field)
    if type(judgment["attempt_number"]) is not int or judgment["attempt_number"] < 1:
        raise ValueError("attempt_number")
    state.validate_target_claims(judgment["target_claims"])
    if type(judgment["poll_limit"]) is not int or judgment["poll_limit"] <= 0:
        raise ValueError("poll_limit")
    deadline = _parse(judgment["poll_deadline"], "poll_deadline")
    created = _parse(judgment["created_at"], "created_at")
    updated = _parse(judgment["updated_at"], "updated_at")
    if deadline < created or updated < created:
        raise ValueError("timestamp")
    polls = judgment["polls"]
    if type(polls) is not list or len(polls) > judgment["poll_limit"]:
        raise ValueError("polls")
    previous = created
    for poll in polls:
        if type(poll) is not dict or set(poll) != {"at", "status"}:
            raise ValueError("polls")
        at = _parse(poll["at"], "polls")
        _identity(poll["status"], "polls")
        if at < previous or at > deadline:
            raise ValueError("polls")
        previous = at
    if updated < previous:
        raise ValueError("updated_at")
    values = (
        judgment["raw_verdict"],
        judgment["normalized_verdict"],
        judgment["source_revision"],
        judgment["verdict_at"],
    )
    if all(value is None for value in values):
        if updated != previous:
            raise ValueError("updated_at")
        return
    if any(value is None for value in values):
        raise ValueError("verdict")
    if type(judgment["raw_verdict"]) is not dict:
        raise ValueError("raw_verdict")
    validate_normalized_verdict(
        judgment["normalized_verdict"], judgment["target_claims"]
    )
    _identity(judgment["source_revision"], "source_revision")
    verdict_at = _parse(judgment["verdict_at"], "verdict_at")
    if verdict_at < previous:
        raise ValueError("verdict_at")
    if updated != verdict_at:
        raise ValueError("updated_at")


def validate_normalized_verdict(verdict: object, target_claims: list[str]) -> None:
    """Accept preserved legacy records or exact official-status records."""
    try:
        state.validate_verdict(verdict, target_claims)
        return
    except ValueError:
        pass
    if type(verdict) is not dict or set(verdict) != {"claims"}:
        raise ValueError("verdict")
    claims = verdict["claims"]
    if type(claims) is not list or len(claims) != len(target_claims):
        raise ValueError("verdict")
    for target_claim, claim in zip(target_claims, claims, strict=True):
        if type(claim) is not dict or set(claim) != {
            "target_claim",
            "claim",
            "status",
            "evidence",
        }:
            raise ValueError("verdict")
        if (
            claim["target_claim"] != target_claim
            or type(claim["claim"]) is not str
            or not claim["claim"]
            or claim["status"] not in OFFICIAL_VERDICT_STATUSES
            or type(claim["evidence"]) is not str
        ):
            raise ValueError("verdict")


def _admit_up_to(
    paths: store.StatePaths,
    candidates: list[dict],
    vacancies: int,
    snapshot_id: str,
    now: datetime,
    owner: str | None = None,
) -> SchedulerReport:
    assignments = []
    for candidate in candidates:
        if len(assignments) >= vacancies:
            break
        attempt_id = str(uuid4())
        assignment_owner = owner if owner is not None else f"scheduler-{uuid4()}"
        candidate_lease = None
        writer_lease = None
        try:
            candidate_lease = leases.acquire_lease(
                paths,
                f"candidate:{candidate['paper_id']}",
                assignment_owner,
                attempt_id,
                now,
                ADMISSION_LEASE_TTL,
            )
            writer_lease = leases.acquire_lease(
                paths,
                f"attempt:{attempt_id}",
                assignment_owner,
                attempt_id,
                now,
                ADMISSION_LEASE_TTL,
            )
            attempts.create_attempt(
                paths,
                attempt_id,
                candidate,
                writer_lease,
                snapshot_id,
                now,
            )
        except leases.LeaseBusy:
            _release_if_acquired(paths, writer_lease, now)
            _release_if_acquired(paths, candidate_lease, now)
            continue
        except ValueError as error:
            _release_if_acquired(paths, writer_lease, now)
            _release_if_acquired(paths, candidate_lease, now)
            if str(error) == "max_runnable_attempts":
                break
            if str(error) == "paper_id":
                continue
            raise
        assignments.append(
            SchedulerAssignment(
                attempt_id=attempt_id,
                paper_id=candidate["paper_id"],
                candidate_lease=candidate_lease,
                writer_lease=writer_lease,
            )
        )
    return SchedulerReport(tuple(assignments))


def _adoptable_owned_tagged_paper(
    index: dict, snapshot: dict, space_id: str
) -> str:
    """Resolve one explicit untracked Space without weakening duplicate checks."""
    space_id = _identity(space_id, "space_id")
    owner = publication_policy.space_owner(space_id)
    if owner not in publication_policy.ALLOWED_SPACE_OWNERS:
        raise ValueError("space_id")
    matches = {
        record["paper_id"]
        for record in snapshot["tagged_spaces"]
        if record.get("space_id") == space_id
    }
    if len(matches) != 1:
        raise ValueError("space_id")
    paper_id = matches.pop()
    if any(
        reference["paper_id"] == paper_id
        for section in ("attempts", "history")
        for reference in index[section].values()
    ) or any(
        record.get("paper_id") == paper_id for record in index["rejections"]
    ):
        raise ValueError("paper_id")
    owned_spaces = {
        record.get("space_id")
        for record in snapshot["tagged_spaces"]
        if record.get("paper_id") == paper_id
        and publication_policy.space_owner(record.get("space_id")) in
        publication_policy.ALLOWED_SPACE_OWNERS
    }
    if owned_spaces != {space_id}:
        raise ValueError("space_id")
    if any(
        record.get("paper_id") == paper_id
        and publication_policy.space_owner(record.get("space_id")) in
        publication_policy.ALLOWED_SPACE_OWNERS
        for record in snapshot["verdicts"]
    ):
        raise ValueError("verdict")
    return paper_id


def _claimed_paper_ids(
    paths: store.StatePaths,
    index: dict,
    snapshot: dict,
    now: datetime,
) -> set[str]:
    claimed = {
        reference["paper_id"]
        for section in ("attempts", "history")
        for reference in index[section].values()
    }
    claimed.update(
        record["paper_id"]
        for record in index["rejections"]
        if type(record) is dict and type(record.get("paper_id")) is str
    )
    claimed.update(_external_claimed_paper_ids(snapshot))
    lease_directory = paths.root / "leases"
    for path in lease_directory.glob("*.json"):
        value = store.read_json(path)
        leases.validate_lease(value)
        if (
            value["resource"].startswith("candidate:")
            and value["released_at"] is None
            and _parse(value["expires_at"], "expires_at") > now
        ):
            claimed.add(value["resource"].removeprefix("candidate:"))
    return claimed


def _attempt_lease(
    paths: store.StatePaths, attempt_id: str
) -> leases.Lease | None:
    path = paths.resource_lease(f"attempt:{attempt_id}")
    if not path.exists():
        return None
    value = store.read_json(path)
    leases.validate_lease(value)
    return leases.Lease(**value)


def _require_owner_available(
    paths: store.StatePaths, owner: str, now: datetime
) -> None:
    owner = _identity(owner, "owner")
    for path in (paths.root / "leases").glob("*.json"):
        value = store.read_json(path)
        leases.validate_lease(value)
        if (
            value["resource"].startswith("attempt:")
            and value["owner"] == owner
            and value["released_at"] is None
            and _parse(value["expires_at"], "expires_at") > now
        ):
            raise OwnerBusy(owner)


def _external_claimed_paper_ids(snapshot: dict) -> set[str]:
    """Return papers already represented by our publishing account."""
    claimed = set()
    for field in ("queued_submissions", "tagged_spaces", "verdicts"):
        for record in snapshot[field]:
            try:
                owner = publication_policy.space_owner(record.get("space_id"))
            except ValueError:
                continue
            if owner in publication_policy.ALLOWED_SPACE_OWNERS:
                claimed.add(record["paper_id"])
    return claimed


def _validate_snapshot(snapshot: dict) -> None:
    store.validate_snapshot(snapshot)
    if frozenset(snapshot) not in {
        frozenset(SNAPSHOT_KEYS),
        frozenset(LIVE_SNAPSHOT_KEYS),
    }:
        raise ValueError("snapshot")
    _identity(snapshot["source_revision"], "source_revision")
    _parse(snapshot["fetched_at"], "fetched_at")
    if type(snapshot["candidates"]) is not list or any(
        type(candidate) is not dict for candidate in snapshot["candidates"]
    ):
        raise ValueError("candidates")
    candidate_ids = [
        _identity(candidate.get("paper_id"), "paper_id")
        for candidate in snapshot["candidates"]
    ]
    if len(candidate_ids) != len(set(candidate_ids)):
        raise ValueError("candidates")
    for field in ("queued_submissions", "tagged_spaces", "verdicts"):
        records = snapshot[field]
        if type(records) is not list or any(
            type(record) is not dict
            or type(record.get("paper_id")) is not str
            or not record["paper_id"]
            for record in records
        ):
            raise ValueError(field)
    if "sources" in snapshot:
        if (
            type(snapshot["sources"]) is not dict
            or type(snapshot["spaces"]) is not list
        ):
            raise ValueError("sources")


def _write_judgment(path: Path, judgment: dict) -> None:
    validate_judgment_record(judgment)
    store._atomic_json_write(path, judgment)


def _assert_unique_submission(
    paths: store.StatePaths,
    attempt_id: str,
    space_id: str,
    submitted_sha: str,
) -> None:
    judgment_root = paths.root / "judgments"
    for path in chain(
        judgment_root.glob("*.json"),
        (judgment_root / "archive").glob("*.json"),
    ):
        judgment = store.read_json(path)
        validate_judgment_record(judgment)
        if judgment["attempt_id"] == attempt_id:
            continue
        if judgment["space_id"] == space_id:
            raise ValueError("space_id")
        if judgment["submitted_sha"] == submitted_sha:
            raise ValueError("submitted_sha")


def _release_if_acquired(
    paths: store.StatePaths, lease: leases.Lease | None, now: datetime
) -> None:
    if lease is not None:
        leases.release_lease(paths, lease, now)


def _require_judgment(judgment: dict, attempt_id: str) -> None:
    if judgment["attempt_id"] != attempt_id:
        raise ValueError("attempt_id")


def _authoritative_attestation(
    paths: store.StatePaths,
    attempt: dict,
    kind: str,
    phase: str,
) -> dict:
    attempt_number = attempt.get("improvement_attempts", 0) + 1
    path = paths.attestation(kind, attempt["attempt_id"], attempt_number)
    if not path.exists():
        raise ValueError(kind)
    record = store.read_json(path)
    attestations.validate_target(paths, path, record)
    transitions = [
        transition
        for transition in attempt.get("transitions", [])
        if transition.get("to") == phase
    ]
    if (
        record.get("kind") != kind
        or record.get("attempt_id") != attempt["attempt_id"]
        or record.get("attempt_number") != attempt_number
        or not transitions
        or transitions[-1].get("attestation_id")
        != record.get("attestation_id")
    ):
        raise ValueError(kind)
    return record


@contextmanager
def _hold_attempt_fence(
    paths: store.StatePaths,
    attempt_id: str,
    lease: leases.Lease,
    now: datetime,
) -> Iterator[None]:
    if lease.resource != f"attempt:{attempt_id}" or lease.attempt_id != attempt_id:
        raise leases.StaleFence(f"attempt:{attempt_id}")
    with leases.hold_fence(paths, lease, now):
        yield


def _identity(value: object, field: str) -> str:
    if type(value) is not str or not value or value != value.strip():
        raise ValueError(field)
    return value


def _datetime(value: object) -> datetime:
    if (
        not isinstance(value, datetime)
        or value.tzinfo is None
        or value.utcoffset() != timedelta(0)
    ):
        raise ValueError("now")
    return value.astimezone(timezone.utc)


def _timestamp(value: object) -> str:
    return _datetime(value).isoformat()


def _sha256_json(value: object) -> str:
    encoded = json.dumps(
        value, allow_nan=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _parse(value: object, field: str) -> datetime:
    if type(value) is not str:
        raise ValueError(field)
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as error:
        raise ValueError(field) from error
    try:
        return _datetime(parsed)
    except ValueError as error:
        raise ValueError(field) from error
