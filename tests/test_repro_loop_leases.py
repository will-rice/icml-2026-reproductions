"""Tests for fenced schema-v6 resource leases and metered-cost reservations."""

from datetime import datetime, timedelta, timezone
import importlib.util
import math
from pathlib import Path
import sys
import threading

import pytest


SCRIPTS = (
    Path(__file__).resolve().parents[1] / "skills" / "icml-repro-loop" / "scripts"
)
TTL = timedelta(minutes=5)
WORK_TTL = timedelta(hours=2)


def load_module(name: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def store():
    return load_module("store")


@pytest.fixture
def leases():
    load_module("store")
    return load_module("leases")


@pytest.fixture
def paths(tmp_path, store):
    value = store.StatePaths(tmp_path / "repro-loop.json")
    store.atomic_json_write(value.index, store.new_index(), store.validate_index)
    return value


@pytest.fixture
def now():
    return datetime(2026, 7, 24, 12, 0, tzinfo=timezone.utc)


def acquire_with_barrier(leases, paths, first_resource, second_resource, now):
    barrier = threading.Barrier(2)
    acquired = []
    errors = []

    def acquire(resource, owner, attempt_id):
        try:
            barrier.wait()
            acquired.append(
                leases.acquire_lease(
                    paths, resource, owner, attempt_id, now, TTL
                )
            )
        except BaseException as error:  # pragma: no cover - asserted below
            errors.append(error)

    threads = [
        threading.Thread(
            target=acquire, args=(first_resource, "worker-1", "a1")
        ),
        threading.Thread(
            target=acquire, args=(second_resource, "worker-2", "a2")
        ),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert errors == []
    return acquired


def register_attempt(store, paths, attempt_id, now, section="attempts"):
    attempt = {
        "attempt_id": attempt_id,
        "paper_id": f"paper-{attempt_id}",
        "phase": "implementing",
        "updated_at": now.isoformat(),
    }
    store.atomic_json_write(paths.attempt(attempt_id), attempt, store.validate_attempt)
    with store.locked_json(paths.index, store.validate_index) as index:
        index[section][attempt_id] = {
            "path": f"repro-loop/attempts/{attempt_id}.json",
            "paper_id": attempt["paper_id"],
            "phase": "implementing",
            "updated_at": now.isoformat(),
        }


def test_different_attempt_writers_acquire_concurrently(leases, paths, now):
    first, second = acquire_with_barrier(
        leases, paths, "attempt:a1", "attempt:a2", now
    )
    assert first.resource != second.resource


def test_different_spaces_publish_concurrently_but_same_space_serializes(
    leases, paths, now
):
    first, second = acquire_with_barrier(
        leases,
        paths,
        "space:hf--org--paper-a",
        "space:hf--org--paper-b",
        now,
    )
    assert first.resource != second.resource
    with pytest.raises(leases.LeaseBusy):
        leases.acquire_lease(
            paths,
            "space:hf--org--paper-a",
            "worker-3",
            "a3",
            now,
            TTL,
        )


def test_expired_lease_takeover_increments_fencing_token(leases, paths, now):
    old = leases.acquire_lease(
        paths, "attempt:a1", "worker-1", "a1", now, TTL
    )
    new = leases.acquire_lease(
        paths, "attempt:a1", "worker-2", "a1", now + TTL, TTL
    )
    assert new.fencing_token == old.fencing_token + 1
    with pytest.raises(leases.StaleFence):
        leases.assert_fence(paths, old, now + TTL)
    leases.assert_fence(paths, new, now + TTL)


def test_first_attempt_claim_requires_active_attempt_and_token_zero(
    leases, paths, now, store
):
    register_attempt(store, paths, "a1", now)

    claimed = leases.claim_attempt(paths, "a1", "worker-1", 0, now)

    assert claimed.resource == "attempt:a1"
    assert claimed.attempt_id == "a1"
    assert claimed.owner == "worker-1"
    assert claimed.fencing_token == 1
    assert claimed.expires_at == (now + WORK_TTL).isoformat()


def test_expired_attempt_reclaim_increments_expected_predecessor_token(
    leases, paths, now, store
):
    register_attempt(store, paths, "a1", now)
    predecessor = leases.acquire_lease(
        paths, "attempt:a1", "worker-1", "a1", now, TTL
    )

    claimed = leases.claim_attempt(
        paths, "a1", "worker-2", predecessor.fencing_token, now + TTL
    )

    assert claimed.owner == "worker-2"
    assert claimed.fencing_token == predecessor.fencing_token + 1
    assert claimed.expires_at == (now + TTL + WORK_TTL).isoformat()

    register_attempt(store, paths, "a2", now)
    released_predecessor = leases.acquire_lease(
        paths, "attempt:a2", "worker-1", "a2", now, TTL
    )
    leases.release_lease(paths, released_predecessor, now + timedelta(minutes=1))
    released_claim = leases.claim_attempt(
        paths,
        "a2",
        "worker-2",
        released_predecessor.fencing_token,
        now + timedelta(minutes=1),
    )
    assert released_claim.fencing_token == released_predecessor.fencing_token + 1


def test_simultaneous_attempt_reclaim_has_one_winner(leases, paths, now, store):
    register_attempt(store, paths, "a1", now)
    predecessor = leases.acquire_lease(
        paths, "attempt:a1", "worker-1", "a1", now, TTL
    )
    barrier = threading.Barrier(2)
    claimed = []
    errors = []

    def reclaim(owner):
        try:
            barrier.wait()
            claimed.append(
                leases.claim_attempt(
                    paths,
                    "a1",
                    owner,
                    predecessor.fencing_token,
                    now + TTL,
                )
            )
        except BaseException as error:  # pragma: no cover - asserted below
            errors.append(error)

    threads = [
        threading.Thread(target=reclaim, args=("worker-2",)),
        threading.Thread(target=reclaim, args=("worker-3",)),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert len(claimed) == 1
    assert claimed[0].fencing_token == 2
    assert len(errors) == 1
    assert isinstance(errors[0], leases.StaleFence)


def test_attempt_claim_rejects_wrong_predecessor_and_live_lease(
    leases, paths, now, store
):
    register_attempt(store, paths, "a1", now)
    live = leases.acquire_lease(
        paths, "attempt:a1", "worker-1", "a1", now, TTL
    )

    with pytest.raises(leases.StaleFence):
        leases.claim_attempt(paths, "a1", "worker-2", 0, now + TTL)
    with pytest.raises(leases.LeaseBusy):
        leases.claim_attempt(
            paths, "a1", "worker-2", live.fencing_token, now
        )


def test_attempt_claim_enforces_predecessor_chronology_boundaries(
    leases, paths, now, store
):
    register_attempt(store, paths, "a1", now)
    expiring = leases.acquire_lease(
        paths, "attempt:a1", "worker-1", "a1", now, TTL
    )
    with pytest.raises(ValueError, match="now"):
        leases.claim_attempt(
            paths,
            "a1",
            "worker-2",
            expiring.fencing_token,
            now - timedelta(microseconds=1),
        )
    expiry_claim = leases.claim_attempt(
        paths, "a1", "worker-2", expiring.fencing_token, now + TTL
    )
    assert expiry_claim.acquired_at == (now + TTL).isoformat()

    register_attempt(store, paths, "a2", now)
    releasing = leases.acquire_lease(
        paths, "attempt:a2", "worker-1", "a2", now, TTL
    )
    released_at = now + timedelta(minutes=1)
    leases.release_lease(paths, releasing, released_at)
    with pytest.raises(ValueError, match="now"):
        leases.claim_attempt(
            paths,
            "a2",
            "worker-2",
            releasing.fencing_token,
            released_at - timedelta(microseconds=1),
        )
    release_claim = leases.claim_attempt(
        paths, "a2", "worker-2", releasing.fencing_token, released_at
    )
    assert release_claim.acquired_at == released_at.isoformat()


def test_attempt_claim_rejects_missing_history_and_cross_attempt_identity(
    leases, paths, now, store
):
    with pytest.raises(ValueError, match="attempt_id"):
        leases.claim_attempt(paths, "missing", "worker", 0, now)

    register_attempt(store, paths, "history", now, section="history")
    with pytest.raises(ValueError, match="attempt_id"):
        leases.claim_attempt(paths, "history", "worker", 0, now)

    register_attempt(store, paths, "a1", now)
    forged = {
        "resource": "attempt:a2",
        "owner": "worker-1",
        "attempt_id": "a2",
        "acquired_at": now.isoformat(),
        "expires_at": (now + TTL).isoformat(),
        "fencing_token": 1,
        "released_at": None,
    }
    store.atomic_json_write(
        paths.resource_lease("attempt:a1"), forged, leases.validate_lease
    )
    with pytest.raises(leases.StaleFence):
        leases.claim_attempt(paths, "a1", "worker-2", 1, now + TTL)


def test_renew_extends_live_lease_without_changing_fence(leases, paths, now):
    lease = leases.acquire_lease(
        paths, "space:hf--org--paper-a", "worker-1", "a1", now, TTL
    )
    renewed = leases.renew_lease(paths, lease, now + timedelta(minutes=1), TTL)
    assert renewed.fencing_token == lease.fencing_token
    assert renewed.expires_at == (now + timedelta(minutes=6)).isoformat()
    with pytest.raises(leases.StaleFence):
        leases.renew_lease(paths, lease, now + timedelta(minutes=6), TTL)


def test_attempt_renew_uses_work_ttl_without_changing_fence(
    leases, paths, now, store
):
    register_attempt(store, paths, "a1", now)
    claimed = leases.claim_attempt(paths, "a1", "worker-1", 0, now)

    renewed = leases.renew_attempt(
        paths, claimed, now + timedelta(hours=1)
    )

    assert renewed.fencing_token == claimed.fencing_token
    assert renewed.acquired_at == claimed.acquired_at
    assert renewed.expires_at == (now + timedelta(hours=3)).isoformat()


def test_attempt_renew_rejects_wrong_owner_released_and_expired_leases(
    leases, paths, now, store
):
    register_attempt(store, paths, "a1", now)
    claimed = leases.claim_attempt(paths, "a1", "worker-1", 0, now)
    wrong_owner = leases.Lease(
        resource=claimed.resource,
        owner="worker-2",
        attempt_id=claimed.attempt_id,
        acquired_at=claimed.acquired_at,
        expires_at=claimed.expires_at,
        fencing_token=claimed.fencing_token,
    )
    with pytest.raises(leases.StaleFence):
        leases.renew_attempt(paths, wrong_owner, now + timedelta(hours=1))

    released = leases.release_lease(paths, claimed, now + timedelta(hours=1))
    with pytest.raises(leases.StaleFence):
        leases.renew_attempt(paths, released, now + timedelta(hours=1))

    register_attempt(store, paths, "a2", now)
    expiring = leases.claim_attempt(paths, "a2", "worker-2", 0, now)
    with pytest.raises(leases.StaleFence):
        leases.renew_attempt(paths, expiring, now + WORK_TTL)


def test_renew_lease_rejects_backdating_and_never_shortens_expiry(
    leases, paths, now, store
):
    register_attempt(store, paths, "a1", now)
    claimed = leases.claim_attempt(paths, "a1", "worker-1", 0, now)

    with pytest.raises(ValueError, match="now"):
        leases.renew_attempt(
            paths, claimed, now - timedelta(microseconds=1)
        )

    boundary = leases.renew_attempt(paths, claimed, now)
    assert boundary.expires_at == claimed.expires_at

    extended = leases.renew_attempt(
        paths, boundary, now + timedelta(hours=1)
    )
    with pytest.raises(ValueError, match="now"):
        leases.renew_attempt(
            paths, extended, now + timedelta(minutes=30)
        )
    assert leases.assert_fence(paths, extended, now).expires_at == extended.expires_at


def test_attempt_renew_rejects_history_missing_and_cross_paper_references(
    leases, paths, now, store
):
    register_attempt(store, paths, "history", now)
    history_lease = leases.claim_attempt(paths, "history", "worker-1", 0, now)
    with store.locked_json(paths.index, store.validate_index) as index:
        index["history"]["history"] = index["attempts"].pop("history")
    with pytest.raises(ValueError, match="attempt_id"):
        leases.renew_attempt(paths, history_lease, now + timedelta(minutes=1))

    register_attempt(store, paths, "missing", now)
    missing_lease = leases.claim_attempt(paths, "missing", "worker-1", 0, now)
    with store.locked_json(paths.index, store.validate_index) as index:
        del index["attempts"]["missing"]
    with pytest.raises(ValueError, match="attempt_id"):
        leases.renew_attempt(paths, missing_lease, now + timedelta(minutes=1))

    register_attempt(store, paths, "cross-paper", now)
    cross_paper_lease = leases.claim_attempt(
        paths, "cross-paper", "worker-1", 0, now
    )
    with store.locked_json(paths.index, store.validate_index) as index:
        index["attempts"]["cross-paper"]["paper_id"] = "other-paper"
    with pytest.raises(ValueError, match="paper_id"):
        leases.renew_attempt(
            paths, cross_paper_lease, now + timedelta(minutes=1)
        )


def test_archive_wins_attempt_lease_race_and_waiting_renewal_fails(
    leases, paths, now, store
):
    register_attempt(store, paths, "a1", now)
    lease = leases.claim_attempt(paths, "a1", "worker-1", 0, now)
    lease_path = paths.resource_lease("attempt:a1")
    archive_holds_lease = threading.Event()
    finish_archive = threading.Event()
    renewal_started = threading.Event()
    renewal_errors = []

    def archive():
        with store._exclusive_lock(lease_path):
            archive_holds_lease.set()
            assert finish_archive.wait(timeout=5)
            with store._exclusive_lock(paths.index):
                index = store.read_json(paths.index)
                store.validate_index(index)
                index["history"]["a1"] = index["attempts"].pop("a1")
                store._atomic_json_write(paths.index, index)

    def renew():
        renewal_started.set()
        try:
            leases.renew_attempt(paths, lease, now + timedelta(minutes=1))
        except BaseException as error:  # pragma: no cover - asserted below
            renewal_errors.append(error)

    archive_thread = threading.Thread(target=archive)
    archive_thread.start()
    assert archive_holds_lease.wait(timeout=5)
    renewal_thread = threading.Thread(target=renew)
    renewal_thread.start()
    assert renewal_started.wait(timeout=5)
    finish_archive.set()
    archive_thread.join(timeout=5)
    renewal_thread.join(timeout=5)

    assert not archive_thread.is_alive()
    assert not renewal_thread.is_alive()
    assert len(renewal_errors) == 1
    assert isinstance(renewal_errors[0], ValueError)
    assert str(renewal_errors[0]) == "attempt_id"


def test_release_allows_takeover_and_old_owner_cannot_release_successor(
    leases, paths, now
):
    old = leases.acquire_lease(
        paths, "attempt:a1", "worker-1", "a1", now, TTL
    )
    released = leases.release_lease(paths, old, now + timedelta(minutes=1))
    assert released.released_at == (now + timedelta(minutes=1)).isoformat()
    new = leases.acquire_lease(
        paths,
        "attempt:a1",
        "worker-2",
        "a1",
        now + timedelta(minutes=1),
        TTL,
    )
    assert new.fencing_token == old.fencing_token + 1
    with pytest.raises(leases.StaleFence):
        leases.release_lease(paths, old, now + timedelta(minutes=2))


def test_release_rejects_owner_at_exact_expiry_before_takeover(
    leases, paths, now
):
    lease = leases.acquire_lease(
        paths, "attempt:a1", "worker-1", "a1", now, TTL
    )

    with pytest.raises(leases.StaleFence):
        leases.release_lease(paths, lease, now + TTL)


def test_expire_stale_leases_marks_only_expired_live_records(leases, paths, now):
    stale = leases.acquire_lease(
        paths, "attempt:a1", "worker-1", "a1", now, TTL
    )
    live = leases.acquire_lease(
        paths, "attempt:a2", "worker-2", "a2", now, TTL * 2
    )
    expired = leases.expire_stale_leases(paths, now + TTL)
    assert expired == [stale.resource]
    with pytest.raises(leases.StaleFence):
        leases.assert_fence(paths, stale, now + TTL)
    leases.assert_fence(paths, live, now + TTL)


def test_subscription_agents_reserve_exactly_zero_metered_cost(
    leases, paths, now
):
    for provider in ("codex-subscription", "antigravity-subscription"):
        reservation = leases.reserve_metered_cost(
            paths, "a1", provider, 9.0, now
        )
        assert reservation.amount_usd == 0.0


@pytest.mark.parametrize("amount", [-0.01, math.inf, -math.inf, math.nan])
def test_metered_reservation_rejects_invalid_amount(
    leases, paths, now, amount
):
    with pytest.raises(ValueError, match="amount_usd"):
        leases.reserve_metered_cost(paths, "a1", "paid-api", amount, now)


def test_metered_reservations_enforce_per_paper_and_global_limits(
    leases, paths, now
):
    first = leases.reserve_metered_cost(paths, "a1", "paid-api", 6.0, now)
    with pytest.raises(leases.CostLimitExceeded, match="paper"):
        leases.reserve_metered_cost(paths, "a1", "other-api", 4.01, now)
    with pytest.raises(leases.CostLimitExceeded, match="global"):
        leases.reserve_metered_cost(paths, "a2", "paid-api", 4.01, now)
    assert first.amount_usd == 6.0


def test_metered_reservations_sum_distinct_attempts_for_the_same_paper(
    leases, paths, now, store
):
    index = store.read_json(paths.index)
    index["resource_limits"]["metered_api_reserved_usd"] = 20.0
    for attempt_id in ("a1", "a2"):
        index["attempts"][attempt_id] = {
            "path": f"repro-loop/attempts/{attempt_id}.json",
            "paper_id": "paper-1",
            "phase": "implementing",
            "updated_at": now.isoformat(),
        }
    store.atomic_json_write(paths.index, index, store.validate_index)

    leases.reserve_metered_cost(paths, "a1", "paid-api", 6.0, now)
    with pytest.raises(leases.CostLimitExceeded, match="paper"):
        leases.reserve_metered_cost(paths, "a2", "other-api", 4.01, now)


def test_reconciliation_releases_reservation_and_records_actual_cost(
    leases, paths, now
):
    reservation = leases.reserve_metered_cost(
        paths, "a1", "paid-api", 6.0, now
    )
    reconciled = leases.reconcile_metered_cost(
        paths, reservation, 2.5, now + timedelta(minutes=1)
    )
    assert reconciled.amount_usd == 6.0
    assert reconciled.actual_amount_usd == 2.5
    assert reconciled.reconciled_at == (now + timedelta(minutes=1)).isoformat()
    replacement = leases.reserve_metered_cost(
        paths, "a2", "paid-api", 7.5, now + timedelta(minutes=1)
    )
    assert replacement.amount_usd == 7.5
    with pytest.raises(leases.StaleFence):
        leases.reconcile_metered_cost(
            paths, reservation, 2.5, now + timedelta(minutes=2)
        )


def test_reconciliation_rejects_actual_cost_above_reserved(
    leases, paths, now
):
    reservation = leases.reserve_metered_cost(
        paths, "a1", "paid-api", 2.0, now
    )
    with pytest.raises(ValueError, match="actual_amount_usd"):
        leases.reconcile_metered_cost(paths, reservation, 2.01, now)


def test_reconciliation_rejects_forged_identity_with_equal_fencing_token(
    leases, paths, now
):
    victim = leases.reserve_metered_cost(
        paths, "a1", "victim-api", 2.0, now
    )
    attacker = leases.reserve_metered_cost(
        paths, "a2", "attacker-api", 2.0, now
    )
    forged = leases.MeteredCostReservation(
        attempt_id=victim.attempt_id,
        paper_id=attacker.paper_id,
        provider=victim.provider,
        amount_usd=attacker.amount_usd,
        reserved_at=attacker.reserved_at,
        fencing_token=attacker.fencing_token,
    )

    with pytest.raises(leases.StaleFence):
        leases.reconcile_metered_cost(
            paths, forged, 1.0, now + timedelta(minutes=1)
        )


def test_reconciliation_rejects_time_before_reservation(leases, paths, now):
    reservation = leases.reserve_metered_cost(
        paths, "a1", "paid-api", 2.0, now
    )

    with pytest.raises(ValueError, match="reconciled_at"):
        leases.reconcile_metered_cost(
            paths, reservation, 1.0, now - timedelta(microseconds=1)
        )


def test_reservation_validation_rejects_reconciliation_before_reservation(
    leases, now
):
    value = {
        "attempt_id": "a1",
        "paper_id": "paper-1",
        "provider": "paid-api",
        "amount_usd": 2.0,
        "reserved_at": now.isoformat(),
        "fencing_token": 1,
        "actual_amount_usd": 1.0,
        "reconciled_at": (now - timedelta(microseconds=1)).isoformat(),
        "cumulative_actual_usd": 1.0,
    }

    with pytest.raises(ValueError, match="reconciled_at"):
        leases.validate_reservation(value)
