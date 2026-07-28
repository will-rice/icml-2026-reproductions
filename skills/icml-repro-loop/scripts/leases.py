"""Fenced resource leases and durable metered-cost reservations."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timedelta, timezone
import math
from pathlib import Path
import sys


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import store  # noqa: E402


MAX_PAPER_COST_USD = 10.0
ATTEMPT_WORK_LEASE_TTL = timedelta(hours=2)
SUBSCRIPTION_PROVIDERS = {
    "antigravity",
    "antigravity-subscription",
    "codex",
    "codex-subscription",
}
LEASE_KEYS = {
    "resource",
    "owner",
    "attempt_id",
    "acquired_at",
    "expires_at",
    "fencing_token",
    "released_at",
}
RESERVATION_KEYS = {
    "attempt_id",
    "paper_id",
    "provider",
    "amount_usd",
    "reserved_at",
    "fencing_token",
    "actual_amount_usd",
    "reconciled_at",
    "cumulative_actual_usd",
}


class LeaseBusy(RuntimeError):
    """Raised when a resource has an unexpired, unreleased owner."""


class StaleFence(RuntimeError):
    """Raised when a mutation presents a lease or reservation's old fence."""


StaleLease = StaleFence


class CostLimitExceeded(ValueError):
    """Raised before a reservation would exceed a configured cost ceiling."""


BudgetExceeded = CostLimitExceeded


@dataclass(frozen=True, slots=True)
class Lease:
    resource: str
    owner: str
    attempt_id: str
    acquired_at: str
    expires_at: str
    fencing_token: int
    released_at: str | None = None


@dataclass(frozen=True, slots=True)
class MeteredCostReservation:
    attempt_id: str
    paper_id: str
    provider: str
    amount_usd: float
    reserved_at: str
    fencing_token: int
    actual_amount_usd: float | None = None
    reconciled_at: str | None = None
    cumulative_actual_usd: float = 0.0


def acquire_lease(
    paths: store.StatePaths,
    resource: str,
    owner: str,
    attempt_id: str,
    now: datetime,
    ttl: timedelta,
) -> Lease:
    """Acquire a resource, fencing any released or expired predecessor."""
    resource = _validate_resource(resource)
    owner = _identity(owner, "owner")
    attempt_id = _identity(attempt_id, "attempt_id")
    acquired_at = _timestamp(now)
    expires_at = _timestamp(_datetime(now) + _ttl(ttl))
    path = paths.resource_lease(resource)
    with store._exclusive_lock(path):
        prior = _read_lease(path)
        if (
            prior is not None
            and prior.released_at is None
            and _parse(prior.expires_at) > _datetime(now)
        ):
            raise LeaseBusy(resource)
        lease = Lease(
            resource=resource,
            owner=owner,
            attempt_id=attempt_id,
            acquired_at=acquired_at,
            expires_at=expires_at,
            fencing_token=1 if prior is None else prior.fencing_token + 1,
        )
        _write(path, asdict(lease), validate_lease)
        return lease


def claim_attempt(
    paths: store.StatePaths,
    attempt_id: str,
    owner: str,
    expected_fencing_token: int,
    now: datetime,
) -> Lease:
    """Claim an active attempt after its expected predecessor becomes stale."""
    attempt_id = _identity(attempt_id, "attempt_id")
    owner = _identity(owner, "owner")
    if (
        type(expected_fencing_token) is not int
        or expected_fencing_token < 0
    ):
        raise ValueError("fencing_token")
    observed_at = _datetime(now)
    with hold_owner_claim(paths, owner):
        resource = f"attempt:{attempt_id}"
        path = paths.resource_lease(resource)
        with store._exclusive_lock(path):
            _preflight_claim_attempt_locked(
                paths,
                path,
                resource,
                attempt_id,
                owner,
                expected_fencing_token,
                observed_at,
            )
        import paper_owner

        paper_owner.recover_release_transactions(paths)
        return _claim_attempt_locked(
            paths,
            attempt_id,
            owner,
            expected_fencing_token,
            observed_at,
        )


def _claim_attempt_locked(
    paths: store.StatePaths,
    attempt_id: str,
    owner: str,
    expected_fencing_token: int,
    observed_at: datetime,
) -> Lease:
    """Claim an attempt while the caller holds this owner's claim lock."""
    resource = f"attempt:{attempt_id}"
    path = paths.resource_lease(resource)
    with store._exclusive_lock(path):
        prior = _preflight_claim_attempt_locked(
            paths,
            path,
            resource,
            attempt_id,
            owner,
            expected_fencing_token,
            observed_at,
        )
        lease = Lease(
            resource=resource,
            owner=owner,
            attempt_id=attempt_id,
            acquired_at=_timestamp(observed_at),
            expires_at=_timestamp(observed_at + ATTEMPT_WORK_LEASE_TTL),
            fencing_token=1 if prior is None else prior.fencing_token + 1,
        )
        _write(path, asdict(lease), validate_lease)
        return lease


def _preflight_claim_attempt_locked(
    paths: store.StatePaths,
    path: Path,
    resource: str,
    attempt_id: str,
    owner: str,
    expected_fencing_token: int,
    observed_at: datetime,
) -> Lease | None:
    """Validate one claim without mutation while its attempt lock is held."""
    index = store.read_json(paths.index)
    store.validate_index(index)
    if attempt_id not in index["attempts"]:
        raise ValueError("attempt_id")
    prior = _read_lease(path)
    if prior is None:
        if expected_fencing_token != 0:
            raise StaleFence(resource)
    else:
        if prior.resource != resource or prior.attempt_id != attempt_id:
            raise StaleFence(resource)
        if prior.fencing_token != expected_fencing_token:
            raise StaleFence(resource)
        if observed_at < _parse(prior.acquired_at):
            raise ValueError("now")
        if prior.released_at is not None:
            if observed_at < _parse(prior.released_at):
                raise ValueError("now")
        elif observed_at < _parse(prior.expires_at):
            raise LeaseBusy(resource)
    _require_owner_available(paths, owner, attempt_id, observed_at)
    return prior


@contextmanager
def hold_owner_claim(paths: store.StatePaths, owner: str) -> Iterator[None]:
    """Serialize all public attempt acquisitions for one persistent owner."""
    owner = _identity(owner, "owner")
    path = paths.resource_lease(f"paper-owner-claim:{owner}")
    with store._exclusive_lock(path):
        yield


def renew_lease(
    paths: store.StatePaths,
    lease: Lease,
    now: datetime,
    ttl: timedelta,
) -> Lease:
    """Extend the current live lease without changing its fence."""
    observed_at = _datetime(now)
    expires_at = _timestamp(observed_at + _ttl(ttl))
    path = paths.resource_lease(lease.resource)
    with store._exclusive_lock(path):
        current = _require_current(path, lease)
        renewed = _renewed_lease(current, observed_at, expires_at)
        _write(path, asdict(renewed), validate_lease)
        return renewed


def renew_attempt(
    paths: store.StatePaths,
    lease: Lease,
    now: datetime,
) -> Lease:
    """Renew one exact live attempt writer for the fixed work interval."""
    expected_resource = f"attempt:{lease.attempt_id}"
    if lease.resource != expected_resource:
        raise StaleFence(expected_resource)
    observed_at = _datetime(now)
    expires_at = _timestamp(observed_at + ATTEMPT_WORK_LEASE_TTL)
    path = paths.resource_lease(expected_resource)
    with store._exclusive_lock(path):
        current = _require_current(path, lease)
        with store._exclusive_lock(paths.index):
            index = store.read_json(paths.index)
            store.validate_index(index)
            reference = index["attempts"].get(lease.attempt_id)
            if reference is None:
                raise ValueError("attempt_id")
            attempt = store.read_json(paths.attempt(lease.attempt_id))
            store.validate_attempt(attempt)
            if attempt["attempt_id"] != lease.attempt_id:
                raise ValueError("attempt_id")
            if attempt["paper_id"] != reference["paper_id"]:
                raise ValueError("paper_id")
            renewed = _renewed_lease(current, observed_at, expires_at)
            _write(path, asdict(renewed), validate_lease)
            return renewed


def release_lease(
    paths: store.StatePaths,
    lease: Lease,
    now: datetime,
) -> Lease:
    """Release the current fence while retaining its monotonic token."""
    observed_at = _datetime(now)
    path = paths.resource_lease(lease.resource)
    with store._exclusive_lock(path):
        return _release_held_fence(paths, lease, observed_at)


def _release_held_fence(
    paths: store.StatePaths,
    lease: Lease,
    observed_at: datetime,
) -> Lease:
    """Release a current lease while its resource lock is already held."""
    released_at = _timestamp(observed_at)
    path = paths.resource_lease(lease.resource)
    current = _require_current(path, lease)
    if _parse(current.expires_at) <= observed_at:
        raise StaleFence(lease.resource)
    if _parse(released_at) < _parse(current.acquired_at):
        raise ValueError("now")
    released = replace(current, released_at=released_at)
    _write(path, asdict(released), validate_lease)
    return released


def assert_fence(
    paths: store.StatePaths,
    lease: Lease,
    now: datetime | None = None,
) -> Lease:
    """Return the persisted lease or reject a stale, released, or expired fence."""
    with hold_fence(paths, lease, now) as current:
        return current


@contextmanager
def hold_fence(
    paths: store.StatePaths,
    lease: Lease,
    now: datetime | None = None,
) -> Iterator[Lease]:
    """Hold the resource lock while a fenced mutation remains in progress."""
    observed_at = _datetime(now or datetime.now(timezone.utc))
    path = paths.resource_lease(lease.resource)
    with store._exclusive_lock(path):
        current = _require_current(path, lease)
        if _parse(current.expires_at) <= observed_at:
            raise StaleFence(lease.resource)
        yield current


def expire_stale_leases(
    paths: store.StatePaths, now: datetime
) -> list[str]:
    """Mark every expired live resource lease released."""
    observed_at = _datetime(now)
    expired: list[str] = []
    lease_directory = paths.root / "leases"
    for path in sorted(lease_directory.glob("*.json")):
        with store._exclusive_lock(path):
            current = _read_lease(path)
            if (
                current is not None
                and current.released_at is None
                and _parse(current.expires_at) <= observed_at
            ):
                released = replace(current, released_at=_timestamp(observed_at))
                _write(path, asdict(released), validate_lease)
                expired.append(current.resource)
    return expired


def reserve_metered_cost(
    paths: store.StatePaths,
    attempt_id: str,
    provider: str,
    amount_usd: float,
    now: datetime,
) -> MeteredCostReservation:
    """Reserve finite paid cost under the global index lock."""
    attempt_id = _identity(attempt_id, "attempt_id")
    provider = _identity(provider, "provider")
    amount = _amount(amount_usd, "amount_usd")
    if provider.lower() in SUBSCRIPTION_PROVIDERS:
        amount = 0.0
    reserved_at = _timestamp(now)
    path = paths.cost_reservation(attempt_id, provider)
    with store._exclusive_lock(paths.index):
        index = store.read_json(paths.index)
        store.validate_index(index)
        paper_id = _paper_id(index, attempt_id)
        reservations = _reservations(paths)
        prior = _reservation_at(path)
        prior_active = (
            0.0
            if prior is None or prior.reconciled_at is not None
            else prior.amount_usd
        )
        global_reserved = sum(
            reservation.amount_usd
            for _candidate, reservation in reservations
            if reservation.reconciled_at is None
        )
        paper_committed = sum(
            reservation.cumulative_actual_usd
            + (
                reservation.amount_usd
                if reservation.reconciled_at is None
                else 0.0
            )
            for _candidate, reservation in reservations
            if reservation.paper_id == paper_id
        )
        prior_committed = 0.0
        if prior is not None:
            prior_committed = prior.cumulative_actual_usd + prior_active
        if paper_committed - prior_committed + amount > MAX_PAPER_COST_USD:
            raise CostLimitExceeded("paper cost exceeds USD 10")
        global_limit = index["resource_limits"]["metered_api_reserved_usd"]
        if global_reserved - prior_active + amount > global_limit:
            raise CostLimitExceeded("global reserved cost limit exceeded")
        reservation = MeteredCostReservation(
            attempt_id=attempt_id,
            paper_id=paper_id,
            provider=provider,
            amount_usd=amount,
            reserved_at=reserved_at,
            fencing_token=1 if prior is None else prior.fencing_token + 1,
            cumulative_actual_usd=(
                0.0 if prior is None else prior.cumulative_actual_usd
            ),
        )
        with store._exclusive_lock(path):
            _write(path, asdict(reservation), validate_reservation)
        return reservation


def reconcile_metered_cost(
    paths: store.StatePaths,
    reservation: MeteredCostReservation,
    actual_amount_usd: float,
    now: datetime,
) -> MeteredCostReservation:
    """Replace an active reservation with its finite actual cost."""
    actual = _amount(actual_amount_usd, "actual_amount_usd")
    reconciled_at = _timestamp(now)
    path = paths.cost_reservation(reservation.attempt_id, reservation.provider)
    with store._exclusive_lock(paths.index):
        with store._exclusive_lock(path):
            current = _reservation_at(path)
            if (
                current is None
                or current.attempt_id != reservation.attempt_id
                or current.paper_id != reservation.paper_id
                or current.provider != reservation.provider
                or current.fencing_token != reservation.fencing_token
                or current.reconciled_at is not None
            ):
                raise StaleFence(reservation.attempt_id)
            if actual > current.amount_usd:
                raise ValueError("actual_amount_usd")
            reconciled = replace(
                current,
                actual_amount_usd=actual,
                reconciled_at=reconciled_at,
                cumulative_actual_usd=current.cumulative_actual_usd + actual,
            )
            _write(path, asdict(reconciled), validate_reservation)
            return reconciled


def validate_lease(value: dict) -> None:
    if type(value) is not dict or set(value) != LEASE_KEYS:
        raise ValueError("lease")
    _validate_resource(value["resource"])
    _identity(value["owner"], "owner")
    _identity(value["attempt_id"], "attempt_id")
    acquired = _parse(value["acquired_at"])
    expires = _parse(value["expires_at"])
    if expires <= acquired:
        raise ValueError("expires_at")
    if type(value["fencing_token"]) is not int or value["fencing_token"] < 1:
        raise ValueError("fencing_token")
    if value["released_at"] is not None:
        released = _parse(value["released_at"])
        if released < acquired:
            raise ValueError("released_at")


def validate_reservation(value: dict) -> None:
    if type(value) is not dict or set(value) != RESERVATION_KEYS:
        raise ValueError("reservation")
    _identity(value["attempt_id"], "attempt_id")
    _identity(value["paper_id"], "paper_id")
    _identity(value["provider"], "provider")
    amount = _amount(value["amount_usd"], "amount_usd")
    reserved = _parse(value["reserved_at"])
    if type(value["fencing_token"]) is not int or value["fencing_token"] < 1:
        raise ValueError("fencing_token")
    actual = value["actual_amount_usd"]
    reconciled = value["reconciled_at"]
    if (actual is None) != (reconciled is None):
        raise ValueError("reconciliation")
    if actual is not None and _amount(actual, "actual_amount_usd") > amount:
        raise ValueError("actual_amount_usd")
    if reconciled is not None:
        if _parse(reconciled) < reserved:
            raise ValueError("reconciled_at")
    _amount(value["cumulative_actual_usd"], "cumulative_actual_usd")


def _require_owner_available(
    paths: store.StatePaths,
    owner: str,
    excluded_attempt_id: str,
    now: datetime,
) -> None:
    for path in (paths.root / "leases").glob("*.json"):
        value = store.read_json(path)
        validate_lease(value)
        if (
            value["resource"].startswith("attempt:")
            and value["attempt_id"] != excluded_attempt_id
            and value["owner"] == owner
            and value["released_at"] is None
            and _parse(value["expires_at"]) > now
        ):
            raise LeaseBusy(f"paper-owner:{owner}")


def _require_current(path: Path, lease: Lease) -> Lease:
    current = _read_lease(path)
    if (
        current is None
        or current.resource != lease.resource
        or current.owner != lease.owner
        or current.attempt_id != lease.attempt_id
        or current.fencing_token != lease.fencing_token
        or current.released_at is not None
    ):
        raise StaleFence(lease.resource)
    return current


def _renewed_lease(
    current: Lease, observed_at: datetime, expires_at: str
) -> Lease:
    if observed_at < _parse(current.acquired_at):
        raise ValueError("now")
    if _parse(current.expires_at) <= observed_at:
        raise StaleFence(current.resource)
    if _parse(expires_at) < _parse(current.expires_at):
        raise ValueError("now")
    return replace(current, expires_at=expires_at)


def _read_lease(path: Path) -> Lease | None:
    if not path.exists():
        return None
    value = store.read_json(path)
    validate_lease(value)
    return Lease(**value)


def _reservation_at(path: Path) -> MeteredCostReservation | None:
    if not path.exists():
        return None
    value = store.read_json(path)
    validate_reservation(value)
    return MeteredCostReservation(**value)


def _reservations(
    paths: store.StatePaths,
) -> list[tuple[Path, MeteredCostReservation]]:
    return [
        (path, reservation)
        for path in sorted((paths.root / "cost-reservations").glob("*.json"))
        if (reservation := _reservation_at(path)) is not None
    ]


def _write(path: Path, value: dict, validator) -> None:
    validator(value)
    store._atomic_json_write(path, value)


def _validate_resource(value: object) -> str:
    if (
        type(value) is not str
        or not value
        or value != value.strip()
        or any(character.isspace() or ord(character) < 32 for character in value)
    ):
        raise ValueError("resource")
    kind, separator, identity = value.partition(":")
    if not separator or not identity:
        raise ValueError("resource")
    store.validate_id(kind)
    return value


def _identity(value: object, field: str) -> str:
    if type(value) is not str or not value or value != value.strip():
        raise ValueError(field)
    return value


def _paper_id(index: dict, attempt_id: str) -> str:
    for section in ("attempts", "history"):
        reference = index[section].get(attempt_id)
        if reference is not None:
            return _identity(reference["paper_id"], "paper_id")
    return attempt_id


def _amount(value: object, field: str) -> float:
    if (
        type(value) not in {int, float}
        or not math.isfinite(value)
        or value < 0
    ):
        raise ValueError(field)
    return float(value)


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


def _parse(value: object) -> datetime:
    if type(value) is not str:
        raise ValueError("timestamp")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as error:
        raise ValueError("timestamp") from error
    return _datetime(parsed)


def _ttl(value: object) -> timedelta:
    if not isinstance(value, timedelta) or value <= timedelta(0):
        raise ValueError("ttl")
    return value
