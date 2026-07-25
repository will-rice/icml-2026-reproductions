"""Immutable live snapshots of challenge datasets and reproduction Spaces."""

from __future__ import annotations

import copy
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import re
import sys


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import store  # noqa: E402


CHALLENGE_REPO = "ICML-2026-agent-repro/challenge"
VERDICTS_REPO = "ICML-2026-agent-repro/verdicts"
ASSESSMENT_KEYS = {
    "paper_id",
    "score",
    "target_claims",
    "claim_bindings",
    "upstream_revision",
    "artifact_access",
    "cpu_only",
    "safety_blocker",
    "licensing_blocker",
    "estimated_api_cost_usd",
}
SNAPSHOT_KEYS = {
    "fetched_at",
    "source_revision",
    "sources",
    "assessments",
    "candidates",
    "queued_submissions",
    "tagged_spaces",
    "verdicts",
    "spaces",
}


def load_assessments(path: Path) -> dict:
    """Load an explicit agent assessment document with its canonical hash."""
    content = path.read_bytes()
    document = json.loads(content)
    _validate_assessment_document(document)
    return {
        "content_sha256": hashlib.sha256(_canonical_json(document)).hexdigest(),
        "document": document,
    }


def claim_text_sha256(text: str) -> str:
    """Return the stable digest used to bind an extracted challenge claim."""
    if type(text) is not str or not text:
        raise ValueError("challenge_claim")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def fetch_live_snapshot(
    client, observed_at: str, assessment_input: dict | None = None
) -> dict:
    """Fetch a deterministic snapshot through an injected Hub client."""
    fetched_at = _timestamp(observed_at)
    if assessment_input is not None:
        _validate_assessment_input(assessment_input)
    challenge_info = client.dataset_info(CHALLENGE_REPO, revision="main")
    verdicts_info = client.dataset_info(VERDICTS_REPO, revision="main")
    challenge_revision = _attribute(challenge_info, "sha")
    verdicts_revision = _attribute(verdicts_info, "sha")
    if (
        assessment_input is not None
        and assessment_input["document"]["challenge_revision"]
        != challenge_revision
    ):
        raise ValueError("challenge_revision")
    index, index_hash = _download_json(
        client, CHALLENGE_REPO, challenge_revision, "index.json"
    )
    challenge, challenge_hash = _download_json(
        client, CHALLENGE_REPO, challenge_revision, "challenge.json"
    )
    verdict_data, verdicts_hash = _download_json(
        client, VERDICTS_REPO, verdicts_revision, "verdicts.json"
    )

    candidates, matched_paper_ids, current_aliases = _candidate_records(
        index,
        challenge,
        None if assessment_input is None else assessment_input["document"],
    )
    paper_aliases = _paper_aliases(index, current_aliases)
    verdicts = _verdict_records(verdict_data, verdicts_revision, paper_aliases)
    verdict_paper_ids = {}
    for record in verdicts:
        verdict_paper_ids.setdefault(record["space_id"], set()).add(
            record["paper_id"]
        )
    listed_spaces = list(client.list_spaces(filter="icml2026-repro", full=True))
    spaces = sorted(
        (
            _space_record(
                space,
                paper_aliases,
                verdict_paper_ids.get(
                    _optional_attribute(space, "id", None), set()
                ),
            )
            for space in listed_spaces
        ),
        key=lambda record: record["space_id"],
    )
    verdict_space_ids = {record["space_id"] for record in verdicts}
    tagged_spaces = [
        {
            "paper_id": paper_id,
            "space_id": space["space_id"],
            "revision": space["revision"],
        }
        for space in spaces
        for paper_id in space["paper_ids"]
    ]
    queued_submissions = [
        {
            "paper_id": paper_id,
            "space_id": space["space_id"],
            "status": "pending",
        }
        for space in spaces
        if space["space_id"] not in verdict_space_ids
        for paper_id in space["paper_ids"]
    ]
    sources = {
        "challenge": {
            "repo_id": CHALLENGE_REPO,
            "revision": challenge_revision,
            "files": {
                "index.json": {
                    "content_sha256": index_hash,
                    "url": _source_url(
                        CHALLENGE_REPO, challenge_revision, "index.json"
                    ),
                },
                "challenge.json": {
                    "content_sha256": challenge_hash,
                    "url": _source_url(
                        CHALLENGE_REPO, challenge_revision, "challenge.json"
                    ),
                },
            },
        },
        "verdicts": {
            "repo_id": VERDICTS_REPO,
            "revision": verdicts_revision,
            "files": {
                "verdicts.json": {
                    "content_sha256": verdicts_hash,
                    "url": _source_url(
                        VERDICTS_REPO, verdicts_revision, "verdicts.json"
                    ),
                }
            },
        },
    }
    assessments = None
    if assessment_input is not None:
        assessment_document = assessment_input["document"]
        assessments = {
            "content_sha256": assessment_input["content_sha256"],
            "challenge_revision": assessment_document["challenge_revision"],
            "assessor": assessment_document["assessor"],
            "assessed_at": assessment_document["assessed_at"],
            "records": copy.deepcopy(assessment_document["assessments"]),
            "matched_paper_ids": matched_paper_ids,
        }
    snapshot = {
        "fetched_at": fetched_at,
        "source_revision": hashlib.sha256(_canonical_json(sources)).hexdigest(),
        "sources": sources,
        "assessments": assessments,
        "candidates": candidates,
        "queued_submissions": queued_submissions,
        "tagged_spaces": tagged_spaces,
        "verdicts": verdicts,
        "spaces": spaces,
    }
    _validate_snapshot_payload(snapshot)
    return snapshot


def canonical_snapshot_id(snapshot: dict) -> str:
    """Return the canonical JSON SHA-256 for a snapshot payload."""
    return hashlib.sha256(_canonical_json(snapshot)).hexdigest()


def persist_snapshot(paths: store.StatePaths, snapshot: dict) -> str:
    """Write one content-addressed snapshot and publish its immutable ID."""
    _validate_snapshot_payload(snapshot)
    snapshot_id = canonical_snapshot_id(snapshot)
    persisted = {"snapshot_id": snapshot_id, **copy.deepcopy(snapshot)}
    store.validate_snapshot(persisted)
    path = paths.root / "snapshots" / f"{snapshot_id}.json"
    with store._exclusive_lock(path):
        if path.exists():
            if store.read_json(path) != persisted:
                raise ValueError("snapshot_id")
        else:
            store._atomic_json_write(path, persisted)
    reference = str(path.relative_to(paths.index.parent))
    with store.locked_json(paths.index, store.validate_index) as index:
        existing = index["snapshots"].get(snapshot_id)
        if existing is not None and existing != reference:
            raise ValueError("snapshot_id")
        index["snapshots"][snapshot_id] = reference
    return snapshot_id


def read_snapshot(paths: store.StatePaths, snapshot_id: str) -> dict:
    """Read one immutable snapshot and verify its content-addressed identity."""
    store.validate_id(snapshot_id)
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
    store.validate_snapshot(snapshot)
    if snapshot.get("snapshot_id") != snapshot_id:
        raise ValueError("snapshot_id")
    payload = {key: value for key, value in snapshot.items() if key != "snapshot_id"}
    _validate_snapshot_payload(payload)
    if canonical_snapshot_id(payload) != snapshot_id:
        raise ValueError("snapshot_id")
    return snapshot


def _candidate_records(
    index: object,
    challenge: object,
    assessment_document: dict | None,
) -> tuple[list[dict], list[str], dict[str, str]]:
    index_records = {
        _paper_id(record): record for record in _records(index, "papers")
    }
    challenge_records = _records(challenge, "papers")
    claims = challenge.get("claims") if type(challenge) is dict else None
    if type(claims) is not dict:
        raise ValueError("claims")
    current_records = {}
    for current in challenge_records:
        paper_id = _paper_id(current)
        if paper_id in current_records:
            raise ValueError("paper_id")
        metadata = copy.deepcopy(index_records.get(paper_id, {}))
        metadata.update(copy.deepcopy(current))
        current_records[paper_id] = metadata
    aliases = _canonical_candidate_aliases(current_records)
    assessments = {}
    if assessment_document is not None:
        for record in assessment_document["assessments"]:
            paper_id = aliases.get(record["paper_id"], record["paper_id"])
            if paper_id in assessments:
                raise ValueError("paper_id")
            assessments[paper_id] = record
    groups = {}
    for paper_id, canonical_id in aliases.items():
        groups.setdefault(canonical_id, []).append(paper_id)
    candidates = []
    matched = []
    for paper_id in sorted(groups):
        alias_ids = sorted(groups[paper_id])
        metadata = copy.deepcopy(current_records[paper_id])
        metadata["paper_id"] = paper_id
        metadata["slug"] = _slug(metadata.get("title"), paper_id)
        if len(alias_ids) > 1:
            metadata["alias_paper_ids"] = [
                alias_id for alias_id in alias_ids if alias_id != paper_id
            ]
        live_claims = _merged_live_claims(claims, paper_id, alias_ids)
        metadata["live_claims"] = live_claims
        assessment = assessments.get(paper_id)
        if assessment is not None and _assessment_matches(assessment, live_claims):
            metadata.update(
                copy.deepcopy(
                    {
                        key: value
                        for key, value in assessment.items()
                        if key != "paper_id"
                    }
                )
            )
            matched.append(paper_id)
        candidates.append(metadata)
    return candidates, sorted(matched), aliases


def _canonical_candidate_aliases(records: dict[str, dict]) -> dict[str, str]:
    parent = {paper_id: paper_id for paper_id in records}
    author_ids = {
        paper_id: {_author_identity(record)} - {None}
        for paper_id, record in records.items()
    }
    buckets = {}
    for paper_id, record in records.items():
        for source_id in _source_identities(record):
            buckets.setdefault(("source", source_id), []).append(paper_id)
        title = _normalized_text(record.get("title"))
        authors = _author_identity(record)
        if title is not None and authors is not None:
            buckets.setdefault(("title-authors", title, authors), []).append(paper_id)

    def root(paper_id):
        while parent[paper_id] != paper_id:
            parent[paper_id] = parent[parent[paper_id]]
            paper_id = parent[paper_id]
        return paper_id

    for bucket in sorted(tuple(sorted(values)) for values in buckets.values()):
        for left_index, left in enumerate(bucket):
            for right in bucket[left_index + 1 :]:
                left_root = root(left)
                right_root = root(right)
                if left_root == right_root:
                    continue
                combined_authors = author_ids[left_root] | author_ids[right_root]
                if len(combined_authors) > 1:
                    continue
                keep, merge = sorted((left_root, right_root))
                parent[merge] = keep
                author_ids[keep] = combined_authors

    groups = {}
    for paper_id in sorted(records):
        groups.setdefault(root(paper_id), []).append(paper_id)
    aliases = {}
    for group in groups.values():
        canonical_id = min(
            group,
            key=lambda paper_id: (_openreview_rank(records[paper_id]), paper_id),
        )
        for paper_id in group:
            aliases[paper_id] = canonical_id
    return aliases


def _merged_live_claims(
    claims: dict, canonical_id: str, alias_ids: list[str]
) -> list[dict]:
    merged = []
    seen = set()
    ordered_ids = [
        canonical_id,
        *(value for value in alias_ids if value != canonical_id),
    ]
    for paper_id in ordered_ids:
        records = copy.deepcopy(claims.get(paper_id, []))
        _validate_live_claims(records)
        for record in records:
            identity = _canonical_json(record)
            if identity not in seen:
                seen.add(identity)
                merged.append(record)
    return merged


def _verdict_records(
    value: object,
    source_revision: str,
    paper_aliases: dict[str, set[str]],
) -> list[dict]:
    if type(value) is not dict or any(
        type(space_id) is not str or not space_id or type(record) is not dict
        for space_id, record in value.items()
    ):
        raise ValueError("verdicts")
    verdicts = []
    for space_id, value_record in value.items():
        source_record = copy.deepcopy(value_record)
        reported_space_id = source_record.get("space_id")
        if reported_space_id is not None and reported_space_id != space_id:
            source_record["reported_space_id"] = reported_space_id
        paper_id = (
            source_record.get("paper_id")
            or source_record.get("orid")
            or source_record.get("paper")
        )
        paper_id = _identity(paper_id, "paper_id")
        for canonical_id in sorted(paper_aliases.get(paper_id, {paper_id})):
            record = copy.deepcopy(source_record)
            record["space_id"] = space_id
            if canonical_id != paper_id:
                record["reported_paper_id"] = paper_id
            record["paper_id"] = canonical_id
            record.setdefault("source_revision", source_revision)
            verdicts.append(record)
    return sorted(
        verdicts, key=lambda record: (record["space_id"], record["paper_id"])
    )


def _download_json(client, repo_id: str, revision: str, filename: str):
    path = client.hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        repo_type="dataset",
        revision=revision,
    )
    content = Path(path).read_bytes()
    return json.loads(content), hashlib.sha256(content).hexdigest()


def _source_url(repo_id: str, revision: str, filename: str) -> str:
    return f"https://huggingface.co/datasets/{repo_id}/resolve/{revision}/{filename}"


def _source_identities(record: dict) -> set[str]:
    identities = set()
    for field in ("arxiv", "alphaxiv"):
        value = record.get(field)
        if type(value) is not str or not value.strip():
            continue
        identity = value.strip().casefold()
        for prefix in (
            "https://arxiv.org/abs/",
            "https://arxiv.org/pdf/",
            "https://alphaxiv.org/abs/",
            "arxiv:",
        ):
            identity = identity.removeprefix(prefix)
        identity = identity.removesuffix(".pdf")
        if identity:
            identities.add(identity)
    return identities


def _normalized_text(value: object) -> str | None:
    if type(value) is not str:
        return None
    normalized = " ".join(value.split()).casefold()
    return normalized or None


def _author_identity(record: dict) -> tuple[str, ...] | None:
    authors = record.get("authors")
    if type(authors) not in {list, tuple} or not authors:
        return None
    normalized = tuple(_normalized_text(author) for author in authors)
    if any(author is None for author in normalized):
        return None
    return normalized


def _openreview_rank(record: dict) -> int:
    value = record.get("or")
    if type(value) is str and re.fullmatch(
        r"https://openreview\.net/forum\?id=[A-Za-z0-9_-]+", value
    ):
        return 0
    for value in (record.get("or"), record.get("orid")):
        if (
            type(value) is str
            and re.fullmatch(r"[A-Za-z0-9_-]+", value)
            and re.search(r"[A-Za-z]", value)
        ):
            return 1
    return 2


def _paper_aliases(
    index: object, current_aliases: dict[str, str]
) -> dict[str, set[str]]:
    aliases = {}
    for record in _records(index, "papers"):
        paper_id = _paper_id(record)
        canonical_id = current_aliases.get(paper_id, paper_id)
        values = [paper_id]
        pid = record.get("pid")
        if pid is not None:
            if type(pid) is int:
                values.append(str(pid))
            else:
                values.append(_identity(pid, "pid"))
        for value in values:
            aliases.setdefault(value, set()).add(canonical_id)
    for paper_id, canonical_id in current_aliases.items():
        aliases.setdefault(paper_id, set()).add(canonical_id)
    return aliases


def _space_record(
    value: object,
    paper_aliases: dict[str, set[str]],
    verdict_paper_ids: set[str],
) -> dict:
    space_id = _attribute(value, "id")
    revision = _attribute(value, "sha")
    tags_value = _optional_attribute(value, "tags", [])
    if type(tags_value) not in {list, tuple} or any(
        type(tag) is not str for tag in tags_value
    ):
        raise ValueError("spaces")
    tags = sorted(tags_value)
    paper_tags = {
        tag.removeprefix("paper-")
        for tag in tags
        if tag.startswith("paper-") and tag.removeprefix("paper-")
    }
    paper_ids = set()
    for paper_tag in paper_tags:
        paper_ids.update(paper_aliases.get(paper_tag, {paper_tag}))
    for verdict_paper_id in verdict_paper_ids:
        paper_ids.update(paper_aliases.get(verdict_paper_id, {verdict_paper_id}))
    return {
        "paper_ids": sorted(paper_ids),
        "space_id": space_id,
        "revision": revision,
        "tags": tags,
    }


def _records(value: object, field: str) -> list:
    if type(value) is dict and type(value.get(field)) is list:
        records = value[field]
        if all(type(record) is dict for record in records):
            return records
    raise ValueError(field)


def _paper_id(record: object) -> str:
    if type(record) is not dict:
        raise ValueError("papers")
    return _identity(record.get("orid"), "orid")


def _assessment_matches(assessment: dict, live_claims: list[dict]) -> bool:
    if not _valid_assessment_record(assessment):
        return False
    return _valid_claim_bindings(
        assessment["claim_bindings"], assessment["target_claims"], live_claims
    )


def _valid_claim_bindings(
    bindings: object, target_claims: list[str], live_claims: object
) -> bool:
    """Check one ordered, digest-pinned live claim binding per target claim."""
    if type(bindings) is not list or type(live_claims) is not list:
        return False
    if len(bindings) != len(target_claims):
        return False
    live_texts = {claim.get("text") for claim in live_claims if type(claim) is dict}
    binding_targets = []
    for binding in bindings:
        if type(binding) is not dict or set(binding) != {
            "target_claim",
            "challenge_claim",
            "challenge_claim_sha256",
        }:
            return False
        target_claim = binding["target_claim"]
        challenge_claim = binding["challenge_claim"]
        digest = binding["challenge_claim_sha256"]
        if (
            type(target_claim) is not str
            or not target_claim
            or type(challenge_claim) is not str
            or not challenge_claim
            or type(digest) is not str
            or digest != claim_text_sha256(challenge_claim)
            or challenge_claim not in live_texts
        ):
            return False
        binding_targets.append(target_claim)
    return binding_targets == target_claims


def _valid_assessment_record(record: object) -> bool:
    if type(record) is not dict or set(record) != ASSESSMENT_KEYS:
        return False
    score = record["score"]
    cost = record["estimated_api_cost_usd"]
    blockers = (record["safety_blocker"], record["licensing_blocker"])
    return (
        type(score) in {int, float}
        and math.isfinite(score)
        and type(cost) in {int, float}
        and math.isfinite(cost)
        and 0 <= cost <= 10
        and type(record["target_claims"]) is list
        and len(record["target_claims"]) >= 2
        and len(set(record["target_claims"])) == len(record["target_claims"])
        and all(type(claim) is str and claim for claim in record["target_claims"])
        and type(record["claim_bindings"]) is list
        and type(record["upstream_revision"]) is str
        and bool(record["upstream_revision"])
        and type(record["artifact_access"]) is bool
        and type(record["cpu_only"]) is bool
        and all(
            value is None or type(value) is str and bool(value)
            for value in blockers
        )
    )


def _validate_assessment_document(value: object) -> None:
    if type(value) is not dict or set(value) != {
        "challenge_revision",
        "assessor",
        "assessed_at",
        "assessments",
    }:
        raise ValueError("assessments")
    _identity(value["challenge_revision"], "challenge_revision")
    _identity(value["assessor"], "assessor")
    _timestamp(value["assessed_at"])
    records = value["assessments"]
    if type(records) is not list or any(type(record) is not dict for record in records):
        raise ValueError("assessments")
    paper_ids = [record.get("paper_id") for record in records]
    if any(type(paper_id) is not str or not paper_id for paper_id in paper_ids):
        raise ValueError("paper_id")
    if len(paper_ids) != len(set(paper_ids)):
        raise ValueError("paper_id")
    _canonical_json(value)


def _validate_assessment_input(value: object) -> None:
    if type(value) is not dict or set(value) != {"content_sha256", "document"}:
        raise ValueError("assessments")
    digest = value["content_sha256"]
    if type(digest) is not str or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
        raise ValueError("content_sha256")
    _validate_assessment_document(value["document"])
    expected = hashlib.sha256(_canonical_json(value["document"])).hexdigest()
    if digest != expected:
        raise ValueError("content_sha256")


def _validate_live_claims(value: object) -> None:
    if type(value) is not list or any(
        type(claim) is not dict
        or type(claim.get("text")) is not str
        or not claim["text"]
        or type(claim.get("status")) is not str
        or not claim["status"]
        for claim in value
    ):
        raise ValueError("claims")


def _validate_snapshot_payload(snapshot: object) -> None:
    if type(snapshot) is not dict or set(snapshot) != SNAPSHOT_KEYS:
        raise ValueError("snapshot")
    _timestamp(snapshot["fetched_at"])
    _identity(snapshot["source_revision"], "source_revision")
    if type(snapshot["sources"]) is not dict or (
        snapshot["assessments"] is not None
        and type(snapshot["assessments"]) is not dict
    ):
        raise ValueError("sources")
    for field in (
        "candidates",
        "queued_submissions",
        "tagged_spaces",
        "verdicts",
        "spaces",
    ):
        if type(snapshot[field]) is not list:
            raise ValueError(field)
    for space in snapshot["spaces"]:
        if type(space) is not dict or set(space) != {
            "paper_ids",
            "space_id",
            "revision",
            "tags",
        }:
            raise ValueError("spaces")
        paper_ids = space["paper_ids"]
        if (
            type(paper_ids) is not list
            or any(type(paper_id) is not str or not paper_id for paper_id in paper_ids)
            or paper_ids != sorted(set(paper_ids))
        ):
            raise ValueError("spaces")
    _canonical_json(snapshot)


def _canonical_json(value: dict) -> bytes:
    return json.dumps(
        value, allow_nan=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")


def _slug(title: object, paper_id: str) -> str:
    source = title if type(title) is str and title else paper_id
    slug = re.sub(r"[^a-z0-9]+", "-", source.lower()).strip("-")
    return slug or paper_id.lower()


def _attribute(value: object, field: str) -> str:
    return _identity(_optional_attribute(value, field, None), field)


def _optional_attribute(value: object, field: str, default):
    if isinstance(value, dict):
        return value.get(field, default)
    return getattr(value, field, default)


def _identity(value: object, field: str) -> str:
    if type(value) is not str or not value or value != value.strip():
        raise ValueError(field)
    return value


def _timestamp(value: object) -> str:
    if type(value) is not str:
        raise ValueError("observed_at")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as error:
        raise ValueError("observed_at") from error
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("observed_at")
    return parsed.isoformat()
