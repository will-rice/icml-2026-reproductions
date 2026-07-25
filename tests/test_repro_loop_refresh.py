"""Tests for immutable, recorded live-refresh snapshots."""

from datetime import datetime, timezone
import hashlib
import importlib
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest


SCRIPTS = (
    Path(__file__).resolve().parents[1] / "skills" / "icml-repro-loop" / "scripts"
)
OBSERVED_AT = "2026-07-24T18:00:00+00:00"


def load_module(name: str):
    sys.path.insert(0, str(SCRIPTS))
    sys.modules.pop(name, None)
    return importlib.import_module(name)


class RecordedHubClient:
    """A deterministic client with the verified official response shapes."""

    def __init__(self, root: Path):
        self.calls = []
        self.files = {}
        index = {
            "papers": [
                {
                    "pid": 101,
                    "orid": "paper-a",
                    "title": "Paper A",
                    "arxiv": "2601.00001",
                },
                {
                    "pid": 102,
                    "orid": "paper-b",
                    "title": "Paper B",
                    "arxiv": "2601.00002",
                },
                {
                    "pid": 103,
                    "orid": "paper-c",
                    "title": "Paper C",
                    "arxiv": "2601.00003",
                },
            ],
            "areas": ["Deep Learning"],
            "areaTree": {"Deep Learning": []},
        }
        challenge = {
            "papers": [
                {"orid": "paper-a", "title": "Paper A", "arxiv": "2601.00001"},
                {"orid": "paper-b", "title": "Paper B", "arxiv": "2601.00002"},
            ],
            "claims": {
                "paper-a": [
                    {"text": "Claim A1", "status": "extracted"},
                    {"text": "Claim A2", "status": "extracted"},
                ],
                "paper-b": [
                    {"text": "Claim B1", "status": "extracted"},
                    {"text": "Claim B2", "status": "extracted"},
                ],
            },
            "areas": ["Deep Learning"],
        }
        verdicts = {
            "org/repro-paper-c": {
                "orid": "paper-c",
                "sha": "judged-space-sha",
            }
        }
        for repo_id, filename, value in (
            ("ICML-2026-agent-repro/challenge", "index.json", index),
            ("ICML-2026-agent-repro/challenge", "challenge.json", challenge),
            ("ICML-2026-agent-repro/verdicts", "verdicts.json", verdicts),
        ):
            path = root / repo_id.rsplit("/", 1)[-1] / filename
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(value), encoding="utf-8")
            self.files[(repo_id, filename)] = path
        self.spaces = [
            SimpleNamespace(
                id=f"org/repro-{paper_id}",
                sha=f"space-sha-{paper_id}",
                tags=["icml2026-repro", f"paper-{paper_id}"],
            )
            for paper_id in ("paper-b", "paper-c")
        ]

    def dataset_info(self, repo_id, *, revision=None):
        self.calls.append(("dataset_info", repo_id, revision))
        sha = "challenge-sha" if repo_id.endswith("challenge") else "verdict-sha"
        return SimpleNamespace(id=repo_id, sha=sha)

    def hf_hub_download(self, *, repo_id, filename, repo_type, revision):
        self.calls.append(
            ("hf_hub_download", repo_id, filename, repo_type, revision)
        )
        return str(self.files[(repo_id, filename)])

    def list_spaces(self, **kwargs):
        self.calls.append(("list_spaces", kwargs))

        def spaces():
            for space in self.spaces:
                self.calls.append(("space_yield", space.id))
                yield space

        return spaces()


@pytest.fixture
def recorded_hub_client(tmp_path):
    return RecordedHubClient(tmp_path)


@pytest.fixture
def assessments_path(tmp_path):
    path = tmp_path / "assessments.json"
    path.write_text(
        json.dumps(
            {
                "challenge_revision": "challenge-sha",
                "assessor": "selection-agent",
                "assessed_at": OBSERVED_AT,
                "assessments": [assessment()],
            }
        ),
        encoding="utf-8",
    )
    return path


def assessment(**updates) -> dict:
    value = {
        "paper_id": "paper-a",
        "score": 10,
        "target_claims": ["Claim A1", "Claim A2"],
        "upstream_revision": "arxiv:2601.00001v1",
        "artifact_access": True,
        "cpu_only": True,
        "safety_blocker": None,
        "licensing_blocker": None,
        "estimated_api_cost_usd": 0.0,
    }
    value.update(updates)
    return value


def canonical_json(value: dict) -> bytes:
    return json.dumps(
        value, allow_nan=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")


def set_live_documents(client, papers, claims, verdicts=None):
    documents = {
        ("ICML-2026-agent-repro/challenge", "index.json"): {"papers": papers},
        ("ICML-2026-agent-repro/challenge", "challenge.json"): {
            "papers": papers,
            "claims": claims,
        },
        ("ICML-2026-agent-repro/verdicts", "verdicts.json"): verdicts or {},
    }
    for key, document in documents.items():
        client.files[key].write_text(json.dumps(document), encoding="utf-8")


def photoagent_records():
    shared = {
        "alphaxiv": "2602.22809",
        "arxiv": "2602.22809",
        "authors": [
            "Mingde Yao",
            "Zhiyuan You",
            "King-Man Tam",
            "Menglu Wang",
            "Tianfan Xue",
        ],
        "title": (
            "PhotoAgent: Exploratory Visual Aesthetic Planning with Large Vision "
            "Models"
        ),
    }
    return [
        {**shared, "or": "", "orid": "71050", "pid": "71050"},
        {
            **shared,
            "or": "https://openreview.net/forum?id=Ws8swqL5ob",
            "orid": "Ws8swqL5ob",
            "pid": "63474",
        },
    ]


def test_refresh_uses_current_challenge_and_explicit_matching_assessments(
    recorded_hub_client, assessments_path
):
    refresh = load_module("refresh")
    scheduler = load_module("scheduler")
    assessment_input = refresh.load_assessments(assessments_path)

    snapshot = refresh.fetch_live_snapshot(
        recorded_hub_client, OBSERVED_AT, assessment_input
    )

    assert [candidate["paper_id"] for candidate in snapshot["candidates"]] == [
        "paper-a",
        "paper-b",
    ]
    assessed, unassessed = snapshot["candidates"]
    assert assessed["target_claims"] == ["Claim A1", "Claim A2"]
    assert assessed["upstream_revision"] == "arxiv:2601.00001v1"
    assert assessed["live_claims"] == [
        {"text": "Claim A1", "status": "extracted"},
        {"text": "Claim A2", "status": "extracted"},
    ]
    assert "score" not in unassessed
    persisted = {
        "snapshot_id": refresh.canonical_snapshot_id(snapshot),
        **snapshot,
    }
    assert scheduler.rank_eligible_candidates(persisted) == [assessed]
    assessment_document = json.loads(assessments_path.read_text(encoding="utf-8"))
    assert snapshot["assessments"]["content_sha256"] == hashlib.sha256(
        canonical_json(assessment_document)
    ).hexdigest()
    assert snapshot["assessments"]["records"] == [assessment()]


def test_raw_refresh_persists_current_revision_and_claims_for_inspection(
    tmp_path, recorded_hub_client
):
    refresh = load_module("refresh")
    store = load_module("store")
    paths = store.StatePaths(tmp_path / "repro-loop.json")
    store.atomic_json_write(paths.index, store.new_index(), store.validate_index)

    snapshot = refresh.fetch_live_snapshot(recorded_hub_client, OBSERVED_AT)
    snapshot_id = refresh.persist_snapshot(paths, snapshot)
    inspected = refresh.read_snapshot(paths, snapshot_id)

    assert inspected["sources"]["challenge"]["revision"] == "challenge-sha"
    assert inspected["assessments"] is None
    assert inspected["candidates"][0]["live_claims"] == [
        {"text": "Claim A1", "status": "extracted"},
        {"text": "Claim A2", "status": "extracted"},
    ]
    assert all("score" not in candidate for candidate in inspected["candidates"])


def test_assessed_refresh_fails_on_live_revision_drift(
    recorded_hub_client, assessments_path
):
    refresh = load_module("refresh")
    document = json.loads(assessments_path.read_text(encoding="utf-8"))
    document["challenge_revision"] = "stale-challenge-sha"
    assessments_path.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(ValueError, match="challenge_revision"):
        refresh.fetch_live_snapshot(
            recorded_hub_client,
            OBSERVED_AT,
            refresh.load_assessments(assessments_path),
        )


def test_fetch_rejects_assessment_mutated_after_hashing(
    recorded_hub_client, assessments_path
):
    refresh = load_module("refresh")
    assessment_input = refresh.load_assessments(assessments_path)
    assessment_input["document"]["assessments"][0]["score"] = 9

    with pytest.raises(ValueError, match="content_sha256"):
        refresh.fetch_live_snapshot(
            recorded_hub_client,
            OBSERVED_AT,
            assessment_input,
        )


def test_assessment_with_non_live_claim_is_not_merged(
    recorded_hub_client, assessments_path
):
    refresh = load_module("refresh")
    assessments = json.loads(assessments_path.read_text(encoding="utf-8"))
    assessments["assessments"][0]["target_claims"] = ["Claim A1", "not live"]
    assessments_path.write_text(json.dumps(assessments), encoding="utf-8")

    snapshot = refresh.fetch_live_snapshot(
        recorded_hub_client,
        OBSERVED_AT,
        refresh.load_assessments(assessments_path),
    )

    assert "score" not in snapshot["candidates"][0]
    assert snapshot["assessments"]["matched_paper_ids"] == []
    assert snapshot["assessments"]["records"] == assessments["assessments"]


def test_verdict_keys_are_authoritative_and_only_unjudged_spaces_are_queued(
    recorded_hub_client, assessments_path
):
    refresh = load_module("refresh")

    snapshot = refresh.fetch_live_snapshot(
        recorded_hub_client,
        OBSERVED_AT,
        refresh.load_assessments(assessments_path),
    )

    assert [record["space_id"] for record in snapshot["tagged_spaces"]] == [
        "org/repro-paper-b",
        "org/repro-paper-c",
    ]
    assert snapshot["verdicts"][0]["space_id"] == "org/repro-paper-c"
    assert snapshot["queued_submissions"] == [
        {
            "paper_id": "paper-b",
            "space_id": "org/repro-paper-b",
            "status": "pending",
        }
    ]


def test_space_pid_and_orid_tags_collapse_to_one_canonical_paper(
    recorded_hub_client,
):
    refresh = load_module("refresh")
    recorded_hub_client.spaces = [
        SimpleNamespace(
            id="org/repro-paper-a",
            sha="space-sha-a",
            tags=["icml2026-repro", "paper-101", "paper-paper-a"],
        )
    ]

    snapshot = refresh.fetch_live_snapshot(recorded_hub_client, OBSERVED_AT)

    assert snapshot["spaces"][0]["paper_ids"] == ["paper-a"]
    assert snapshot["tagged_spaces"] == [
        {
            "paper_id": "paper-a",
            "space_id": "org/repro-paper-a",
            "revision": "space-sha-a",
        }
    ]
    assert snapshot["queued_submissions"] == [
        {
            "paper_id": "paper-a",
            "space_id": "org/repro-paper-a",
            "status": "pending",
        }
    ]


def test_untagged_space_recovers_paper_identity_from_verdict(recorded_hub_client):
    refresh = load_module("refresh")
    recorded_hub_client.spaces = [
        SimpleNamespace(
            id="org/repro-paper-c",
            sha="judged-space-sha",
            tags=["icml2026-repro"],
        )
    ]

    snapshot = refresh.fetch_live_snapshot(recorded_hub_client, OBSERVED_AT)

    assert snapshot["spaces"][0]["paper_ids"] == ["paper-c"]
    assert snapshot["tagged_spaces"] == [
        {
            "paper_id": "paper-c",
            "space_id": "org/repro-paper-c",
            "revision": "judged-space-sha",
        }
    ]
    assert snapshot["queued_submissions"] == []


def test_untagged_unjudged_space_remains_unassociated_provenance(
    recorded_hub_client,
):
    refresh = load_module("refresh")
    recorded_hub_client.spaces = [
        SimpleNamespace(
            id="org/legacy-reproduction",
            sha="legacy-space-sha",
            tags=["icml2026-repro"],
        )
    ]

    snapshot = refresh.fetch_live_snapshot(recorded_hub_client, OBSERVED_AT)

    assert snapshot["spaces"] == [
        {
            "paper_ids": [],
            "space_id": "org/legacy-reproduction",
            "revision": "legacy-space-sha",
            "tags": ["icml2026-repro"],
        }
    ]
    assert snapshot["tagged_spaces"] == []
    assert snapshot["queued_submissions"] == []


def test_ambiguous_space_excludes_every_canonical_paper(recorded_hub_client):
    refresh = load_module("refresh")
    recorded_hub_client.spaces = [
        SimpleNamespace(
            id="org/ambiguous-reproduction",
            sha="ambiguous-space-sha",
            tags=["icml2026-repro", "paper-101", "paper-paper-b"],
        )
    ]

    snapshot = refresh.fetch_live_snapshot(recorded_hub_client, OBSERVED_AT)

    assert snapshot["spaces"][0]["paper_ids"] == ["paper-a", "paper-b"]
    assert snapshot["tagged_spaces"] == [
        {
            "paper_id": "paper-a",
            "space_id": "org/ambiguous-reproduction",
            "revision": "ambiguous-space-sha",
        },
        {
            "paper_id": "paper-b",
            "space_id": "org/ambiguous-reproduction",
            "revision": "ambiguous-space-sha",
        },
    ]
    assert snapshot["queued_submissions"] == [
        {
            "paper_id": "paper-a",
            "space_id": "org/ambiguous-reproduction",
            "status": "pending",
        },
        {
            "paper_id": "paper-b",
            "space_id": "org/ambiguous-reproduction",
            "status": "pending",
        },
    ]


def test_photoagent_alias_is_canonicalized_and_verdict_excludes_candidate(
    tmp_path,
    recorded_hub_client,
):
    refresh = load_module("refresh")
    scheduler = load_module("scheduler")
    store = load_module("store")
    claims = {
        "71050": [
            {"text": "PhotoAgent uses long-horizon planning.", "status": "extracted"}
        ],
        "Ws8swqL5ob": [
            {"text": "PhotoAgent uses long-horizon planning.", "status": "extracted"},
            {"text": "UGC-Edit contains 7,000 photos.", "status": "extracted"},
        ],
    }
    set_live_documents(
        recorded_hub_client,
        photoagent_records(),
        claims,
        {
            "org/repro-photoagent": {
                "orid": "Ws8swqL5ob",
                "sha": "photoagent-space-sha",
            }
        },
    )
    recorded_hub_client.spaces = [
        SimpleNamespace(
            id="org/repro-photoagent",
            sha="photoagent-space-sha",
            tags=["icml2026-repro", "paper-Ws8swqL5ob"],
        )
    ]
    document = {
        "challenge_revision": "challenge-sha",
        "assessor": "selection-agent",
        "assessed_at": OBSERVED_AT,
        "assessments": [
            assessment(
                paper_id="Ws8swqL5ob",
                target_claims=[
                    "PhotoAgent uses long-horizon planning.",
                    "UGC-Edit contains 7,000 photos.",
                ],
            )
        ],
    }
    assessment_input = {
        "content_sha256": hashlib.sha256(canonical_json(document)).hexdigest(),
        "document": document,
    }

    snapshot = refresh.fetch_live_snapshot(
        recorded_hub_client, OBSERVED_AT, assessment_input
    )
    paths = store.StatePaths(tmp_path / "repro-loop.json")
    store.atomic_json_write(paths.index, store.new_index(), store.validate_index)
    snapshot_id = refresh.persist_snapshot(paths, snapshot)
    report = scheduler.scheduler_pass(
        paths,
        snapshot_id,
        datetime.fromisoformat(OBSERVED_AT).astimezone(timezone.utc),
    )

    assert [candidate["paper_id"] for candidate in snapshot["candidates"]] == [
        "Ws8swqL5ob"
    ]
    assert snapshot["candidates"][0]["alias_paper_ids"] == ["71050"]
    assert snapshot["verdicts"][0]["paper_id"] == "Ws8swqL5ob"
    assert report.assignments == ()


def test_par_numeric_space_tag_normalizes_to_openreview_id(recorded_hub_client):
    refresh = load_module("refresh")
    shared = {
        "alphaxiv": "2602.04883",
        "authors": [
            "Yanru Qu",
            "Cheng-Yen Hsieh",
            "Zaixiang Zheng",
            "Ge Liu",
            "Quanquan Gu",
        ],
        "title": (
            "Protein Autoregressive Modeling via Multiscale Structure Generation"
        ),
    }
    papers = [
        {**shared, "or": "", "orid": "71037", "pid": "71037"},
        {
            **shared,
            "arxiv": "2602.04883",
            "or": "https://openreview.net/forum?id=08tW615mgI",
            "orid": "08tW615mgI",
            "pid": "66808",
        },
    ]
    set_live_documents(
        recorded_hub_client, papers, {"71037": [], "08tW615mgI": []}
    )
    recorded_hub_client.spaces = [
        SimpleNamespace(
            id="org/repro-par",
            sha="par-space-sha",
            tags=["icml2026-repro", "paper-71037"],
        )
    ]

    snapshot = refresh.fetch_live_snapshot(recorded_hub_client, OBSERVED_AT)

    assert snapshot["spaces"][0]["paper_ids"] == ["08tW615mgI"]
    assert snapshot["tagged_spaces"] == [
        {
            "paper_id": "08tW615mgI",
            "space_id": "org/repro-par",
            "revision": "par-space-sha",
        }
    ]
    assert snapshot["queued_submissions"][0]["paper_id"] == "08tW615mgI"


def test_duplicate_candidate_claims_are_merged_and_exactly_deduplicated(
    recorded_hub_client,
):
    refresh = load_module("refresh")
    shared_claim = {"text": "Shared PhotoAgent claim.", "status": "extracted"}
    canonical_claim = {"text": "Canonical-only claim.", "status": "extracted"}
    alias_claim = {"text": "Alias-only claim.", "status": "extracted"}
    set_live_documents(
        recorded_hub_client,
        photoagent_records(),
        {
            "71050": [shared_claim, alias_claim],
            "Ws8swqL5ob": [shared_claim, canonical_claim],
        },
    )
    recorded_hub_client.spaces = []

    snapshot = refresh.fetch_live_snapshot(recorded_hub_client, OBSERVED_AT)

    assert snapshot["candidates"][0]["live_claims"] == [
        shared_claim,
        canonical_claim,
        alias_claim,
    ]


def test_same_title_and_arxiv_with_conflicting_authors_remain_distinct(
    recorded_hub_client,
):
    refresh = load_module("refresh")
    papers = [
        {
            "arxiv": "2602.99999",
            "authors": ["Alice Author"],
            "or": "",
            "orid": "71099",
            "pid": "71099",
            "title": "A Shared Title",
        },
        {
            "arxiv": "2602.99999",
            "authors": ["Bob Author"],
            "or": "https://openreview.net/forum?id=Distinct99",
            "orid": "Distinct99",
            "pid": "66999",
            "title": "A Shared Title",
        },
    ]
    set_live_documents(
        recorded_hub_client,
        papers,
        {"71099": [], "Distinct99": []},
    )
    recorded_hub_client.spaces = []

    snapshot = refresh.fetch_live_snapshot(recorded_hub_client, OBSERVED_AT)

    assert [candidate["paper_id"] for candidate in snapshot["candidates"]] == [
        "71099",
        "Distinct99",
    ]
    assert all(
        "alias_paper_ids" not in candidate for candidate in snapshot["candidates"]
    )


@pytest.mark.parametrize("authors", [["Same Author"], None])
def test_empty_normalized_source_identities_do_not_merge_candidates(
    recorded_hub_client,
    authors,
):
    refresh = load_module("refresh")
    malformed_sources = [
        ("arxiv", "arxiv:"),
        ("arxiv", "https://arxiv.org/abs/"),
        ("arxiv", "https://arxiv.org/pdf/.pdf"),
        ("alphaxiv", "https://alphaxiv.org/abs/"),
    ]
    papers = []
    for index, (field, source) in enumerate(malformed_sources):
        paper = {
            field: source,
            "orid": f"Malformed{index}",
            "pid": f"71{index:03d}",
            "title": f"Distinct malformed paper {index}",
        }
        if authors is not None:
            paper["authors"] = authors
        papers.append(paper)
    claims = {paper["orid"]: [] for paper in papers}
    set_live_documents(recorded_hub_client, papers, claims)
    recorded_hub_client.spaces = []

    snapshot = refresh.fetch_live_snapshot(recorded_hub_client, OBSERVED_AT)

    assert [candidate["paper_id"] for candidate in snapshot["candidates"]] == [
        "Malformed0",
        "Malformed1",
        "Malformed2",
        "Malformed3",
    ]
    assert all(
        "alias_paper_ids" not in candidate for candidate in snapshot["candidates"]
    )


def test_valid_exact_normalized_source_identity_still_merges(recorded_hub_client):
    refresh = load_module("refresh")
    papers = [
        {
            "arxiv": "arxiv:2602.12345",
            "orid": "71111",
            "pid": "71111",
            "title": "Numeric source record",
        },
        {
            "alphaxiv": "https://alphaxiv.org/abs/2602.12345",
            "or": "https://openreview.net/forum?id=ValidSource",
            "orid": "ValidSource",
            "pid": "66111",
            "title": "Canonical source record",
        },
    ]
    set_live_documents(
        recorded_hub_client,
        papers,
        {"71111": [], "ValidSource": []},
    )
    recorded_hub_client.spaces = []

    snapshot = refresh.fetch_live_snapshot(recorded_hub_client, OBSERVED_AT)

    assert [candidate["paper_id"] for candidate in snapshot["candidates"]] == [
        "ValidSource"
    ]
    assert snapshot["candidates"][0]["alias_paper_ids"] == ["71111"]


def test_live_snapshot_id_is_content_addressed(
    tmp_path, recorded_hub_client, assessments_path
):
    refresh = load_module("refresh")
    store = load_module("store")
    paths = store.StatePaths(tmp_path / "state" / "repro-loop.json")
    store.atomic_json_write(paths.index, store.new_index(), store.validate_index)
    snapshot = refresh.fetch_live_snapshot(
        recorded_hub_client,
        OBSERVED_AT,
        refresh.load_assessments(assessments_path),
    )

    snapshot_id = refresh.persist_snapshot(paths, snapshot)

    assert snapshot_id == hashlib.sha256(canonical_json(snapshot)).hexdigest()
    assert snapshot["sources"]["challenge"]["revision"] == "challenge-sha"
    assert snapshot["sources"]["verdicts"]["revision"] == "verdict-sha"
    persisted = store.read_json(paths.root / "snapshots" / f"{snapshot_id}.json")
    assert persisted == {"snapshot_id": snapshot_id, **snapshot}
    original = (paths.root / "snapshots" / f"{snapshot_id}.json").read_bytes()
    assert refresh.persist_snapshot(paths, snapshot) == snapshot_id
    assert (
        paths.root / "snapshots" / f"{snapshot_id}.json"
    ).read_bytes() == original


def test_refresh_calls_only_pinned_hub_reads(
    recorded_hub_client, assessments_path
):
    refresh = load_module("refresh")
    refresh.fetch_live_snapshot(
        recorded_hub_client,
        OBSERVED_AT,
        refresh.load_assessments(assessments_path),
    )

    downloads = [
        call for call in recorded_hub_client.calls if call[0] == "hf_hub_download"
    ]
    assert downloads == [
        (
            "hf_hub_download",
            "ICML-2026-agent-repro/challenge",
            "index.json",
            "dataset",
            "challenge-sha",
        ),
        (
            "hf_hub_download",
            "ICML-2026-agent-repro/challenge",
            "challenge.json",
            "dataset",
            "challenge-sha",
        ),
        (
            "hf_hub_download",
            "ICML-2026-agent-repro/verdicts",
            "verdicts.json",
            "dataset",
            "verdict-sha",
        ),
    ]


def test_persisted_assessed_snapshot_can_admit_candidate(
    tmp_path, recorded_hub_client, assessments_path
):
    refresh = load_module("refresh")
    scheduler = load_module("scheduler")
    store = load_module("store")
    paths = store.StatePaths(tmp_path / "repro-loop.json")
    store.atomic_json_write(paths.index, store.new_index(), store.validate_index)
    snapshot = refresh.fetch_live_snapshot(
        recorded_hub_client,
        OBSERVED_AT,
        refresh.load_assessments(assessments_path),
    )
    snapshot_id = refresh.persist_snapshot(paths, snapshot)

    report = scheduler.scheduler_pass(
        paths,
        snapshot_id,
        datetime.fromisoformat(OBSERVED_AT).astimezone(timezone.utc),
    )

    assert report.paper_ids == ("paper-a",)


def test_read_fresh_snapshot_rejects_content_corruption(
    tmp_path, recorded_hub_client, assessments_path
):
    refresh = load_module("refresh")
    scheduler = load_module("scheduler")
    store = load_module("store")
    paths = store.StatePaths(tmp_path / "repro-loop.json")
    store.atomic_json_write(paths.index, store.new_index(), store.validate_index)
    snapshot = refresh.fetch_live_snapshot(
        recorded_hub_client,
        OBSERVED_AT,
        refresh.load_assessments(assessments_path),
    )
    snapshot_id = refresh.persist_snapshot(paths, snapshot)
    snapshot_path = paths.root / "snapshots" / f"{snapshot_id}.json"
    corrupted = store.read_json(snapshot_path)
    corrupted["candidates"][0]["title"] = "corrupted"
    store.atomic_json_write(snapshot_path, corrupted, store.validate_snapshot)

    with pytest.raises(ValueError, match="snapshot_id"):
        scheduler.read_fresh_snapshot(
            paths,
            snapshot_id,
            datetime.fromisoformat(OBSERVED_AT).astimezone(timezone.utc),
        )
