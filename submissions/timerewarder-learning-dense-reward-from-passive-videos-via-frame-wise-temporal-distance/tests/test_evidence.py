import json
from pathlib import Path

from timerewarder_repro.evidence import (
    CLAIMS,
    CLAIM_SHA256,
    build_evidence_bundle,
    measurement_sha256,
    write_canonical_json,
)

PROJECT = Path(__file__).parents[1]


def _bundle() -> dict[str, object]:
    return build_evidence_bundle(
        PROJECT / "artifacts/manifest.json",
        PROJECT / "artifacts/acquisition.json",
        PROJECT / "artifacts/checkpoints.json",
        PROJECT / "artifacts/source",
        PROJECT / "artifacts/representative.json",
    )


def test_bundle_binds_six_live_claims_and_honest_statuses() -> None:
    bundle = _bundle()

    assert bundle["attempt_id"] == "bf0d2300-4479-4e3c-ba99-bb023ee6751e"
    assert bundle["paper_id"] == "XztRm216YS"
    assert [item["claim"] for item in bundle["claims"]] == list(CLAIMS)
    assert [
        item["challenge_claim_sha256"] for item in bundle["claims"]
    ] == list(CLAIM_SHA256)
    assert [item["status"] for item in bundle["claims"]] == [
        "verified",
        "verified",
        "verified",
        "partial",
        "unavailable",
        "unavailable",
    ]
    assert len(bundle["measurements"]["source_audit"]["function_span_sha256"]) == 33
    assert bundle["measurements"]["formula"]["case_count"] == 106
    assert bundle["measurements"]["formula"]["transition_case_count"] == 3
    assert bundle["measurements"]["theory"]["aliasing_counterexample"][
        "single_frame_average"
    ] == 2.5
    assert bundle["measurements"]["fixture"]["diagnostic_only"] is True


def test_measurement_hash_and_canonical_bytes_are_reproducible(
    tmp_path: Path,
) -> None:
    first = _bundle()
    second = _bundle()
    first_path = tmp_path / "first.json"
    second_path = tmp_path / "second.json"

    write_canonical_json(first, first_path)
    write_canonical_json(second, second_path)

    assert measurement_sha256(first) == first["measurement_sha256"]
    assert first["measurement_sha256"] == second["measurement_sha256"]
    assert first_path.read_bytes() == second_path.read_bytes()
    assert first_path.read_bytes().endswith(b"\n")
    json.loads(
        first_path.read_text(),
        parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)),
    )


def test_unavailable_claims_name_missing_evidence() -> None:
    claims = _bundle()["claims"]

    assert "successful/failed rollout" in claims[4]["limitations"]
    assert "Meta-World" in claims[5]["limitations"]
    assert "200,000" in claims[5]["limitations"]
    assert "comparative" in claims[3]["limitations"]
