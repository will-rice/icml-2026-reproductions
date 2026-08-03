import hashlib
import math
import re


POINTS = {"verified": 2, "falsified": 2, "toy": 1, "inconclusive": 0}
ENVELOPE_KEYS = {
    "claim_expectations",
    "judged_before_deadline_probability",
    "remaining_hours_p90",
    "reusable_implementation",
    "direct_artifact_score",
    "full_score_claim_paths",
    "remaining_time_variance_hours2",
    "primary_risk",
}
CLAIM_KEYS = {
    "challenge_claim_sha256",
    "p_verified",
    "p_falsified",
    "p_toy",
}


def claim_points(status: str) -> int:
    if type(status) is not str or status.casefold() not in POINTS:
        raise ValueError("status")
    return POINTS[status.casefold()]


def _probability(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(field)
    number = float(value)
    if not math.isfinite(number) or not 0.0 <= number <= 1.0:
        raise ValueError(field)
    return number


def validate_envelope(value: object, live_claims: list[dict]) -> None:
    if type(value) is not dict or set(value) != ENVELOPE_KEYS:
        raise ValueError("score_rate")
    expectations = value["claim_expectations"]
    if type(expectations) is not list or len(expectations) != len(live_claims):
        raise ValueError("claim_expectations")
    expected_digests = [
        hashlib.sha256(claim["text"].encode("utf-8")).hexdigest()
        for claim in live_claims
    ]
    actual_digests = []
    for record in expectations:
        if type(record) is not dict or set(record) != CLAIM_KEYS:
            raise ValueError("claim_expectations")
        probabilities = [
            _probability(record[field], field)
            for field in ("p_verified", "p_falsified", "p_toy")
        ]
        if sum(probabilities) > 1.0 + 1e-12:
            raise ValueError("probability")
        digest = record["challenge_claim_sha256"]
        if (
            type(digest) is not str
            or re.fullmatch(r"[0-9a-f]{64}", digest) is None
        ):
            raise ValueError("challenge_claim_sha256")
        actual_digests.append(digest)
    if actual_digests != expected_digests:
        raise ValueError("claim_expectations")
    _probability(
        value["judged_before_deadline_probability"],
        "judged_before_deadline_probability",
    )
    hours = value["remaining_hours_p90"]
    variance = value["remaining_time_variance_hours2"]
    if (
        isinstance(hours, bool)
        or not isinstance(hours, (int, float))
        or not math.isfinite(hours)
        or hours <= 0
    ):
        raise ValueError("remaining_hours_p90")
    if (
        isinstance(variance, bool)
        or not isinstance(variance, (int, float))
        or not math.isfinite(variance)
        or variance < 0
    ):
        raise ValueError("remaining_time_variance_hours2")
    if type(value["reusable_implementation"]) is not bool:
        raise ValueError("reusable_implementation")
    risk = value["primary_risk"]
    if type(risk) is not str or not risk.strip():
        raise ValueError("primary_risk")
    artifact_score = value["direct_artifact_score"]
    full_paths = value["full_score_claim_paths"]
    if type(artifact_score) is not int or artifact_score not in range(6):
        raise ValueError("direct_artifact_score")
    if type(full_paths) is not int or not 0 <= full_paths <= len(live_claims):
        raise ValueError("full_score_claim_paths")


def expected_points(value: dict) -> float:
    total = 0.0
    for claim in value["claim_expectations"]:
        total += 2 * claim["p_verified"]
        total += 2 * claim["p_falsified"]
        total += claim["p_toy"]
    return total


def priority(value: dict) -> float:
    return (
        expected_points(value)
        * value["judged_before_deadline_probability"]
        / max(float(value["remaining_hours_p90"]), 0.25)
    )


def ranking_key(candidate: dict) -> tuple:
    envelope = candidate["score_rate"]
    return (
        -priority(envelope),
        -int(envelope["reusable_implementation"]),
        -envelope["direct_artifact_score"],
        -envelope["full_score_claim_paths"],
        envelope["remaining_time_variance_hours2"],
        candidate["estimated_api_cost_usd"],
        candidate["paper_id"],
    )
