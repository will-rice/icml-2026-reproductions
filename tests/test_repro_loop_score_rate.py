from pathlib import Path
import hashlib
import importlib
import sys

import pytest


SCRIPTS = Path(__file__).resolve().parents[1] / "skills/icml-repro-loop/scripts"
sys.path.insert(0, str(SCRIPTS))
score_rate = importlib.import_module("score_rate")


def envelope(*, hours=2.0, deadline_probability=0.8, reusable=False):
    return {
        "claim_expectations": [
            {
                "challenge_claim_sha256": hashlib.sha256(b"Claim A").hexdigest(),
                "p_verified": 0.5,
                "p_falsified": 0.25,
                "p_toy": 0.1,
            },
            {
                "challenge_claim_sha256": hashlib.sha256(b"Claim B").hexdigest(),
                "p_verified": 0.0,
                "p_falsified": 0.5,
                "p_toy": 0.25,
            },
        ],
        "judged_before_deadline_probability": deadline_probability,
        "remaining_hours_p90": hours,
        "reusable_implementation": reusable,
        "direct_artifact_score": 4,
        "full_score_claim_paths": 2,
        "remaining_time_variance_hours2": 0.25,
        "primary_risk": "Artifact schema may have drifted.",
    }


def candidate(paper_id="paper-a", *, cost=0.0, **updates):
    value = {
        "paper_id": paper_id,
        "estimated_api_cost_usd": cost,
        "score_rate": envelope(),
    }
    value.update(updates)
    return value


def test_official_claim_point_mapping_is_exact():
    assert score_rate.claim_points("verified") == 2
    assert score_rate.claim_points("falsified") == 2
    assert score_rate.claim_points("toy") == 1
    assert score_rate.claim_points("inconclusive") == 0
    with pytest.raises(ValueError, match="status"):
        score_rate.claim_points("unknown")


def test_expected_points_and_priority_follow_approved_formula():
    value = envelope()
    assert score_rate.expected_points(value) == pytest.approx(2.85)
    assert score_rate.priority(value) == pytest.approx(1.14)


def test_envelope_binds_every_live_claim_once():
    live_claims = [{"text": "Claim A"}, {"text": "Claim B"}]
    score_rate.validate_envelope(envelope(), live_claims)
    invalid = envelope()
    invalid["claim_expectations"].pop()
    with pytest.raises(ValueError, match="claim_expectations"):
        score_rate.validate_envelope(invalid, live_claims)


def test_probability_mass_cannot_exceed_one():
    invalid = envelope()
    invalid["claim_expectations"][0]["p_toy"] = 0.3
    with pytest.raises(ValueError, match="probability"):
        score_rate.validate_envelope(
            invalid, [{"text": "Claim A"}, {"text": "Claim B"}]
        )


@pytest.mark.parametrize(
    ("left", "right", "expected_first"),
    [
        (
            candidate("paper-a"),
            candidate(
                "paper-b",
                score_rate=envelope(deadline_probability=0.7),
            ),
            "paper-a",
        ),
        (
            candidate("paper-a"),
            candidate("paper-b", score_rate=envelope(hours=3.0)),
            "paper-a",
        ),
        (
            candidate("paper-a", score_rate=envelope(reusable=True)),
            candidate("paper-b"),
            "paper-a",
        ),
        (
            candidate("paper-a"),
            candidate("paper-b", score_rate={**envelope(), "direct_artifact_score": 3}),
            "paper-a",
        ),
        (
            candidate("paper-a"),
            candidate("paper-b", score_rate={**envelope(), "full_score_claim_paths": 1}),
            "paper-a",
        ),
        (
            candidate("paper-a"),
            candidate(
                "paper-b",
                score_rate={**envelope(), "remaining_time_variance_hours2": 0.5},
            ),
            "paper-a",
        ),
        (candidate("paper-a", cost=0.0), candidate("paper-b", cost=1.0), "paper-a"),
        (candidate("paper-a"), candidate("paper-b"), "paper-a"),
    ],
    ids=[
        "deadline-probability",
        "remaining-p90-hours",
        "reusable-implementation",
        "direct-artifacts",
        "full-score-paths",
        "time-variance",
        "paid-cost",
        "paper-id",
    ],
)
def test_ranking_key_uses_approved_deterministic_order(left, right, expected_first):
    ranked = sorted([left, right], key=score_rate.ranking_key)

    assert ranked[0]["paper_id"] == expected_first
