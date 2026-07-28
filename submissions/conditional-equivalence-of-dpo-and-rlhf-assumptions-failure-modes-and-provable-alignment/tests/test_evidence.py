from hashlib import sha256

from conditional_dpo_repro.evidence import (
    build_evidence,
    canonical_json_bytes,
    validate_evidence,
)

EXPECTED_LIVE_CLAIMS = (
    "The paper proves DPO-RLHF equivalence is conditional on the RLHF-optimal "
    "policy preferring human-preferred responses (Section 3).",
    "When the equivalence assumption fails, DPO optimizes relative advantage "
    "over the reference policy rather than absolute human-preference alignment "
    "(Section 3).",
    "The paper characterizes undesirable solution spaces in which policies "
    "reduce DPO loss while preferring dispreferred responses (Section 3).",
    "Constrained Preference Optimization augments RLHF with constraints and "
    "derives a stationary DPO-like loss with an adaptive reference-based margin "
    "(Section 4.3).",
    "The paper gives a soft-margin ranking interpretation showing DPO can "
    "implement margin ranking with potentially negative targets (Section 5).",
    "Experiments on standard benchmarks report state-of-the-art performance "
    "for CPO (Section 6).",
)
EXPECTED_LIVE_CLAIM_HASHES = (
    "588c9334124771dc2ff7fc51494f4328329ab13dc21d4522a0e91b6f6417240a",
    "4820743d0eac6cc30b4a75d2be41f49193b0ea4ad4168bea2200a9f16cc77a86",
    "6c26fe711e2f10b44cb933b89b12982fef3cf3bcc760668a0b0fa9d15e1965dc",
    "a80267886061211c131041549df22264e0c713a9759a76f0ab37bac69a436af1",
    "7d797875f18478f305a8dc08d860a29ba4f15c3b97fb4c9d41e55363975553be",
    "8df1fece656f02adbdf85fb78bc8993591f1abc9ee78c957388ab4b4eac37dcd",
)


def test_bundle_has_all_live_claims_in_order(project_root):
    value = build_evidence(project_root)
    validate_evidence(value, project_root / "schema/evidence-v1.schema.json")
    assert value["paper_id"] == "7UEBX1KU1y"
    assert value["attempt_id"] == "933665ed-b7ed-4d73-9b07-35704660a184"
    assert len(value["claims"]) == 6
    assert [claim["targeted"] for claim in value["claims"]] == [
        True, True, True, True, True, False
    ]
    assert tuple(
        claim["challenge_claim"] for claim in value["claims"]
    ) == EXPECTED_LIVE_CLAIMS
    assert tuple(
        claim["challenge_claim_sha256"] for claim in value["claims"]
    ) == EXPECTED_LIVE_CLAIM_HASHES
    assert value["claims"][-1]["outcome"] == "not_reproduced"


def test_bundle_is_byte_deterministic(project_root):
    first = canonical_json_bytes(build_evidence(project_root))
    second = canonical_json_bytes(build_evidence(project_root))
    assert first == second
    assert first.endswith(b"\n")
    assert b"NaN" not in first and b"Infinity" not in first
