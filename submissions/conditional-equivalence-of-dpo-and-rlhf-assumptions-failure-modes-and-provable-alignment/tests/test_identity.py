from hashlib import sha256

from conditional_dpo_repro.claims import (
    ATTEMPT_ID,
    LIVE_CLAIM_HASHES,
    LIVE_CLAIMS,
    PAPER_ID,
    SNAPSHOT_ID,
    UPSTREAM_REVISION,
    load_claim_bindings,
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


def test_identity_is_bound_to_admitted_attempt():
    assert PAPER_ID == "7UEBX1KU1y"
    assert ATTEMPT_ID == "933665ed-b7ed-4d73-9b07-35704660a184"
    assert SNAPSHOT_ID == (
        "09017559ff2c5746f1a37458ba9a330bd4e18654ae9c3f873bb0785c76626199"
    )
    assert UPSTREAM_REVISION == "arxiv:2605.20834v1"


def test_all_six_live_claims_equal_admitted_constants(project_root):
    claims = load_claim_bindings(project_root / "sources/paper.json")
    assert LIVE_CLAIMS == EXPECTED_LIVE_CLAIMS
    assert LIVE_CLAIM_HASHES == EXPECTED_LIVE_CLAIM_HASHES
    assert tuple(item.challenge_claim for item in claims) == EXPECTED_LIVE_CLAIMS
    assert tuple(
        item.challenge_claim_sha256 for item in claims
    ) == EXPECTED_LIVE_CLAIM_HASHES
    assert sum(item.targeted for item in claims) == 5
    assert tuple(
        sha256(text.encode("utf-8")).hexdigest() for text in EXPECTED_LIVE_CLAIMS
    ) == EXPECTED_LIVE_CLAIM_HASHES
    assert claims[-1].targeted is False
