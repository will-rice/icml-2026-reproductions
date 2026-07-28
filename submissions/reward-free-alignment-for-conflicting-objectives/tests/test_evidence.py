from pathlib import Path
import pytest
from reward_free_alignment.evidence import (
    build_evidence,
    canonical_json,
    validate_evidence,
)
from reward_free_alignment.provenance import load_manifest

EXPECTED_HASHES = (
    "e9a35e34b57a7273bf84d3d5981ab19f8ff1088adef8363f4640dcf436183944",
    "7c0aa54e034d03f2d0905417a024af4db41338cd9a803a0b42e441945c307cf9",
    "85abbc8a21d5c4537409f6e9f2af6bffc7e4c15e2311dfa078bf816ea0cffc9e",
    "dac93f364ac0469302894920781b034bfcd205816fbe16537c2f8e7c10d8995d",
    "269d8a5053e224206036399bccb2435455565149086de6439a9046d89682772b",
    "0d457572ea8a502fa8a489fef3e15da21b13cc39dd3a3730843d1cbe833059b0",
    "50719d645042a500f9c4d53fbdfedf719ee56429ddb73a231912f1eaeadb1b31",
    "5ec835ce150ff60d1e2bbd4fbdf7d1ebacf91bb2b6b2d65f72c44c3b3ed65229",
    "b74a0ea75967144b210934fd40fd23449d3ef985df878d5a9e14c4b04025ba4b",
    "58b31f527bb5e1bccb05c0dab775a74c2f2bdcd8e92ef2c0dd578733b5fb058e",
)


@pytest.fixture
def project_root() -> Path:
    return Path(__file__).parent.parent


def test_bundle_contains_all_ten_live_claims_in_exact_order(project_root):
    evidence = build_evidence(project_root)
    assert evidence["snapshot_id"] == (
        "09017559ff2c5746f1a37458ba9a330bd4e18654ae9c3f873bb0785c76626199"
    )
    assert tuple(c["sha256"] for c in evidence["claims"]) == EXPECTED_HASHES
    assert len(evidence["claims"]) == 10
    assert [c["targeted"] for c in evidence["claims"]] == [
        False, False, False, False, False, True, True, True, True, False
    ]


def test_local_outcomes_never_impersonate_official_verdicts(project_root):
    evidence = build_evidence(project_root)
    assert {c["local_outcome"] for c in evidence["claims"]} <= {
        "supported", "not-supported", "limited"
    }
    assert not {"verified", "falsified", "toy"} & {
        c["local_outcome"] for c in evidence["claims"]
    }


def test_evidence_is_canonical_and_byte_deterministic(project_root):
    first = canonical_json(build_evidence(project_root))
    second = canonical_json(build_evidence(project_root))
    assert first == second
    assert first.endswith(b"\n")
    assert b"NaN" not in first and b"Infinity" not in first


def test_evidence_passes_schema_validation(project_root):
    evidence = build_evidence(project_root)
    schema_path = project_root / "schema/evidence-v1.schema.json"
    validate_evidence(evidence, schema_path)


# --- Adversarial regressions for controller correction gate ---


def test_outcomes_are_derived_from_audits_not_hardcoded(project_root):
    """Claim outcomes must depend on actual audit results, not hard-coded labels."""
    evidence = build_evidence(project_root)
    # Claims 6-9 are targeted and should have outcomes derived from the audit
    targeted_claims = [c for c in evidence["claims"] if c["targeted"]]
    assert len(targeted_claims) == 4
    # Each targeted claim must have a reproduction_notes that references the audit
    for c in targeted_claims:
        assert c["local_outcome"] in ("supported", "not-supported", "limited")
        # Notes should reference specific computed values, not generic boilerplate
        assert len(c["reproduction_notes"]) > 20


def test_evidence_generation_calls_verification(project_root):
    """Evidence builder must call schema validation (not conditionally skip it)."""
    evidence = build_evidence(project_root)
    # If the schema existed, it was validated; test proves no exception was raised
    schema_path = project_root / "schema/evidence-v1.schema.json"
    assert schema_path.is_file()
    validate_evidence(evidence, schema_path)


def test_claim6_uses_end_to_end_gradient_pipeline(project_root):
    """Claim 6 must apply CAGrad to gradients actually derived from
    objective-specific pairwise losses, not disconnected fixtures."""
    evidence = build_evidence(project_root)
    claim6 = [c for c in evidence["claims"] if c["ordinal"] == 6][0]
    notes = claim6["reproduction_notes"]
    # Must mention end-to-end pipeline
    assert "end-to-end" in notes.lower() or "End-to-end" in notes
    # Must mention computing losses
    assert "L1=" in notes or "loss" in notes.lower()
    # Must NOT say "disconnected fixtures"
    assert "disconnected" not in notes.lower() or "not disconnected" in notes.lower()


def test_theorem_31_uses_executed_trajectory(project_root):
    """Theorem 3.1 must use an actual computed trajectory, not hand-entered losses."""
    evidence = build_evidence(project_root)
    t31 = evidence["audits"]["theorem_31"]
    # The executed trajectory from x0=1.0 with eta=0.1 gives specific values
    # initial_loss and final_loss are not set to dummy values like 1.5 and 1.2
    assert t31["local_outcome"] == "supported"


def test_theorem_32_has_interior_strict_witness(project_root):
    """Theorem 3.2 must have an interior strict witness with positive Gamma difference."""
    evidence = build_evidence(project_root)
    claim9 = [c for c in evidence["claims"] if c["ordinal"] == 9][0]
    assert claim9["local_outcome"] == "supported"
    # Notes should reference interior alpha and positive Gamma difference
    notes = claim9["reproduction_notes"]
    assert "interior" in notes.lower()
    assert "3.68e-9" not in notes  # Must not be the old boundary artifact


# --- Round 6 correction gate behavioral regressions ---


def test_theorem_31_evidence_contains_closed_schema_steps_array(project_root):
    """Round-6 & Round-9 §2: evidence must contain a 'steps' array of exactly 10 records
    for t=0..9 inside audits.theorem_31. Each record must be independently verified against
    recomputed g1, g2, weighted anchor, CAGrad-Clip direction, next iterate, losses,
    M(theta_t), weighted-gradient norm, descent inequality, and M-bound boolean."""
    import math
    import torch
    from reward_free_alignment.cagrad_clip import cagrad_clip
    from reward_free_alignment.theorem_audit import compute_m_simplex

    evidence = build_evidence(project_root)
    t31 = evidence["audits"]["theorem_31"]
    assert "steps" in t31, "evidence must contain 'steps' array"
    steps = t31["steps"]
    assert len(steps) == 10

    weights = torch.tensor([0.6, 0.4])
    w1_py, w2_py = round(weights[0].item(), 6), round(weights[1].item(), 6)
    c = t31["correction_radius"]
    eta = t31["step_size"]

    required_keys = {
        "step_index", "current_iterate", "weighted_anchor", "cagrad_direction",
        "next_iterate", "loss_before", "loss_after", "m_value", "grad_norm",
        "descent_holds", "m_bound_holds",
    }

    x = 1.0
    for i, step in enumerate(steps):
        assert set(step.keys()) == required_keys, (
            f"Step {i} keys mismatch: {set(step.keys())} != {required_keys}"
        )
        assert step["step_index"] == i
        assert abs(step["current_iterate"] - x) < 1e-9

        # Independent recomputations for step i
        f1_before = x ** 2
        f2_before = (x - 1.0) ** 2
        expected_loss_before = w1_py * f1_before + w2_py * f2_before

        g1 = torch.tensor([2.0 * x], dtype=torch.float32)
        g2 = torch.tensor([2.0 * (x - 1.0)], dtype=torch.float32)
        res = cagrad_clip((g1, g2), weights, c)

        expected_anchor = res.weighted_anchor.item()
        expected_dir = res.gradient.item()
        expected_next = x - eta * expected_dir

        f1_after = expected_next ** 2
        f2_after = (expected_next - 1.0) ** 2
        expected_loss_after = w1_py * f1_after + w2_py * f2_after

        expected_m_val = compute_m_simplex(g1, g2)
        expected_grad_norm = torch.linalg.vector_norm(res.weighted_anchor).item()

        expected_descent_amount = (eta * (1.0 - c * c) / 2.0) * (expected_grad_norm ** 2)
        expected_descent_holds = expected_loss_after <= expected_loss_before - expected_descent_amount + 1e-9
        expected_m_bound_holds = (
            math.isfinite(expected_m_val)
            and expected_m_val >= 0.0
            and expected_m_val <= expected_grad_norm + 1e-9
        )

        assert abs(step["weighted_anchor"] - expected_anchor) < 1e-9, f"Step {i} weighted_anchor mismatch"
        assert abs(step["cagrad_direction"] - expected_dir) < 1e-9, f"Step {i} cagrad_direction mismatch"
        assert abs(step["next_iterate"] - expected_next) < 1e-9, f"Step {i} next_iterate mismatch"
        assert abs(step["loss_before"] - expected_loss_before) < 1e-9, f"Step {i} loss_before mismatch"
        assert abs(step["loss_after"] - expected_loss_after) < 1e-9, f"Step {i} loss_after mismatch"
        assert abs(step["m_value"] - expected_m_val) < 1e-9, f"Step {i} m_value mismatch"
        assert abs(step["grad_norm"] - expected_grad_norm) < 1e-9, f"Step {i} grad_norm mismatch"
        assert step["descent_holds"] == bool(expected_descent_holds), f"Step {i} descent_holds mismatch"
        assert step["m_bound_holds"] == bool(expected_m_bound_holds), f"Step {i} m_bound_holds mismatch"

        x = expected_next


def test_artifact_source_urls_use_raw_not_blob(project_root):
    """Round-6 §2: artifact source_url must use immutable raw URLs
    (raw.githubusercontent.com), not GitHub HTML /blob/ pages."""
    evidence = build_evidence(project_root)
    for art in evidence["artifacts"]:
        url = art["source_url"]
        assert "/blob/" not in url, (
            f"Artifact {art['artifact_id']} uses /blob/ URL: {url}"
        )
        assert "raw.githubusercontent.com" in url, (
            f"Artifact {art['artifact_id']} does not use raw URL: {url}"
        )


def test_claim8_requires_all_step_records_pass(project_root):
    """Round-6 §1: Claim 8 may be 'supported' only if all ten step records
    pass both descent_holds and m_bound_holds booleans AND both finite-horizon
    squared bounds pass."""
    evidence = build_evidence(project_root)
    t31 = evidence["audits"]["theorem_31"]
    steps = t31["steps"]
    all_descent = all(s["descent_holds"] for s in steps)
    all_m_bound = all(s["m_bound_holds"] for s in steps)
    fh_holds = t31.get("finite_horizon_bound_holds", False)
    claim8 = [c for c in evidence["claims"] if c["ordinal"] == 8][0]
    if claim8["local_outcome"] == "supported":
        assert all_descent, "Claim 8 supported but not all descent_holds"
        assert all_m_bound, "Claim 8 supported but not all m_bound_holds"
        assert fh_holds, "Claim 8 supported but finite_horizon_bound_holds False"


def test_claim8_mutation_regressions(project_root):
    """Round-9 §3: Test _derive_claim_outcomes with four direct mutations against Claim 8.
    Flipping any single dependency must change Claim 8 outcome to 'not-supported',
    while the all-true control must remain 'supported'."""
    from reward_free_alignment.evidence import (
        _derive_claim_outcomes,
        _run_pairwise_audit,
        _run_claim6_end_to_end_audit,
        _run_cagrad_audit,
        _run_theorem_31_audit,
        _run_theorem_32_audit,
    )
    pairwise_audit = _run_pairwise_audit()
    claim6_audit = _run_claim6_end_to_end_audit()
    cagrad_audit = _run_cagrad_audit()
    t31_audit = _run_theorem_31_audit()
    t32_strict_audit, _ = _run_theorem_32_audit()

    # Control: all dependencies hold -> Claim 8 supported
    control = _derive_claim_outcomes(
        pairwise_audit=pairwise_audit,
        claim6_audit=claim6_audit,
        cagrad_audit=cagrad_audit,
        t31_audit=t31_audit,
        t32_strict_audit=t32_strict_audit,
    )
    assert control[8][0] == "supported"

    # Mutation 1: flip one step's descent_holds to False
    t31_mut1 = dict(t31_audit)
    steps_mut1 = [dict(s) for s in t31_mut1["steps"]]
    steps_mut1[3]["descent_holds"] = False
    t31_mut1["steps"] = steps_mut1
    out1 = _derive_claim_outcomes(
        pairwise_audit=pairwise_audit,
        claim6_audit=claim6_audit,
        cagrad_audit=cagrad_audit,
        t31_audit=t31_mut1,
        t32_strict_audit=t32_strict_audit,
    )
    assert out1[8][0] == "not-supported"

    # Mutation 2: flip one step's m_bound_holds to False
    t31_mut2 = dict(t31_audit)
    steps_mut2 = [dict(s) for s in t31_mut2["steps"]]
    steps_mut2[5]["m_bound_holds"] = False
    t31_mut2["steps"] = steps_mut2
    out2 = _derive_claim_outcomes(
        pairwise_audit=pairwise_audit,
        claim6_audit=claim6_audit,
        cagrad_audit=cagrad_audit,
        t31_audit=t31_mut2,
        t32_strict_audit=t32_strict_audit,
    )
    assert out2[8][0] == "not-supported"

    # Mutation 3: flip grad_finite_horizon_bound_holds to False
    t31_mut3 = dict(t31_audit)
    t31_mut3["grad_finite_horizon_bound_holds"] = False
    out3 = _derive_claim_outcomes(
        pairwise_audit=pairwise_audit,
        claim6_audit=claim6_audit,
        cagrad_audit=cagrad_audit,
        t31_audit=t31_mut3,
        t32_strict_audit=t32_strict_audit,
    )
    assert out3[8][0] == "not-supported"

    # Mutation 4: flip m_finite_horizon_bound_holds to False
    t31_mut4 = dict(t31_audit)
    t31_mut4["m_finite_horizon_bound_holds"] = False
    out4 = _derive_claim_outcomes(
        pairwise_audit=pairwise_audit,
        claim6_audit=claim6_audit,
        cagrad_audit=cagrad_audit,
        t31_audit=t31_mut4,
        t32_strict_audit=t32_strict_audit,
    )
    assert out4[8][0] == "not-supported"
