from fac_evidence.bundle import build_evidence_bundle


def test_evidence_bundle_keeps_reported_values_out_of_reproduced_measurements():
    bundle = build_evidence_bundle()

    claim_by_sha = {claim["claim_sha256"]: claim for claim in bundle["claims"]}

    performance_claim = claim_by_sha["aa863c883e3570fb5243d7d552f94d618ab83d157571a3ce3dec6a6886424789"]

    assert performance_claim["status"] == "inconclusive"
    assert performance_claim["reproduced_measurements"] == []
    assert performance_claim["paper_reported_context"]["main_result"]["ours"]["toxicity_auprc"] == 62.60
