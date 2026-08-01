from udm_grpo_repro.core import (
    AUDIT,
    CLAIMS,
    build_evidence_bundle,
    classify_claims,
    summarize_audit,
)


def test_source_audit_confirms_core_training_switches():
    summary = summarize_audit(AUDIT)

    assert summary["uses_forward_process"] is True
    assert summary["uses_clean_response"] is True
    assert summary["train_steps"] == 3
    assert summary["train_start"] == [0, 3]
    assert summary["guidance_scale"] == 1


def test_mechanism_claims_verified_but_metric_claims_inconclusive():
    statuses = {claim["sha256"]: claim["status"] for claim in classify_claims(CLAIMS, AUDIT)}

    assert statuses["0a031c59fd7bbe00c59e97c020bf7d036d73fb2cb3bbd6385b68a37b063a4d74"] == "verified"
    assert statuses["716661282da161f7fa1075a8570fa6017884d3c3f6165e5c85963f76637349d7"] == "verified"
    assert statuses["e00307483455d73ef3222df24dfde3d6646fcdf26017b2af7905cf722d5cdd61"] == "verified"
    assert statuses["a25ea6f983664d3f04398287721601f9518163aac45bb16a9a2c1dba6dce396f"] == "inconclusive"
    assert statuses["1d8a28f488fcd9fafe58329ad1b52b829c49cb2423d46c3b8e80f39a9e3b1eaf"] == "inconclusive"
    assert statuses["b24538ba4305cac11eef9766e68faf5811ae1a5e15b33930d5ca20eacd477546"] == "inconclusive"


def test_bundle_has_no_reproduced_metric_or_ablation_measurements():
    bundle = build_evidence_bundle()

    assert bundle["paper_id"] == "WJcFtJriqv"
    assert bundle["upstream"]["code"].endswith("@d1bec49f4500873606f8345d81692143de059891")
    assert bundle["reproduced_metric_measurements"] == []
    assert bundle["reproduced_ablation_measurements"] == []
    assert build_evidence_bundle() == build_evidence_bundle()
