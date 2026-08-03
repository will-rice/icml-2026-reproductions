import copy
import pytest

from lrr_repro.claim_scope import audit_nonlinear_invariant_claim
from lrr_repro.provenance import load_paper_context


@pytest.fixture
def paper_context(project_root):
    return load_paper_context(project_root)


def test_exact_live_claim_is_falsified(paper_context, cache_dir):
    audit = audit_nonlinear_invariant_claim(paper_context, cache_dir)
    assert audit.status == "falsified"
    assert audit.exact_claim_supported is False
    assert set(audit.contradictions) == {
        "v1 Table 2 is a learned post-condition example, not a backend comparison",
        "LR versus MILP sample/runtime results are for RSR-Bench, not NLA-DigBench",
        "NLA-DigBench compares Bitween with DIG and SymInfer, not MILP",
        "v5 Table 2 reports novel Agentic Bitween query functions",
    }


def test_missing_locator_is_inconclusive(paper_context, cache_dir):
    ctx_copy = copy.deepcopy(paper_context)
    del ctx_copy["versions"]["v1"]["nla_digbench"]
    assert audit_nonlinear_invariant_claim(ctx_copy, cache_dir).status == "inconclusive"


def test_tampered_locator_fails_pdf_verification(paper_context, cache_dir):
    ctx_copy = copy.deepcopy(paper_context)
    ctx_copy["versions"]["v1"]["rsr_bench"]["lr_samples"] = 999999
    assert audit_nonlinear_invariant_claim(ctx_copy, cache_dir).status == "inconclusive"


def test_plausible_tampered_lr_samples_fails_pdf_verification(paper_context, cache_dir):
    ctx_copy = copy.deepcopy(paper_context)
    # 20 occurs elsewhere in v1 (e.g. Table 2), but is not the lr_samples in Section 5.3.1
    ctx_copy["versions"]["v1"]["rsr_bench"]["lr_samples"] = 20
    assert audit_nonlinear_invariant_claim(ctx_copy, cache_dir).status == "inconclusive"


def test_tampered_table2_samples_fails_pdf_verification(paper_context, cache_dir):
    ctx_copy = copy.deepcopy(paper_context)
    ctx_copy["versions"]["v1"]["table_2"]["sample_count"] = 50
    assert audit_nonlinear_invariant_claim(ctx_copy, cache_dir).status == "inconclusive"


def test_tampered_milp_samples_fails_pdf_verification(paper_context, cache_dir):
    ctx_copy = copy.deepcopy(paper_context)
    ctx_copy["versions"]["v1"]["rsr_bench"]["milp_samples"] = 20
    assert audit_nonlinear_invariant_claim(ctx_copy, cache_dir).status == "inconclusive"
