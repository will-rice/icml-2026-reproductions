from fractions import Fraction
import pytest

from lrr_repro.provenance import load_paper_context
from lrr_repro.theory import (
    FiniteRSR,
    audit_claim_a1,
    modular_addition_rsr,
    nonuniform_rsr,
    verify_theory_locators,
)


def test_correlated_queries_need_only_uniform_marginals():
    rsr = modular_addition_rsr(4)
    audit = audit_claim_a1(
        rsr,
        truth={0: 1, 1: 1, 2: 2, 3: 3},
        hypothesis={0: 0, 1: 1, 2: 2, 3: 3},
        rho=Fraction(1, 2),
        xi=Fraction(1, 4),
    )
    assert audit.marginal_uniform is True
    assert audit.epsilon == Fraction(1, 4)
    assert audit.good_input_fraction >= Fraction(3, 4)
    assert audit.minimum_recovery_probability >= Fraction(1, 2)
    assert audit.implication_holds is True


def test_nonuniform_marginal_is_rejected():
    identity = {0: 0, 1: 1, 2: 2, 3: 3}
    with pytest.raises(ValueError, match="marginal uniformity"):
        audit_claim_a1(nonuniform_rsr(), identity, identity, Fraction(1, 2), Fraction(1, 4))


def test_verify_theory_locators(project_root, cache_dir):
    ctx = load_paper_context(project_root)
    verify_theory_locators(ctx, cache_dir)


def test_verify_theory_locators_rejects_invented_locator(project_root, cache_dir):
    import copy

    ctx = load_paper_context(project_root)
    ctx_tampered = copy.deepcopy(ctx)
    ctx_tampered["versions"]["v5"]["definitions"]["def_4_1"]["name"] = (
        "Invented Nonexistent Definition 99.9"
    )
    with pytest.raises(ValueError, match="PDF text missing"):
        verify_theory_locators(ctx_tampered, cache_dir)
