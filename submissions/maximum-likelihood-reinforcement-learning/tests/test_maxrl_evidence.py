from __future__ import annotations

import math
from pathlib import Path

from maxrl_repro.evidence import (
    audit_source,
    bernoulli_checks,
    build_evidence,
    maxrl_estimator_expectation,
    ml_gradient,
    truncated_maxrl_gradient,
)


SOURCE_ROOT = Path("/tmp/maxrl-src-3cd57e67")


def test_estimator_matches_truncated_gradient():
    for p in (0.05, 0.2, 0.5, 0.8):
        for n in (1, 2, 4, 8):
            assert math.isclose(
                maxrl_estimator_expectation(p, n),
                truncated_maxrl_gradient(p, n),
                rel_tol=0.0,
                abs_tol=1e-12,
            )


def test_truncation_gap_decreases_with_compute():
    p = 0.2
    gaps = [
        abs(ml_gradient(p) - truncated_maxrl_gradient(p, n))
        for n in (1, 2, 4, 8, 16)
    ]
    assert gaps == sorted(gaps, reverse=True)


def test_source_audit_finds_maxrl_implementation():
    audit = audit_source(SOURCE_ROOT)
    assert audit["commit"] == "7197bbb46a2ecd866da52f6b401ff20a34fe9390"
    assert audit["core_algos"]["has_maxrl_branch"]
    assert audit["core_algos"]["has_mean_reward_denominator"]
    assert audit["qwen3_script"]["sets_maxrl_estimator"]
    assert audit["readme"]["states_gpu_installation"]


def test_evidence_marks_large_scale_claims_unavailable(tmp_path):
    evidence = build_evidence(SOURCE_ROOT, tmp_path)
    statuses = {claim["id"]: claim["status"] for claim in evidence["claims"]}
    assert statuses["pareto_dominance"] == "unavailable"
    assert statuses["twenty_x_scaling"] == "unavailable"
    assert len(bernoulli_checks()) == 20
    assert (tmp_path / "evidence" / "results.json").is_file()
    assert (tmp_path / "pages" / "report.md").is_file()
