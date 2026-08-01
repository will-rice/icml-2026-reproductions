from __future__ import annotations

from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from top_w_repro.evidence import build_bundle
from top_w_repro.pages import build_pages


@pytest.fixture(scope="module")
def bundle():
    return build_bundle()


def test_all_audits_pass(bundle):
    for name, audit in bundle["audits"].items():
        assert audit["passed"], name


def test_claim_statuses_are_honest(bundle):
    results = bundle["claim_results"]
    assert results["claim_1"]["status"] == "verified"
    assert results["claim_2"]["status"] == "verified"
    assert results["claim_3"]["status"] == "unreplicated"
    assert "Not reproduced" in results["claim_3"]["evidence"]
    assert results["claim_4"]["status"] == "unreplicated"
    assert "Not reproduced" in results["claim_4"]["evidence"]
    assert results["claim_5"]["status"] == "unreplicated"
    assert "Not reproduced" in results["claim_5"]["evidence"]
    assert bundle["estimated_api_cost_usd"] == 0.0


def test_bundle_pins_upstream_sources(bundle):
    assert bundle["upstream_revision"].startswith("arxiv:2602.10346v2")
    assert (
        bundle["upstream"]["revision"]
        == "5949bfae5e6a81bc279c65923f1adc1c9f2e2059"
    )
    assert "logit_processor_w1.py" in bundle["upstream"]["files"]


def test_pages_surface_the_audit_numbers(bundle):
    pages = build_pages(bundle)
    assert set(pages) == {
        "00-summary.md",
        "01-claim-1-wasserstein-objective.md",
        "02-claim-2-exact-subset-update.md",
        "03-claim-3-gsm8k-not-reproduced.md",
        "04-claim-4-gpqa-not-reproduced.md",
        "05-claim-5-openended-not-reproduced.md",
        "06-methods-and-provenance.md",
    }
    prefix = bundle["audits"]["prefix_vs_bruteforce"]
    claim_2 = pages["02-claim-2-exact-subset-update.md"]
    assert (
        f"{prefix['optimal_value_matches']}/{prefix['trials']}" in claim_2
    )
    assert str(prefix["subsets_enumerated_per_trial"]) in claim_2

    controls = bundle["audits"]["geometry_mechanism"]
    claim_1 = pages["01-claim-1-wasserstein-objective.md"]
    assert f"{controls['potential_max_error']:.1e}" in claim_1
    assert f"{controls['mean_shuffle_jaccard']:.3f}" in claim_1
    assert (
        f"{controls['uniform_metric_prefix_matches']}/{controls['trials']}"
        in claim_1
    )

    relaxation = bundle["audits"]["theorem_relaxation"]
    assert (
        f"{relaxation['prefix_suboptimal_instances']}/{relaxation['trials']}"
        in pages["02-claim-2-exact-subset-update.md"]
    )

    claim_3 = pages["03-claim-3-gsm8k-not-reproduced.md"]
    assert "unreplicated" in claim_3
    assert "NOT" in claim_3

    claim_4 = pages["04-claim-4-gpqa-not-reproduced.md"]
    assert "unreplicated" in claim_4
    assert "run_gpqa.sh" in claim_4

    claim_5 = pages["05-claim-5-openended-not-reproduced.md"]
    assert "unreplicated" in claim_5
    assert "alpaca_generate_w.py" in claim_5

    methods = pages["06-methods-and-provenance.md"]
    for sha in bundle["upstream"]["files"].values():
        assert sha in methods


def test_generated_artifacts_match_current_code():
    """The committed bundle and pages must be regenerable from source."""
    bundle_path = ROOT / "evidence" / "bundle.json"
    pages_dir = ROOT / "pages"
    assert bundle_path.exists()
    assert (pages_dir / "00-summary.md").exists()
