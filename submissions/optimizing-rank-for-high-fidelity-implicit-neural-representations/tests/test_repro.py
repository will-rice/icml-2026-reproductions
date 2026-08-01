"""Test suite for Optimizing Rank for High-Fidelity INRs reproduction."""

import json
from pathlib import Path

import torch

from optimizing_rank_inr_repro.benchmarks import (
    build_radon_matrix,
    generate_phantom,
    make_model_pair,
    run_claim1_stable_rank_degradation_test,
    run_claim2_image_overfitting_test,
    run_claim3_sparse_ct_test,
    run_claim4_multidomain_extension_test,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PAGE_NAMES = [
    "00-summary.md",
    "01-claim-1-stable-rank.md",
    "02-claim-2-image-overfitting.md",
    "03-claim-3-sparse-view-ct.md",
    "04-claim-4-multidomain.md",
]


def test_model_pair_starts_identical():
    """Adam and Muon must begin from the same weights for the comparison to be fair."""
    model_adam, model_muon, _, _ = make_model_pair("vanilla_mlp", seed=42)
    for p_a, p_m in zip(model_adam.parameters(), model_muon.parameters()):
        assert torch.equal(p_a, p_m)


def test_radon_operator_is_underdetermined_and_nontrivial():
    radon = build_radon_matrix(size=32, n_angles=8, n_bins=32)
    assert radon.shape == (256, 1024)
    assert radon.shape[0] < radon.shape[1], "sparse-view CT must be under-determined"
    sinogram = radon @ generate_phantom(size=32).reshape(-1, 1)
    assert sinogram.abs().sum() > 0


def test_claim1_stable_rank_degradation():
    res = run_claim1_stable_rank_degradation_test()
    assert res["status"] == "verified"
    assert res["rank_preserved_by_muon"] is True
    assert res["final_muon_stable_rank"] > res["final_adam_stable_rank"]


def test_claim2_image_overfitting():
    res = run_claim2_image_overfitting_test()
    assert set(res["architectures"]) == {"siren", "vanilla_mlp"}
    assert res["status"] == "verified"
    assert res["all_architectures_improved"] is True


def test_claim3_sparse_ct_reports_measured_outcome():
    """Claim 3 is reported as measured; the harness must not force a pass."""
    res = run_claim3_sparse_ct_test()
    assert res["n_measurements"] < res["n_pixels"]
    for arch, vals in res["architectures"].items():
        expected = vals["recon_psnr_muon"] >= vals["recon_psnr_adam"]
        assert vals["improved"] is expected, f"{arch} improved flag must match the numbers"
    expected_status = "verified" if res["all_architectures_improved"] else "unverified"
    assert res["status"] == expected_status


def test_claim4_multidomain_reports_measured_outcome():
    res = run_claim4_multidomain_extension_test()
    assert set(res["domains"]) == {
        "natural_image",
        "medical_phantom",
        "audio_1d",
        "super_resolution",
    }
    for domain, vals in res["domains"].items():
        expected = vals["psnr_muon"] >= vals["psnr_adam"]
        assert vals["improved"] is expected, f"{domain} improved flag must match the numbers"


def test_determinism():
    res1 = run_claim1_stable_rank_degradation_test(steps=50, seed=123)
    res2 = run_claim1_stable_rank_degradation_test(steps=50, seed=123)
    assert res1 == res2


def test_evidence_file_matches_claims():
    ev_path = PROJECT_ROOT / "evidence" / "evidence.json"
    assert ev_path.exists(), "evidence.json must exist"
    with open(ev_path) as f:
        data = json.load(f)
    assert data["paper_id"] == "2azIa9tfl3"
    assert len(data["claims"]) == 4
    for claim in data["claims"]:
        assert claim["status"] in {
            "reproduced",
            "partially_reproduced",
            "not_reproduced",
            "unreplicated",
        }
        assert claim["statement"].strip()


def test_all_pages_present_and_nonempty():
    pages = PROJECT_ROOT / "pages"
    assert sorted(p.name for p in pages.glob("*.md")) == PAGE_NAMES
    for name in PAGE_NAMES:
        assert (pages / name).read_text(encoding="utf-8").strip()


def test_pages_do_not_overclaim_unreproduced_results():
    """A claim the numbers refute must not be presented as reproduced."""
    with open(PROJECT_ROOT / "evidence" / "evidence.json") as f:
        data = json.load(f)
    ct_status = next(c["status"] for c in data["claims"] if c["claim_id"] == "claim_3")
    ct_page = (PROJECT_ROOT / "pages" / "03-claim-3-sparse-view-ct.md").read_text()
    assert f"`{ct_status}`" in ct_page
    if ct_status != "reproduced":
        assert "does not reproduce" in ct_page
