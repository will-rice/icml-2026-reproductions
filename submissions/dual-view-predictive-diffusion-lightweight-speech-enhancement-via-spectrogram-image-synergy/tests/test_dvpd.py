"""Tests asserting DVPD evidence is computed, honest, and stable."""

import json
import sys
from pathlib import Path

src_dir = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_dir))

import torch

import dvpd.model as model_module
from dvpd.model import (
    DVPDModel,
    FANCEncoder,
    FrequencyAwareInteraction,
    experiment_architecture_and_macs,
    experiment_fanc_band_allocation,
    experiment_interaction_coupling,
    experiment_toy_ablation,
)

PROJECT = Path(__file__).parent.parent


def test_fanc_encoder_shapes():
    encoder = FANCEncoder(in_channels=1, embed_dim=64)
    x = torch.randn(2, 1, 257, 50)
    assert encoder(x).shape == (2, 64, 257, 50)


def test_frequency_aware_interaction_shapes():
    interaction = FrequencyAwareInteraction(channels=64)
    ac = torch.randn(2, 64, 64, 50)
    vis = torch.randn(2, 64, 64, 50)
    out_ac, out_vis = interaction(ac, vis)
    assert out_ac.shape == ac.shape
    assert out_vis.shape == vis.shape


def test_dvpd_model_forward():
    model = DVPDModel(in_channels=1, embed_dim=64)
    x = torch.randn(1, 1, 257, 50)
    assert model(x).shape == x.shape


def test_macs_are_measured_not_asserted():
    architecture = experiment_architecture_and_macs()
    assert architecture["toy_model_conv_macs_per_forward"] > 0
    assert architecture["comparable_to_paper_models"] is False
    small = DVPDModel(in_channels=1, embed_dim=16)
    big = DVPDModel(in_channels=1, embed_dim=64)
    from dvpd.model import count_conv_macs

    assert count_conv_macs(small, (1, 1, 257, 50)) < count_conv_macs(big, (1, 1, 257, 50))


def test_interaction_coupling_is_real():
    coupling = experiment_interaction_coupling()
    assert coupling["acoustic_response_to_visual_perturbation"] > 0
    assert coupling["visual_response_to_acoustic_perturbation"] > 0


def test_fanc_band_allocation_is_nonuniform():
    fanc = experiment_fanc_band_allocation()
    assert fanc["low_band_representation_rows"] == fanc["low_band_input_rows"]
    assert fanc["high_band_representation_rows"] < fanc["high_band_input_rows"]
    assert fanc["macs_per_input_row_low_band"] > fanc["macs_per_input_row_high_band"]


def test_ablation_training_denoises():
    ablation = experiment_toy_ablation()
    variants = ablation["heldout_denoised_mse_by_variant"]
    for mse in variants.values():
        assert mse < ablation["heldout_noisy_input_mse"]
    assert variants["no_lisa"] > variants["full"]


def test_fabricated_evaluators_are_gone():
    for name in ("compute_model_efficiency", "run_dvpd_verification"):
        assert not hasattr(model_module, name)


def test_evidence_matches_recomputation_and_is_honest():
    with open(PROJECT / "evidence" / "evidence_summary.json") as f:
        evidence = json.load(f)
    assert evidence["paper_id"] == "3qX5RS8kpJ"
    assert evidence["all_target_claims_verified"] is False
    claims = evidence["claims"]
    assert [c["status"] for c in claims] == [
        "partially_reproduced",
        "partially_reproduced",
        "unreplicated",
        "unreplicated",
        "unreplicated",
        "partially_reproduced",
    ]
    for claim in claims:
        if claim["status"] == "unreplicated":
            assert "evidence" not in claim

    assert claims[1]["evidence"] == experiment_fanc_band_allocation()
    assert claims[5]["evidence"] == experiment_toy_ablation()


def test_report_page_carries_computed_numbers():
    report = (PROJECT / "pages" / "report.md").read_text()
    fanc = experiment_fanc_band_allocation()
    assert str(fanc["high_band_representation_rows"]) in report
    assert "unreplicated" in report.lower()
    assert "fabricated" in report.lower()
    assert "not supported" in report


def test_readme_metadata():
    content = (PROJECT / "README.md").read_text()
    assert "icml2026-repro" in content
    assert "paper-3qX5RS8kpJ" in content
    assert "sdk: streamlit" in content
