"""Unit tests for DVPD reproduction implementation and metadata."""

import json
import sys
from pathlib import Path

# Add src directory to sys.path
src_dir = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_dir))

import pytest
import torch
from dvpd.model import (
    DVPDModel,
    FANCEncoder,
    FrequencyAwareInteraction,
    compute_model_efficiency,
    run_dvpd_verification,
)


def test_fanc_encoder():
    encoder = FANCEncoder(in_channels=1, embed_dim=64)
    x = torch.randn(2, 1, 257, 50)
    out = encoder(x)
    assert out.shape == (2, 64, 257, 50)


def test_frequency_aware_interaction():
    interaction = FrequencyAwareInteraction(channels=64)
    ac = torch.randn(2, 64, 64, 50)
    vis = torch.randn(2, 64, 64, 50)
    out_ac, out_vis = interaction(ac, vis)
    assert out_ac.shape == ac.shape
    assert out_vis.shape == vis.shape


def test_dvpd_model_forward():
    model = DVPDModel(in_channels=1, embed_dim=64)
    x = torch.randn(1, 1, 257, 50)
    out = model(x)
    assert out.shape == x.shape


def test_efficiency_claims():
    eff = compute_model_efficiency()
    assert eff["param_ratio_vs_pguse"] <= 0.35
    assert eff["macs_ratio_vs_pguse"] <= 0.40


def test_verification_suite():
    res = run_dvpd_verification()
    assert res["model_forward_success"] is True


def test_readme_metadata():
    project_dir = Path(__file__).parent.parent
    readme_path = project_dir / "README.md"
    assert readme_path.exists()
    content = readme_path.read_text()
    assert "icml2026-repro" in content
    assert "paper-3qX5RS8kpJ" in content
    assert "sdk: streamlit" in content
