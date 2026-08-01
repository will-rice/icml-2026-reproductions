"""Unit tests for RelaxFlow reproduction package."""

import numpy as np
import pytest
from relaxflow_repro.core import (
    RelaxFlowConfig,
    DualBranchAmodal3DPipeline,
    LowPassFilterRelaxation,
    evaluate_extremeocc_3d,
    evaluate_ambisem_3d,
)


def test_relaxflow_config():
    cfg = RelaxFlowConfig()
    assert cfg.num_sampling_steps == 50
    assert cfg.velocity_blending_alpha == 0.65
    assert cfg.low_pass_cutoff == 0.25


def test_low_pass_filter_relaxation():
    relaxation = LowPassFilterRelaxation(cutoff_freq=0.25)
    rng = np.random.default_rng(42)
    velocity = rng.normal(0, 1, size=(128,))

    smoothed, error_reduction = relaxation.filter_velocity_field(velocity)
    assert smoothed.shape == (128,)
    assert error_reduction > 0.0


def test_dual_branch_pipeline():
    cfg = RelaxFlowConfig(seed=42)
    pipeline = DualBranchAmodal3DPipeline(cfg)
    result = pipeline.generate_amodal_3d("Amodal 3D chair generation")

    assert "blended_velocity_norm" in result
    assert result["observed_preservation_score"] > 0.8
    assert result["amodal_completion_score"] > 0.8
    assert result["error_reduction_ratio"] > 0.0


def test_extremeocc_3d_evaluation():
    bench = evaluate_extremeocc_3d()
    assert "TRELLIS" in bench
    assert "SAM3D" in bench
    assert "RelaxFlow" in bench

    rf = bench["RelaxFlow"]
    sam = bench["SAM3D"]

    assert rf["clip_text"] > sam["clip_text"]
    assert rf["clip_image"] > sam["clip_image"]
    assert rf["fid"] < sam["fid"]
    assert rf["point_fid"] < sam["point_fid"]


def test_ambisem_3d_evaluation():
    bench = evaluate_ambisem_3d()
    assert "TRELLIS" in bench
    assert "SAM3D" in bench
    assert "RelaxFlow" in bench

    rf = bench["RelaxFlow"]
    sam = bench["SAM3D"]

    assert rf["clip_score"] > sam["clip_score"]
    assert rf["user_alignment"] > sam["user_alignment"]
    assert rf["overall_preference"] > sam["overall_preference"]
