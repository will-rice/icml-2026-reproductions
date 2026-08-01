"""Tests asserting Motive evidence is computed, honest, and stable."""

import json
import pathlib

import motive.attribution as attribution
from motive.attribution import (
    compute_motion_mask,
    experiment_dynamics_vs_magnitude,
    experiment_frame_length_bias,
    experiment_motion_mask_localization,
    make_moving_square_video,
)

PROJECT = pathlib.Path(__file__).parent.parent


def test_motion_mask_shape():
    frames = make_moving_square_video(4)
    mask = compute_motion_mask(frames)
    assert mask.shape == (1, 3, 4, 4)
    assert (mask >= 0.0).all()


def test_mask_localizes_true_motion():
    result = experiment_motion_mask_localization()
    assert result["mean_mask_in_moving_patches"] > 0.3
    assert result["mean_mask_in_static_patches"] < 0.01
    assert result["fraction_of_mask_weight_on_true_motion"] > 0.95
    assert result["masked_attribution_norm"] < result["unmasked_attribution_norm"]


def test_frame_length_bias_measured_and_fixed():
    result = experiment_frame_length_bias()
    assert result["raw_growth_ratio_longest_vs_shortest"] > 1.5
    assert 0.8 < result["normalized_growth_ratio_longest_vs_shortest"] < 1.25
    assert result["raw_ranking"] == ["weak_long", "strong_short"]
    assert result["normalized_ranking"] == ["strong_short", "weak_long"]


def test_influence_tracks_dynamics_not_magnitude():
    result = experiment_dynamics_vs_magnitude()
    assert result["jitter_has_higher_raw_motion"] is True
    assert result["coherent_influence_cosine"] > result["jitter_influence_cosine"]


def test_fabricated_evaluators_are_gone():
    for name in ("evaluate_vbench_motion", "evaluate_human_preference"):
        assert not hasattr(attribution, name)


def test_evidence_matches_recomputation_and_is_honest():
    with open(PROJECT / "evidence_summary.json") as f:
        evidence = json.load(f)
    assert evidence["paper_id"] == "zAl9heLw4q"
    assert evidence["all_target_claims_verified"] is False
    claims = evidence["target_claims"]
    assert len(claims) == 9

    statuses = [c["status"] for c in claims]
    assert statuses == [
        "partially_reproduced",
        "partially_reproduced",
        "unreplicated",
        "unreplicated",
        "partially_reproduced",
        "unreplicated",
        "unreplicated",
        "partially_reproduced",
        "partially_reproduced",
    ]
    for claim in claims:
        if claim["status"] == "unreplicated":
            assert "evidence_details" not in claim

    assert claims[0]["evidence_details"] == experiment_motion_mask_localization()
    assert claims[8]["evidence_details"] == experiment_dynamics_vs_magnitude()


def test_report_page_carries_computed_numbers():
    report = (PROJECT / "pages" / "report.md").read_text()
    localization = experiment_motion_mask_localization()
    assert str(localization["mean_mask_in_moving_patches"]) in report
    assert "unreplicated" in report.lower()
    assert "fabricat" in report.lower()
