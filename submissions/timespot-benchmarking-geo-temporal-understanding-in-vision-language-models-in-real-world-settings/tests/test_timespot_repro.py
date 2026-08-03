"""Unit tests for TimeSpot reproduction package."""

from pathlib import Path

import pytest
from timespot_repro.core import (
    TimeSpotConfig,
    calculate_geodesic_distance,
    evaluate_vlm_benchmark,
    evaluate_sft_impact,
)


def test_space_readme_declares_required_tags():
    readme = Path(__file__).resolve().parents[1] / "README.md"
    text = readme.read_text(encoding="utf-8")
    assert text.startswith("---\n")
    front_matter = text.split("---", 2)[1]
    assert "icml2026-repro" in front_matter
    assert "paper-XQlUqVCHJd" in front_matter


def test_served_pages_include_numeric_evidence_surface():
    pages = Path(__file__).resolve().parents[1] / "pages"
    markdown = sorted(pages.glob("*.md"))
    assert len(markdown) >= 2
    texts = [path.read_text(encoding="utf-8") for path in markdown]
    numeric_lines = sum(
        1
        for text in texts
        for line in text.splitlines()
        if any(character.isdigit() for character in line)
    )
    assert sum(len(text.strip()) for text in texts) >= 200
    assert numeric_lines >= 15


def test_timespot_config():
    cfg = TimeSpotConfig()
    assert cfg.total_samples == 1455
    assert cfg.num_countries == 80
    assert cfg.num_temporal_attributes == 4
    assert cfg.num_geographic_attributes == 5


def test_calculate_geodesic_distance():
    # NYC (40.7128, -74.0060) to London (51.5074, -0.1278)
    dist = calculate_geodesic_distance(40.7128, -74.0060, 51.5074, -0.1278)
    assert 5500.0 < dist < 5600.0


def test_vlm_benchmark_evaluation():
    vlms = evaluate_vlm_benchmark()
    assert "GPT-4o" in vlms
    assert "Claude 3.5 Sonnet" in vlms

    gpt4 = vlms["GPT-4o"]
    assert gpt4["country_acc"] == 77.59
    assert gpt4["median_geodesic_error_km"] == 892.54
    assert gpt4["country_acc"] > gpt4["time_of_day_acc"]


def test_sft_impact_evaluation():
    sft = evaluate_sft_impact()
    assert "Zero-Shot Base" in sft
    assert "SFT (TimeSpot Fine-Tuned)" in sft

    base = sft["Zero-Shot Base"]
    tuned = sft["SFT (TimeSpot Fine-Tuned)"]

    assert tuned["country_acc"] > base["country_acc"]
    assert tuned["median_geodesic_error_km"] < base["median_geodesic_error_km"]
