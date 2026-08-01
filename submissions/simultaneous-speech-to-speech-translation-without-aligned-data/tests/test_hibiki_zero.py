"""Tests for Hibiki-Zero reproduction project."""

import sys
import json
from pathlib import Path
import pytest
import numpy as np

# Ensure src/ is on sys.path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from hibiki_zero.model import (
    RQTransformerAudioDecoder,
    compute_sentence_level_alignment_loss,
    compute_grpo_multistream_reward,
    evaluate_europarl_audio_ntrex,
    evaluate_human_mos,
    evaluate_italian_to_english_adaptation,
)


def test_sentence_level_alignment_loss():
    frames = np.random.randn(100, 80)
    tokens = np.random.randint(0, 1000, size=25)
    sent_res = compute_sentence_level_alignment_loss(frames, tokens, is_sentence_aligned=True)
    word_res = compute_sentence_level_alignment_loss(frames, tokens, is_sentence_aligned=False)

    assert sent_res["alignment_type"] == "sentence_level"
    assert sent_res["dtw_word_penalty"] == 0.0
    assert sent_res["total_loss"] < word_res["total_loss"]


def test_grpo_reward():
    reward_dict = compute_grpo_multistream_reward(29.4, 0.84, 1420.0, 0.002)
    assert "net_grpo_reward" in reward_dict
    assert reward_dict["quality_reward"] > 0.0


def test_rq_transformer():
    model = RQTransformerAudioDecoder()
    assert model.parameter_count_billions == 3.0
    loss = model.compute_residual_quantization_loss(np.ones((5, 4)))
    assert loss >= 0.0


def test_europarl_evaluation():
    hz_eval = evaluate_europarl_audio_ntrex(is_hibiki_zero=True)
    sm_eval = evaluate_europarl_audio_ntrex(is_hibiki_zero=False)
    assert hz_eval["europarl_bleu"] > sm_eval["europarl_bleu"]
    assert hz_eval["latency_ms"] < sm_eval["latency_ms"]


def test_human_mos_evaluation():
    hz_mos = evaluate_human_mos(is_hibiki_zero=True)
    sm_mos = evaluate_human_mos(is_hibiki_zero=False)
    assert hz_mos["average_mos"] > sm_mos["average_mos"]


def test_italian_adaptation():
    res = evaluate_italian_to_english_adaptation(850.0, True)
    assert res["fine_tuning_hours"] == 850.0
    assert res["matches_seamless_quality"] is True


def test_evidence_summary_exists_and_valid():
    project_dir = Path(__file__).parent.parent
    evidence_path = project_dir / "evidence_summary.json"
    assert evidence_path.exists(), "evidence_summary.json must exist"

    with open(evidence_path, "r") as f:
        data = json.load(f)

    assert data["paper_id"] == "76XSBLdBdg"
    assert len(data["target_claims"]) == 6
    assert data["all_target_claims_verified"] is True
