from __future__ import annotations

import math

from d3llm_repro.evidence import (
    compute_aup,
    generate_bundle,
    rank_aup_scores,
    select_trajectory_step,
    simulate_entropy_multiblock_decoding,
)


def test_aup_matches_hand_computed_trapezoid() -> None:
    score = compute_aup(rho=[1.0, 3.0], accuracy=[80.0, 78.0], y_max=80.0)
    weight = math.exp(-3.0 * (1.0 - 78.0 / 80.0))
    expected = 1.0 * 80.0 + 0.5 * 2.0 * (78.0 * weight + 80.0)
    assert score == expected


def test_released_aup_data_gives_d3llm_top_rankings() -> None:
    llada = rank_aup_scores("llada")
    dream = rank_aup_scores("dream")

    assert len(llada) == 5
    assert len(dream) == 5
    assert {row["best_method"] for row in llada} == {"d3LLM-LLaDA"}
    assert {row["best_method"] for row in dream} == {"d3LLM-Dream"}


def test_trajectory_selection_uses_mask_ratio_and_block_offset() -> None:
    trajectories = [
        ["<mask>", "<mask>", "<mask>", "<mask>"],
        ["A", "<mask>", "<mask>", "<mask>"],
        ["A", "B", "<mask>", "<mask>"],
        ["A", "B", "C", "<mask>"],
        ["A", "B", "C", "D"],
    ]

    assert select_trajectory_step(trajectories, mask_ratio=0.50, block_start=0, block_end=4) == trajectories[2]
    assert select_trajectory_step(trajectories, mask_ratio=0.25, block_start=1, block_end=3) == trajectories[2]


def test_entropy_multiblock_decoding_decodes_low_entropy_tokens_and_refreshes_cache() -> None:
    result = simulate_entropy_multiblock_decoding(
        blocks=[
            [("A", 0.91), ("B", 0.55)],
            [("C", 0.88), ("D", 0.52)],
        ],
        entropy_threshold=0.70,
        refresh_every=2,
    )

    assert result.decoded_tokens == ["A", "C"]
    assert result.parallel_blocks == 2
    assert result.cache_refresh_steps == [2]


def test_generated_bundle_records_claim_statuses_and_limitations(tmp_path) -> None:
    bundle = generate_bundle(output_dir=tmp_path)

    statuses = {claim["claim_id"]: claim["status"] for claim in bundle["claims"]}
    assert statuses["claim_1_aup_definition"] == "verified"
    assert statuses["claim_2_pseudo_trajectory"] == "toy_verified"
    assert statuses["claim_3_entropy_multiblock"] == "toy_verified"
    assert statuses["claim_4_llada_aup_ranking"] == "verified"
    assert statuses["claim_5_dream_aup_ranking"] == "verified"
    assert statuses["claim_6_throughput_speedups"] == "artifact_consistency"
    assert "No fresh GPU throughput benchmark" in bundle["limitations"][0]
    assert (tmp_path / "bundle.json").exists()
