from pathlib import Path

from threadweaver_repro.evidence import (
    EXPECTED_UPSTREAM_REVISION,
    audit_upstream,
    build_evidence,
    critical_path_latency,
    parse_parallel_trajectory,
    p_grpo_advantages,
    trie_attention,
)


UPSTREAM = Path(__file__).resolve().parents[3] / "scratch" / "threadweaver-upstream"


def test_upstream_audit_pins_revision_and_core_artifacts():
    audit = audit_upstream(UPSTREAM)

    assert audit["git_revision"] == EXPECTED_UPSTREAM_REVISION
    assert audit["required_paths"]["data_generation/src/generate_trajectories.py"]
    assert audit["required_paths"]["threadweaver_sft/src/prefix_tree_utils_v1.py"]
    assert audit["required_paths"]["threadweaver_rl/train_par.slurm"]
    assert audit["markers"]["two_stage_generator"]
    assert audit["markers"]["trie_training"]
    assert audit["markers"]["p_grpo_mean_centered"]
    assert audit["markers"]["requires_8x80g_gpu"]


def test_parallel_trajectory_parser_binds_outlines_to_threads():
    trajectory = """
<think>
<Parallel>
<Outlines>
<Outline>1: derive identity</Outline>
<Outline>2: numeric check</Outline>
</Outlines>
<Thread>1: Use product-to-sum.</Thread>
<Thread>2: Substitute decimals.</Thread>
</Parallel>
</think>
"""

    parsed = parse_parallel_trajectory(trajectory)

    assert parsed.outlines == {"1": "derive identity", "2": "numeric check"}
    assert parsed.threads == {
        "1": "Use product-to-sum.",
        "2": "Substitute decimals.",
    }
    assert parsed.outline_thread_ids_match
    assert parsed.threads_are_independent


def test_trie_attention_blocks_sibling_leakage_but_allows_ancestors():
    result = trie_attention(
        [
            ["prompt", "<Parallel>", "<Thread>1", "alpha", "</Thread>"],
            ["prompt", "<Parallel>", "<Thread>2", "beta", "</Thread>"],
        ]
    )

    thread1_alpha = result.positions[("alpha", 0)]
    thread2_beta = result.positions[("beta", 0)]
    shared_parallel = result.positions[("<Parallel>", 0)]

    assert result.allowed[thread1_alpha][shared_parallel]
    assert result.allowed[thread2_beta][shared_parallel]
    assert not result.allowed[thread1_alpha][thread2_beta]
    assert not result.allowed[thread2_beta][thread1_alpha]


def test_critical_path_latency_uses_longest_parallel_thread():
    latency = critical_path_latency(sequential_tokens=8, thread_tokens=[5, 13, 7])

    assert latency["sequential_token_latency"] == 33
    assert latency["critical_path_token_latency"] == 21
    assert latency["token_latency_speedup"] == 33 / 21


def test_p_grpo_mean_centering_preserves_small_acceleration_scale():
    result = p_grpo_advantages(
        correctness=[1.0, 1.0, 1.0],
        acceleration=[0.0, 0.1, 0.2],
        acceleration_weight=0.1,
    )

    assert result["rewards"] == [1.0, 1.01, 1.02]
    assert result["p_grpo_advantages"] == [-0.01, 0.0, 0.01]
    assert result["standard_grpo_advantages"][2] > 1.0


def test_evidence_marks_hardware_only_metrics_unreplicated():
    evidence = build_evidence(UPSTREAM)
    statuses = {
        claim["claim_id"]: claim["status"]
        for claim in evidence["claims"]
    }

    assert statuses["two_stage_generator"] == "verified"
    assert statuses["trie_rollout_design"] == "toy"
    assert statuses["parallelization_aware_rl"] == "toy"
    assert statuses["six_benchmark_accuracy"] == "unreplicated"
    assert statuses["reported_token_latency_speedup"] == "unreplicated"
    assert evidence["costs"]["metered_api_usd"] == 0.0
