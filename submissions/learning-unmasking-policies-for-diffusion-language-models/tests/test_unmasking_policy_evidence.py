from pathlib import Path

from unmasking_policies_repro.evidence import (
    ATTEMPT_ID,
    CLAIMS,
    block_schedule,
    build_evidence_bundle,
    confidence_policy,
    left_to_right_order,
    masked_mdp_step,
    repository_audit,
    write_evidence,
)


def test_claim_statuses_match_design_strategy():
    bundle = build_evidence_bundle(
        repo_files={
            "README.md": "Markov Decision Process single-block transformer confidence policy evaluation",
            "configs/experiment_configs/policy.yaml": "block_length: 32\npolicy_type: dit_confidence",
        }
    )

    assert [claim["challenge_claim_sha256"] for claim in bundle["claims"]] == [
        "333c510d8a8d69cc59827726bb86dd399983b01e7a253af8887d0f2251cda61b",
        "bf6aebbeea700b651067e91333f97aef0e4fffec15565daf27c4cf0e89b06056",
        "28fdce6dd76e8df860c9149960ceaf5f1edf8398eb3da00be9d4844911be16f5",
        "d2a26b39ada20dfa224d97c827be80a8f15fe7a06bdc9811c6a53f086fc2e607",
        "1969db18a5c2b4252edee8f22c1e947bb7d185870e1016c0f8a491141c5415ac",
    ]
    assert [claim["status"] for claim in bundle["claims"]] == [
        "verified",
        "verified",
        "inconclusive",
        "inconclusive",
        "toy",
    ]
    assert len(CLAIMS) == 5


def test_masked_mdp_step_unmasks_selected_positions_only():
    step = masked_mdp_step(
        tokens=["[MASK]", "[MASK]", "[MASK]", "[MASK]"],
        predictions=["A", "B", "C", "D"],
        action=[1, 3],
    )

    assert step == {
        "state": ["[MASK]", "[MASK]", "[MASK]", "[MASK]"],
        "action": [1, 3],
        "next_state": ["[MASK]", "B", "[MASK]", "D"],
        "unmasked": 2,
        "done": False,
    }


def test_confidence_policy_selects_top_k_with_stable_tie_break():
    assert confidence_policy([0.2, 0.9, 0.9, 0.1], budget=2) == [1, 2]
    assert confidence_policy([0.2, 0.9, 0.9, 0.1], budget=0) == []


def test_block_schedule_covers_sequence_without_overlap():
    assert block_schedule(length=10, block_length=4) == [
        [0, 1, 2, 3],
        [4, 5, 6, 7],
        [8, 9],
    ]


def test_left_to_right_expert_order_returns_mask_positions():
    assert left_to_right_order(["[MASK]", "fixed", "[MASK]", "[MASK]"]) == [0, 2, 3]


def test_repository_audit_detects_mechanism_terms_and_hashes_files():
    audit = repository_audit(
        {
            "README.md": "MDP environment policy confidence single-block transformer",
            "eval/pipeline.py": "sampling_mode bernoulli-argmax save_path eval_results",
        }
    )

    assert audit["terms_found"] == {
        "mdp": True,
        "environment": True,
        "confidence": True,
        "single_block_transformer": True,
        "evaluation": True,
    }
    assert sorted(audit["file_sha256"]) == ["README.md", "eval/pipeline.py"]


def test_write_evidence_creates_bundle_and_report(tmp_path: Path):
    bundle_path, report_path = write_evidence(
        tmp_path,
        repo_files={"README.md": "MDP environment confidence single-block transformer"},
    )

    assert bundle_path.read_text().count(ATTEMPT_ID) == 1
    assert report_path.read_text().startswith("# Learning Unmasking Policies Reproduction Evidence")
