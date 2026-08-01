import json

from interplay_repro.evidence import (
    CLAIMS,
    UPSTREAM_PINS,
    audit_dataset_splits,
    audit_official_code,
    build_evidence_bundle,
)


def test_bundle_records_immutable_pins_and_claim_bindings():
    bundle = build_evidence_bundle(
        code_artifacts={
            "utils/solution_dependency_graph.py": "class SolutionDependencyGraph: pass\n",
            "verl/dataset_context.py": "class ContextDataset: pass\nTEMPLATE = 'zootopia'\n",
            "verl/reward_fn.py": "def process_reward(trace): return 1\n",
            "scripts/eval_checkpoints.py": "pass_at_k = 128\n",
        },
        dataset_samples={
            "composition/test/op10-1k.jsonl": '{"question":"q","answer":"a","solution":"step"}\n',
            "context/crazy_zootopia/test/op10_1k.jsonl": '{"context":"zoo","question":"q","answer":"a"}\n',
        },
        raw_result_artifacts={},
    )

    assert bundle["paper_id"] == "TBaUfO9znF"
    assert bundle["snapshot_id"] == "e24ef4f585f2af51c2a85898401421e84ff4d6e8b74fb53e138c9164f1e83d57"
    assert bundle["upstream_pins"] == UPSTREAM_PINS
    assert all("main" not in pin for pin in bundle["upstream_pins"].values())
    assert [result["claim_sha256"] for result in bundle["claim_results"]] == [
        claim["challenge_claim_sha256"] for claim in CLAIMS
    ]
    json.dumps(bundle)


def test_artifact_audit_detects_required_research_components():
    audit = audit_official_code(
        {
            "utils/solution_dependency_graph.py": "class DependencyGraph:\n    pass\n",
            "verl/dataset.py": "def build_dataset(): return 'op10'\n",
            "verl/dataset_context.py": "contextual_templates = ['crazy_zootopia']\n",
            "verl/reward_fn.py": "def process_reward(answer, trace): return 'process'\n",
            "scripts/eval_checkpoints.py": "def pass_at_128(): pass\n",
        }
    )

    assert audit["dependency_graph_code"]["status"] == "present"
    assert audit["context_dataset_code"]["status"] == "present"
    assert audit["process_reward_code"]["status"] == "present"
    assert audit["evaluation_code"]["status"] == "present"


def test_dataset_split_audit_records_hashes_and_operation_depths():
    audit = audit_dataset_splits(
        {
            "composition/test/op10-1k.jsonl": '{"question":"q","answer":"a"}\n',
            "composition/heldout/op14-50k.jsonl": '{"question":"q","answer":"a"}\n',
            "context/crazy_zootopia/test/op10_1k.jsonl": '{"context":"zoo","question":"q","answer":"a"}\n',
        }
    )

    assert audit["composition_depths"] == [10, 14]
    assert audit["context_depths"] == [10]
    assert audit["sample_hashes"]["composition/test/op10-1k.jsonl"]
    assert audit["sample_record_count"] == 3


def test_numeric_training_claims_are_not_verified_without_raw_results():
    bundle = build_evidence_bundle(
        code_artifacts={
            "utils/solution_dependency_graph.py": "DependencyGraph",
            "verl/dataset_context.py": "context",
            "verl/reward_fn.py": "process_reward",
            "scripts/eval_checkpoints.py": "pass@128",
        },
        dataset_samples={
            "composition/test/op10-1k.jsonl": '{"question":"q","answer":"a"}\n',
            "context/crazy_zootopia/test/op10_1k.jsonl": '{"context":"zoo","question":"q","answer":"a"}\n',
        },
        raw_result_artifacts={},
    )

    numeric_results = bundle["claim_results"][1:]
    assert all(result["status"] in {"toy", "unavailable", "inconclusive"} for result in numeric_results)
    assert all("raw official result" in result["limitation"] for result in numeric_results)
