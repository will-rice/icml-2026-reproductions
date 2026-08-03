from pathlib import Path
import json

from robomme_repro.evidence import TARGET_CLAIMS, UPSTREAM_PINS, build_bundle


ARTIFACT_ROOT = Path("/tmp/icml-robomme-repo")
PROJECT = Path(__file__).resolve().parents[1]


def test_upstream_pins_are_immutable():
    assert UPSTREAM_PINS["benchmark_repo"].endswith(
        "@03a8faf57cfbd334dfea1bb7e60079a888d70453"
    )
    assert UPSTREAM_PINS["policy_repo"].endswith(
        "@ecf086c3be7c2223167d9bb2f6ef1f0a6e24353b"
    )
    assert UPSTREAM_PINS["preprocessed_data"].endswith(
        "@ddf0baf55b633cc6657dcd53ac0e089a273de612"
    )
    assert UPSTREAM_PINS["mme_vla_suite"].endswith(
        "@5db4d53ddb98c7f80cab08792dd53d985d712ab1"
    )


def test_target_claims_bind_current_challenge_hashes():
    hashes = {claim["challenge_claim_sha256"] for claim in TARGET_CLAIMS}
    assert hashes == {
        "36b8db60ef193ce6da36b30be54d33856b268a93ea4582487650146b7ff27cec",
        "44a693f9fed0a7d249d8c4caa696205797c7ea344d6cc73560267c8b91d288aa",
        "9c421693d19a62760e4fb953af3c3bb3f43ed1e9f1ad8b3afbdcf2eee67e6e60",
        "dd2f1bba318425211ac424545696fa9a23b2d2fc5eba76cdcb7123ab03bd0958",
    }


def test_bundle_verifies_sixteen_tasks_and_memory_categories():
    bundle = build_bundle(ARTIFACT_ROOT)
    taxonomy = bundle["observations"]["task_taxonomy"]
    assert taxonomy["task_count"] == 16
    assert taxonomy["categories"] == {
        "Temporal memory": ["BinFill", "PickXtimes", "SwingXtimes", "StopCube"],
        "Spatial memory": [
            "VideoUnmask",
            "VideoUnmaskSwap",
            "ButtonUnmask",
            "ButtonUnmaskSwap",
        ],
        "Object memory": [
            "PickHighlight",
            "VideoRepick",
            "VideoPlaceButton",
            "VideoPlaceOrder",
        ],
        "Procedural memory": ["MoveCube", "InsertPeg", "PatternLock", "RouteStick"],
    }
    assert bundle["claim_results"]["task_taxonomy"]["status"] == "verified"


def test_bundle_counts_split_records_for_all_tasks():
    bundle = build_bundle(ARTIFACT_ROOT)
    assert bundle["observations"]["split_records"] == {
        "train": {"tasks": 16, "record_count": 1600},
        "val": {"tasks": 16, "record_count": 800},
        "test": {"tasks": 16, "record_count": 800},
    }


def test_bundle_verifies_training_timestep_count():
    bundle = build_bundle(ARTIFACT_ROOT)
    stats = bundle["observations"]["training_timestep_stats"]
    assert stats["total_samples"] == 768897
    assert stats["rounded_claim_value"] == "770K"
    assert bundle["claim_results"]["training_timesteps"]["status"] == "verified"


def test_bundle_verifies_fourteen_mme_vla_variants():
    bundle = build_bundle(ARTIFACT_ROOT)
    variants = bundle["observations"]["mme_vla_variants"]
    assert variants["variant_count"] == 14
    assert variants["history_config_count"] == 14
    assert set(variants["representation_types"]) == {
        "symbolic",
        "perceptual",
        "recurrent",
    }
    assert set(variants["integration_types"]) == {"context", "modulation", "expert"}


def test_generated_bundle_has_required_claim_statuses():
    bundle = json.loads((PROJECT / "evidence" / "bundle.json").read_text())
    assert {result["status"] for result in bundle["claim_results"].values()} <= {
        "verified",
        "toy",
        "inconclusive",
        "unavailable",
    }
    assert bundle["paper_id"] == "8m30ogkPk2"


def test_space_readme_contains_required_metadata():
    readme = (PROJECT / "README.md").read_text()
    assert 'emoji: "🤖"' in readme
    assert "paper-8m30ogkPk2" in readme
    assert "icml2026-repro" in readme
    assert "sdk: gradio" in readme
