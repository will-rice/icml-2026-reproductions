import json
from pathlib import Path

from generate_evidence import (
    build_bundle,
    dataset_totals,
    model_setup_terms,
    perturbation_terms,
)


FIXTURES = Path(__file__).parent / "fixtures"


def test_collection_cardinality_and_setup_terms():
    collection = json.loads((FIXTURES / "model_collection.json").read_text())
    card = (FIXTURES / "model_card.md").read_text()

    assert len(collection["items"]) == 14
    assert model_setup_terms(card) == {
        "architecture": True,
        "training_data": True,
        "training_budget": True,
        "initialization": True,
        "tokenizer_only_difference": True,
    }


def test_dataset_totals_capture_exact_mismatch():
    sizes = json.loads((FIXTURES / "dataset_sizes.json").read_text())
    totals = dataset_totals(sizes)

    assert totals["named_rows"] == 4883
    assert totals["with_general_rows"] == 4951
    assert totals["claimed_rows"] == 4941
    assert totals["matches_claimed_rows"] is False


def test_perturbation_terms_are_detected_case_insensitively():
    card = (FIXTURES / "model_card.md").read_text()

    assert perturbation_terms(card) == {
        "input-medium": True,
        "diacritic": True,
        "orthographic": True,
        "morphology": True,
        "noise": True,
        "latex": True,
        "stem": True,
        "math": True,
    }


def test_bundle_is_conservative_and_deterministic(tmp_path):
    collection = json.loads((FIXTURES / "model_collection.json").read_text())
    sizes = json.loads((FIXTURES / "dataset_sizes.json").read_text())
    card = (FIXTURES / "model_card.md").read_text()

    bundle = build_bundle(collection, sizes, {"toksuite/tiktoken-gpt-4o": card})
    output = tmp_path / "bundle.json"
    output.write_text(json.dumps(bundle, indent=2, sort_keys=True) + "\n")

    reread = json.loads(output.read_text())
    assert reread == bundle
    assert [claim["status"] for claim in bundle["claims"]] == [
        "verified",
        "falsified",
        "verified",
    ]
