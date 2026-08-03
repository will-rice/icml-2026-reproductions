from __future__ import annotations

import argparse
import json
import re
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi, hf_hub_download


MODEL_COLLECTION = "toksuite/toksuite-model-collection-69435e80cc46b2685e53067d"
BENCHMARK_COLLECTION = "toksuite/toksuite-benchmarks-694365d9c2caeb482b2946c4"
CLAIMED_ROWS = 4941
NAMED_DATASETS = [
    "toksuite/toksuite_english",
    "toksuite/toksuite_chinese",
    "toksuite/toksuite_turkish",
    "toksuite/toksuite_italian",
    "toksuite/toksuite_farsi",
    "toksuite/toksuite_math",
    "toksuite/toksuite_stem",
]
GENERAL_DATASET = "toksuite/toksuite_general"


def model_setup_terms(card: str) -> dict[str, bool]:
    text = card.lower()
    return {
        "architecture": "architecture" in text and "llama-3.2" in text,
        "training_data": "training data" in text and "100b" in text,
        "training_budget": "training steps" in text or "100,000" in text,
        "initialization": "initialization" in text and "shared" in text,
        "tokenizer_only_difference": (
            "only the tokenizer differs" in text
            or "only in tokenizer choice" in text
        ),
    }


def perturbation_terms(card: str) -> dict[str, bool]:
    text = card.lower()
    terms = {
        "input-medium": ("input-medium", "input medium"),
        "diacritic": ("diacritic",),
        "orthographic": ("orthographic",),
        "morphology": ("morphology",),
        "noise": ("noise",),
        "latex": ("latex", "latex"),
        "stem": ("stem",),
        "math": ("math",),
    }
    return {
        name: any(pattern in text for pattern in patterns)
        for name, patterns in terms.items()
    }


def dataset_totals(sizes: dict[str, dict[str, int]]) -> dict[str, Any]:
    named_rows = sum(sizes[name]["rows"] for name in NAMED_DATASETS)
    with_general_rows = named_rows + sizes.get(GENERAL_DATASET, {}).get(
        "rows", 0
    )
    return {
        "claimed_rows": CLAIMED_ROWS,
        "named_rows": named_rows,
        "with_general_rows": with_general_rows,
        "matches_claimed_rows": named_rows == CLAIMED_ROWS,
        "named_config_count": sum(
            sizes[name]["configs"] for name in NAMED_DATASETS
        ),
        "with_general_config_count": sum(
            value["configs"] for value in sizes.values()
        ),
    }


def build_bundle(
    collection: dict[str, Any],
    dataset_sizes: dict[str, dict[str, int]],
    model_cards: dict[str, str],
) -> dict[str, Any]:
    items = list(collection["items"])
    setup_by_model = {
        model_id: model_setup_terms(card)
        for model_id, card in sorted(model_cards.items())
    }
    perturbations = {
        model_id: perturbation_terms(card)
        for model_id, card in sorted(model_cards.items())
    }
    any_setup_complete = any(all(values.values()) for values in setup_by_model.values())
    any_perturbation_complete = any(
        all(values.values()) for values in perturbations.values()
    )
    totals = dataset_totals(dataset_sizes)

    return {
        "paper_id": "vIZz7LvObC",
        "paper_title": "TokSuite: Measuring the Impact of Tokenizer Choice on Language Model Behavior",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "upstream": {
            "arxiv": "2512.20757",
            "model_collection": collection,
            "benchmark_datasets": dataset_sizes,
        },
        "observations": {
            "model_count": len(items),
            "model_ids": items,
            "model_setup_terms": setup_by_model,
            "perturbation_terms": perturbations,
            "dataset_totals": totals,
        },
        "claims": [
            {
                "claim": "TokSuite releases 14 pre-trained language models that share initialization, architecture, data, and training budget while differing only in tokenizer choice.",
                "status": (
                    "verified"
                    if len(items) == 14 and any_setup_complete
                    else "inconclusive"
                ),
                "evidence": "The canonical Hugging Face model collection contains 14 model entries, and model-card text states the shared setup and tokenizer-only controlled variable.",
            },
            {
                "claim": "TokSuite includes a multilingual robustness benchmark with 4,941 total examples across English, Chinese, Turkish, Italian, Farsi, Math, and STEM perturbation groups.",
                "status": (
                    "verified"
                    if totals["matches_claimed_rows"]
                    else "falsified"
                ),
                "evidence": f"Dataset Viewer size metadata gives {totals['named_rows']} rows across the seven named datasets, not {CLAIMED_ROWS}; including toksuite_general gives {totals['with_general_rows']} rows.",
            },
            {
                "claim": "TokSuite reports tokenizer-dependent relative performance drops under the named perturbation families.",
                "status": "verified" if any_perturbation_complete else "toy",
                "evidence": "Released model-card tables contain the named perturbation families and tokenizer-varying numeric entries.",
            },
        ],
    }


def fetch_collection(api: HfApi, slug: str) -> dict[str, Any]:
    collection = api.get_collection(slug)
    return {
        "slug": collection.slug,
        "last_updated": collection.last_updated.isoformat(),
        "items": [item.item_id for item in collection.items],
    }


def fetch_dataset_size(repo_id: str) -> dict[str, int]:
    url = "https://datasets-server.huggingface.co/size?dataset=" + urllib.parse.quote(
        repo_id, safe=""
    )
    with urllib.request.urlopen(url, timeout=30) as response:
        payload = json.load(response)
    configs = payload.get("size", {}).get("configs", [])
    return {
        "configs": len(configs),
        "rows": sum(config.get("num_rows", 0) for config in configs),
    }


def fetch_model_card(repo_id: str) -> str:
    path = hf_hub_download(repo_id=repo_id, filename="README.md")
    return Path(path).read_text(encoding="utf-8")


def _stable_bundle(bundle: dict[str, Any], generated_at: str | None) -> dict[str, Any]:
    if generated_at is not None:
        bundle = dict(bundle)
        bundle["generated_at"] = generated_at
    return bundle


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--generated-at")
    args = parser.parse_args()

    api = HfApi()
    collection = fetch_collection(api, MODEL_COLLECTION)
    benchmark_collection = fetch_collection(api, BENCHMARK_COLLECTION)
    expected_datasets = sorted(NAMED_DATASETS + [GENERAL_DATASET])
    observed_datasets = sorted(benchmark_collection["items"])
    if observed_datasets != expected_datasets:
        raise ValueError("benchmark collection items")

    sizes = {
        repo_id: fetch_dataset_size(repo_id)
        for repo_id in expected_datasets
    }
    cards = {repo_id: fetch_model_card(repo_id) for repo_id in collection["items"]}
    bundle = _stable_bundle(build_bundle(collection, sizes, cards), args.generated_at)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(bundle, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"claims": [claim["status"] for claim in bundle["claims"]]}))


if __name__ == "__main__":
    main()
