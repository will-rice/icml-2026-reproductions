from __future__ import annotations

import json
import sys
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
SRC = PROJECT / "src"
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tedbench_repro.evidence import build_bundle


def test_bundle_records_pinned_sources_and_selected_claims():
    bundle = build_bundle()

    assert bundle["paper_id"] == "jPKqiaPTEd"
    assert bundle["upstream_pins"] == {
        "arxiv": "arxiv:2605.18552",
        "code_repo": "github:BorgwardtLab/TEDBench@ad3c208db13e5e0e124719300ec19fffab4c33e1",
        "ted_dataset": "hf-dataset:TEDBench/ted@825dfceb2acd92cebc62a5b1bb95e8a13407160a",
        "afdb_dataset": "hf-dataset:TEDBench/afdb@bb3caa7a24f9adf9758392298c88c76610dda2b5",
        "cath_dataset": "hf-dataset:TEDBench/cath@ce80cd3e7307bece444423ad0e32943ffce96f35",
        "miae_b_model": "hf-model:TEDBench/miae-b@864ad47cf09276d76df5d97ba2505db1f5dfc57d",
    }
    assert [claim["challenge_claim_sha256"] for claim in bundle["target_claims"]] == [
        "51f432978dc75b230f4bab006d45b7ef51f500cbbf68ba18606258d2750f87e4",
        "16690fadb62eb2ab2926b3c2a514d9509412d67db420202c8209dd7d114ae71f",
        "d3be0d566cf89cb984fe76a66979ca62f1358508bb9b63ea93cc0637abf853cb",
        "3316799a694033b48db8483f376328c5376835ee71c2085803991f60376aa0f5",
    ]


def test_bundle_contains_dataset_and_miae_observations():
    bundle = build_bundle()
    datasets = bundle["observations"]["datasets"]
    model = bundle["observations"]["models"]["miae_b"]

    assert datasets["ted"]["splits"] == {"train": 369740, "val": 46217, "test": 46218}
    assert datasets["ted"]["class_count"] == 965
    assert datasets["afdb"]["total_structures"] == 749679
    assert datasets["cath"] == {
        "repo": "TEDBench/cath",
        "revision": "ce80cd3e7307bece444423ad0e32943ffce96f35",
        "splits": {"test": 28010},
        "class_count": 965,
        "experimental": True,
        "source": "CATH 4.4 40% non-redundant representative set",
    }
    assert model["mask_ratio"] == 0.9
    assert model["has_geometric_encoder"] is True
    assert model["has_decoder"] is True
    assert model["decoder_embed_dim"] == 512


def test_generated_bundle_is_deterministic_json():
    import generate_evidence

    generate_evidence.main()
    first = json.loads((PROJECT / "evidence" / "bundle.json").read_text())
    generate_evidence.main()
    second = json.loads((PROJECT / "evidence" / "bundle.json").read_text())

    assert first == second == build_bundle()


def test_scoring_pages_present_substantive_markdown():
    pages = sorted((PROJECT / "pages").glob("*.md"))

    assert pages
    assert sum(len(page.read_text(encoding="utf-8").strip()) for page in pages) >= 200


def test_space_metadata_uses_valid_emoji_and_tags():
    readme = (PROJECT / "README.md").read_text(encoding="utf-8")

    assert 'emoji: "🧬"' in readme
    assert "paper-jPKqiaPTEd" in readme
    assert "icml2026-repro" in readme
