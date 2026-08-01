import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from generate_evidence import build_evidence, find_evidence


def test_find_evidence_reports_required_patterns(tmp_path: Path) -> None:
    source = tmp_path / "source.txt"
    source.write_text(
        "Algorithm 1 Ambient Dataloops Training Algorithm. "
        "We perform posterior sampling and add restored point to dataset.",
        encoding="utf-8",
    )

    result = find_evidence(
        source,
        ["Algorithm 1", "posterior sampling", "restored point"],
    )

    assert result["found"] is True
    assert result["path"] == str(source)
    assert all(pattern in result["matched_patterns"] for pattern in ["Algorithm 1", "posterior sampling"])


def test_build_evidence_assigns_conservative_claim_statuses(tmp_path: Path) -> None:
    paper_text = tmp_path / "paper.txt"
    model_card = tmp_path / "model.md"
    space_app = tmp_path / "app.py"
    paper_text.write_text(
        "Algorithm 1 Ambient Dataloops Training Algorithm posterior sampling restored point. "
        "Table 1 CIFAR-10 with 90% corrupted and 10% clean data. "
        "Figure 3 madness regime. Table 2 COCO zero-shot generation and GenEval. "
        "Figure 4 achieves a 14.3% increase in diversity for a minor 0.2% in designability. "
        "Theoretical Modeling reduce the estimation error. Proof of Lemma 1.",
        encoding="utf-8",
    )
    model_card.write_text(
        "Ambient Dataloops demonstrates improvements in text-to-image generation. "
        "Data from DiffusionDB were treated as noisy samples, and refined once. "
        "Next, we use the trained model to refine the synthetic samples by using posterior samples.",
        encoding="utf-8",
    )
    space_app.write_text(
        "model_dict_path = hf_hub_download(repo_id=\"adrianrm/ambient-dataloops\", filename=\"model.safetensors\")",
        encoding="utf-8",
    )

    evidence = build_evidence(
        paper_text=paper_text,
        model_card=model_card,
        space_app=space_app,
        model_revision="7fe372764f0486d834a666ac0c71748bf4d84710",
        space_revision="7360baf66f929ad297b2b7d5684209e7c04b7414",
    )

    statuses = {claim["id"]: claim["status"] for claim in evidence["claims"]}
    assert statuses[1] == "verified"
    assert statuses[2] == "inconclusive"
    assert statuses[3] == "inconclusive"
    assert statuses[4] == "inconclusive"
    assert statuses[5] == "inconclusive"
    assert statuses[6] == "verified"
    assert evidence["artifact_revisions"]["model"] == "7fe372764f0486d834a666ac0c71748bf4d84710"
    assert evidence["artifact_revisions"]["space"] == "7360baf66f929ad297b2b7d5684209e7c04b7414"
    assert evidence["environment"] == {"python_requirement": ">=3.10,<3.13"}


def test_build_evidence_output_is_json_serializable(tmp_path: Path) -> None:
    paper_text = tmp_path / "paper.txt"
    model_card = tmp_path / "model.md"
    space_app = tmp_path / "app.py"
    paper_text.write_text("Algorithm 1 posterior sampling Theoretical Modeling Proof of Lemma 1", encoding="utf-8")
    model_card.write_text("DiffusionDB refined once posterior samples", encoding="utf-8")
    space_app.write_text("hf_hub_download(repo_id=\"adrianrm/ambient-dataloops\")", encoding="utf-8")

    evidence = build_evidence(
        paper_text=paper_text,
        model_card=model_card,
        space_app=space_app,
        model_revision="model-sha",
        space_revision="space-sha",
    )

    encoded = json.dumps(evidence, sort_keys=True)
    assert "Ambient Dataloops" in encoded
