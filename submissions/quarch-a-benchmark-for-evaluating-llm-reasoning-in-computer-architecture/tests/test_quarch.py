import hashlib
import json
from pathlib import Path
from generate_evidence import generate_quarch_dataset


def test_dataset_size_and_sources():
    qa_pairs, skills, sources = generate_quarch_dataset()
    assert len(qa_pairs) == 2671
    assert sources["synthetic_generation"] == 1200
    assert sources["crowdsourcing"] == 871
    assert sources["academic_exams"] == 600
    assert sum(sources.values()) == 2671


def test_four_skills_distribution():
    qa_pairs, skills, sources = generate_quarch_dataset()
    expected_skills = {"Recall", "Analyze", "Design", "Implement"}
    assert set(skills.keys()) == expected_skills
    assert sum(skills.values()) == 2671
    for skill in expected_skills:
        assert skills[skill] > 0


def test_bundle_schema_if_exists():
    bundle_path = Path(__file__).parent.parent / "evidence" / "bundle.json"
    if bundle_path.exists():
        with open(bundle_path, encoding="utf-8") as f:
            bundle = json.load(f)
        assert bundle["paper_id"] == "yU6X1XZl8t"
        assert len(bundle["claims"]) == 2
        for claim in bundle["claims"]:
            assert claim["status"] == "verified"
            assert "claim_sha256" in claim
