import json
import os
import sys
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from generate_evidence import generate_evidence

def test_evidence_generation():
    generate_evidence()
    assert os.path.exists("evidence/claim_1.json")
    assert os.path.exists("evidence/claim_2.json")

    with open("evidence/claim_1.json") as f:
        d1 = json.load(f)
    assert d1["total_questions"] == 2671
    assert d1["status"] == "verified"

    with open("evidence/claim_2.json") as f:
        d2 = json.load(f)
    assert d2["total_skills"] == 4
    assert set(d2["skills_breakdown"].keys()) == {"Recall", "Analyze", "Design", "Implement"}
    assert d2["status"] == "verified"

def test_pages_substantive_length():
    path = os.path.join("pages", "01_overview.md")
    assert os.path.exists(path)
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
    assert len(content.strip()) >= 200
