from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from generate_evidence import generate_evidence


def test_generate_evidence():
    bundle = generate_evidence()
    assert bundle["paper_id"] == "3BW15kSPfN"
    assert "claim_results" in bundle
    assert len(bundle["claim_results"]) == 2


def test_evidence_bundle_file():
    bundle_path = Path("evidence/bundle.json")
    if bundle_path.exists():
        data = json.loads(bundle_path.read_text(encoding="utf-8"))
        assert data["paper_id"] == "3BW15kSPfN"
