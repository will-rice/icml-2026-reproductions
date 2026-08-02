import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def test_generate_evidence_records_pinned_upstream_and_all_claim_outcomes(tmp_path):
    """Catches missing upstream pinning or incomplete six-claim evidence bundles."""
    try:
        from src.frigid_repro import generate_evidence
    except ModuleNotFoundError as exc:
        pytest.fail(f"FRIGID evidence generator is missing: {exc}")

    output_path = tmp_path / "evidence.json"

    bundle = generate_evidence(output_path)

    assert output_path.exists()
    assert json.loads(output_path.read_text()) == bundle
    assert bundle["paper_id"] == "wTgx7b2D9r"
    assert bundle["attempt_id"] == "86bd82c3-48c0-4260-be38-045e8aa0fb29"
    assert bundle["upstream_revision"] == (
        "github:coleygroup/FRIGID@4914e52424278ac7de7b699fa7dfbee528cbc751"
    )
    assert [claim["ordinal"] for claim in bundle["claims"]] == [1, 2, 3, 4, 5, 6]
    assert {claim["local_outcome"] for claim in bundle["claims"]} <= {"supported", "limited"}
    assert bundle["claims"][0]["local_outcome"] == "supported"
    assert bundle["claims"][1]["local_outcome"] == "supported"
    assert bundle["claims"][2]["local_outcome"] == "limited"
    assert bundle["costs"]["api_cost_usd"] == 0.0
    assert Path(bundle["generated_files"]["evidence_json"]).name == "evidence.json"
