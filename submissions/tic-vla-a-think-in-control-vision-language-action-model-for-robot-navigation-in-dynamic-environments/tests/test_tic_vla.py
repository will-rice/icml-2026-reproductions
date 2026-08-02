import json
import subprocess
import sys
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]


def test_generated_bundle_does_not_promote_paper_tables_to_verified(tmp_path):
    """Catches generators that copy Table 2/Table 3 paper values as verified evidence."""

    subprocess.run(
        [
            sys.executable,
            str(PROJECT / "generate_evidence.py"),
            "--output-dir",
            str(tmp_path),
        ],
        check=True,
        cwd=PROJECT,
    )

    bundle = json.loads((tmp_path / "evidence" / "bundle.json").read_text())
    by_claim = {claim["claim"]: claim for claim in bundle["claims"]}

    table_claims = [
        claim
        for claim in by_claim.values()
        if "Table 2" in claim["claim"] or "Table 3" in claim["claim"]
    ]
    assert table_claims
    assert all(claim["status"] == "inconclusive" for claim in table_claims)
    assert all("not reproduced" in claim["evidence"].lower() for claim in table_claims)
    assert all("paper_reported_context" in claim for claim in table_claims)

    report = (tmp_path / "pages" / "01-measurements.md").read_text()
    assert "not reproduced" in report.lower()
    assert "| table 2 | inconclusive |" in report.lower()
    assert "| table 3 rtx 4060 | inconclusive |" in report.lower()
    assert "| table 3 jetson orin nx | inconclusive |" in report.lower()
