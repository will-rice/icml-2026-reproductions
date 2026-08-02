import json
import subprocess
import sys
from pathlib import Path


def test_generate_evidence_cli_writes_bundle(tmp_path):
    output = tmp_path / "bundle.json"
    result = subprocess.run(
        [sys.executable, "generate_evidence.py", "--offline-fixture", "--output", str(output)],
        check=True,
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
    )

    assert "wrote" in result.stdout.lower()
    bundle = json.loads(output.read_text())
    assert bundle["paper_id"] == "aJdgt8xDMy"
    assert len(bundle["claim_results"]) == 6
