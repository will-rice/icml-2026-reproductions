import json
import subprocess
import sys
from pathlib import Path


def test_generate_evidence_cli_writes_bundle(tmp_path):
    output = tmp_path / "bundle.json"
    project = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, "generate_evidence.py", "--offline-fixture", "--output", str(output)],
        cwd=project,
        check=True,
        text=True,
        capture_output=True,
    )

    assert "wrote" in result.stdout.lower()
    bundle = json.loads(output.read_text())
    assert bundle["paper_id"] == "ldCiNVFt8O"
    assert len(bundle["claim_results"]) == 6
