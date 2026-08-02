import json
import subprocess
import sys
from pathlib import Path


def test_generate_evidence_cli_writes_requested_bundle_only(tmp_path):
    output = tmp_path / "bundle.json"
    project = Path(__file__).resolve().parents[1]
    project_bundle = project / "evidence" / "bundle.json"
    if project_bundle.exists():
        project_bundle.unlink()

    result = subprocess.run(
        [sys.executable, "generate_evidence.py", "--offline-fixture", "--output", str(output)],
        cwd=project,
        check=True,
        text=True,
        capture_output=True,
    )

    assert "wrote" in result.stdout.lower()
    bundle = json.loads(output.read_text())
    assert bundle["paper_id"] == "vCc2NAe0OS"
    assert len(bundle["claim_results"]) == 6
    assert not project_bundle.exists()
