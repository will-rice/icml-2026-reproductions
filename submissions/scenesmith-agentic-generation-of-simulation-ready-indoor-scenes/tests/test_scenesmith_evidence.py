import json
import subprocess
import sys
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]


def test_generate_evidence_reports_source_verified_and_unavailable_claims(tmp_path):
    output = tmp_path / "results.json"

    result = subprocess.run(
        [
            sys.executable,
            str(PROJECT / "generate_evidence.py"),
            "--output",
            str(output),
        ],
        cwd=PROJECT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(output.read_text(encoding="utf-8"))
    statuses = {claim["id"]: claim["status"] for claim in payload["claims"]}

    assert statuses == {
        "pipeline_stages": "source_verified",
        "agent_roles": "source_verified",
        "asset_integration": "source_verified",
        "large_scale_metrics": "unavailable",
        "user_study": "unavailable",
        "robot_eval": "source_verified",
    }
    assert payload["upstream"]["commit"] == "67cc408fd38334b4a926efef45e284302ed5055b"
    assert "210 prompt generated-scene metrics" in payload["missing_artifacts"]
    assert "205 participant user-study records" in payload["missing_artifacts"]


def test_space_readme_declares_controller_tags():
    readme = PROJECT / "README.md"

    text = readme.read_text(encoding="utf-8")

    assert "sdk: gradio" in text
    assert "app_file: app.py" in text
    assert "icml2026-repro" in text
    assert "paper-WwS8CTpUA6" in text
