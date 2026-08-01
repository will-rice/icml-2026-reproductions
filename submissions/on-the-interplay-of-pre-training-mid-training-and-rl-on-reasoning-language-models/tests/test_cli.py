import subprocess
import sys
from pathlib import Path


def test_generate_evidence_help_runs_from_script_path():
    project_root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, str(project_root / "generate_evidence.py"), "--help"],
        cwd=project_root.parents[1],
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0
    assert "Generate Interplay-LM evidence bundle" in result.stdout
