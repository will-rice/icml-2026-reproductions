import subprocess
import sys


def test_generate_failure_preserves_existing_output(project_root, tmp_path):
    output = tmp_path / "evidence.json"
    output.write_bytes(b"preserve\n")
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "conditional_dpo_repro.cli",
            "generate",
            "--project-root",
            str(project_root / "missing"),
            "--output",
            str(output),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert output.read_bytes() == b"preserve\n"
    assert "Traceback" not in result.stderr
