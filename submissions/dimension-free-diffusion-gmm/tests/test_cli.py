from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def run_cli(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "diffusion_gmm_repro.cli",
            *args,
        ],
        text=True,
        capture_output=True,
        check=False,
    )


def test_pilot_then_assemble_is_byte_deterministic(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    assert run_cli("pilot", "--output-dir", str(first), "--small-test").returncode == 0
    assert run_cli("assemble", "--output-dir", str(first)).returncode == 0
    assert run_cli("pilot", "--output-dir", str(second), "--small-test").returncode == 0
    assert run_cli("assemble", "--output-dir", str(second)).returncode == 0
    assert (first / "results.json").read_bytes() == (second / "results.json").read_bytes()
    assert (first / "measurements.csv").read_bytes() == (
        second / "measurements.csv"
    ).read_bytes()


def test_scaled_and_verify_commands(tmp_path: Path) -> None:
    output_dir = tmp_path / "scaled_run"
    assert (
        run_cli("scaled", "--output-dir", str(output_dir), "--small-test").returncode
        == 0
    )
    assert run_cli("assemble", "--output-dir", str(output_dir)).returncode == 0
    assert run_cli("verify", "--output-dir", str(output_dir)).returncode == 0


def test_cli_rejects_invalid_arguments(tmp_path: Path) -> None:
    result = run_cli("pilot", "--output-dir", str(tmp_path), "--invalid-flag")
    assert result.returncode != 0
