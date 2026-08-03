import subprocess
from pathlib import Path


def test_validation_lockfile_is_versioned() -> None:
    project = Path(__file__).resolve().parents[1]
    repository = Path(
        subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=project,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )

    subprocess.run(
        [
            "git",
            "ls-files",
            "--error-unmatch",
            str((project / "uv.lock").relative_to(repository)),
        ],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    )
