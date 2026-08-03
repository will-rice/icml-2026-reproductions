import subprocess
from pathlib import Path


def test_validation_lockfiles_are_versioned() -> None:
    project = Path(__file__).resolve().parents[1]
    repository = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        cwd=project,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    repository_path = Path(repository)

    for lockfile in (project / "uv.lock", project / "conversion" / "uv.lock"):
        subprocess.run(
            [
                "git",
                "ls-files",
                "--error-unmatch",
                str(lockfile.relative_to(repository_path)),
            ],
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
        )


def test_space_requirements_include_imported_model_dependency() -> None:
    project = Path(__file__).resolve().parents[1]
    requirements = (project / "requirements.txt").read_text().splitlines()

    assert "safetensors==0.6.2" in requirements
