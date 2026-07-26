from pathlib import Path
import tomllib


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_editable_install_uses_source_tree_clean_build_backend() -> None:
    config = tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text())

    assert not list((PROJECT_ROOT / "src").glob("*.egg-info"))
    assert config["build-system"] == {
        "requires": ["hatchling"],
        "build-backend": "hatchling.build",
    }
    assert config["tool"]["hatch"]["build"]["targets"]["wheel"]["packages"] == [
        "src/recurrent_sampler_repro"
    ]
