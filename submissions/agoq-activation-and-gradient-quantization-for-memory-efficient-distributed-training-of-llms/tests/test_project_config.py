from pathlib import Path


def test_uv_does_not_install_editable_metadata_into_source_tree() -> None:
    project = Path(__file__).resolve().parents[1]
    config = (project / "pyproject.toml").read_text()

    assert "[tool.uv]\npackage = false" in config


def test_space_pins_python_with_gradio_audioop_support() -> None:
    project = Path(__file__).resolve().parents[1]
    readme = (project / "README.md").read_text()

    assert 'python_version: "3.12"' in readme


def test_space_caps_huggingface_hub_for_gradio_4() -> None:
    project = Path(__file__).resolve().parents[1]

    requirements = (project / "requirements.txt").read_text().splitlines()
    assert "huggingface-hub<1.0" in requirements


def test_space_caps_starlette_for_gradio_4_template_api() -> None:
    project = Path(__file__).resolve().parents[1]

    requirements = (project / "requirements.txt").read_text().splitlines()
    assert "starlette<1.0" in requirements
