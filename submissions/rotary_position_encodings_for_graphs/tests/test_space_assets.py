import importlib.util
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_scoring_pages_exist_for_official_judge():
    pages = sorted((PROJECT_ROOT / "pages").glob("*.md"))
    total_characters = sum(len(path.read_text(encoding="utf-8").strip()) for path in pages)

    assert total_characters >= 200


def test_space_app_imports_from_project_source():
    spec = importlib.util.spec_from_file_location("wire_space_app", PROJECT_ROOT / "app.py")
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)

    spec.loader.exec_module(module)

    assert hasattr(module, "demo")
