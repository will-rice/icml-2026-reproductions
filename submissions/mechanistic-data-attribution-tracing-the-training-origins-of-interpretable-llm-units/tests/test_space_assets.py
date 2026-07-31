import importlib.util
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def test_space_app_imports_from_project_source():
    spec = importlib.util.spec_from_file_location("space_app", PROJECT_ROOT / "app.py")
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert hasattr(module, "demo")


def test_evidence_bundle_generation(tmp_path: Path):
    from generate_evidence import generate_bundle

    bundle = generate_bundle(tmp_path)
    assert bundle["paper_id"] == "PQaxfoEcRc"
    assert len(bundle["claims"]) == 3
    assert (tmp_path / "bundle.json").exists()
