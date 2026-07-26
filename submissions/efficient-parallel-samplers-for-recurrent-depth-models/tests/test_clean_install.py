from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_editable_install_leaves_source_tree_clean() -> None:
    assert not list((PROJECT_ROOT / "src").glob("*.egg-info"))
