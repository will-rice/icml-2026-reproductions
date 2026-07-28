from pathlib import Path
import socket
import pytest
import importlib.util

EXPECTED_PAGES = (
    "00-summary.md",
    "01-objective-losses.md",
    "02-cagrad-clip.md",
    "03-theorem-31.md",
    "04-theorem-32.md",
    "05-limitations-and-provenance.md",
)

EXPECTED_HASHES = (
    "e9a35e34b57a7273bf84d3d5981ab19f8ff1088adef8363f4640dcf436183944",
    "7c0aa54e034d03f2d0905417a024af4db41338cd9a803a0b42e441945c307cf9",
    "85abbc8a21d5c4537409f6e9f2af6bffc7e4c15e2311dfa078bf816ea0cffc9e",
    "dac93f364ac0469302894920781b034bfcd205816fbe16537c2f8e7c10d8995d",
    "269d8a5053e224206036399bccb2435455565149086de6439a9046d89682772b",
    "0d457572ea8a502fa8a489fef3e15da21b13cc39dd3a3730843d1cbe833059b0",
    "50719d645042a500f9c4d53fbdfedf719ee56429ddb73a231912f1eaeadb1b31",
    "5ec835ce150ff60d1e2bbd4fbdf7d1ebacf91bb2b6b2d65f72c44c3b3ed65229",
    "b74a0ea75967144b210934fd40fd23449d3ef985df878d5a9e14c4b04025ba4b",
    "58b31f527bb5e1bccb05c0dab775a74c2f2bdcd8e92ef2c0dd578733b5fb058e",
)


@pytest.fixture
def project_root() -> Path:
    return Path(__file__).parent.parent


def test_root_pages_are_direct_complete_and_judge_readable(project_root):
    pages = tuple(sorted(path.name for path in (project_root / "pages").glob("*.md")))
    assert pages == EXPECTED_PAGES
    summary = (project_root / "pages/00-summary.md").read_text("utf-8")
    for digest in EXPECTED_HASHES:
        assert digest in summary
    assert "not an official verdict" in summary.lower()
    for page_name in EXPECTED_PAGES:
        content = (project_root / "pages" / page_name).read_text("utf-8")
        assert len(content.strip()) >= 200, f"Page {page_name} must have >= 200 characters"


def fail_network(*args, **kwargs):
    raise RuntimeError("Network calls disabled during local app import")


def test_app_loads_only_committed_pages_and_evidence(project_root, monkeypatch):
    monkeypatch.setattr(socket, "create_connection", fail_network)
    app_path = project_root / "app.py"
    spec = importlib.util.spec_from_file_location("app", app_path)
    assert spec is not None and spec.loader is not None
    app_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(app_mod)

    assert app_mod.EVIDENCE["paper_id"] == "vSzRJyg6k0"
    assert tuple(path.name for path in app_mod.PAGE_PATHS) == EXPECTED_PAGES


def test_readme_frontmatter_short_description(project_root):
    readme_path = project_root / "README.md"
    content = readme_path.read_text(encoding="utf-8")
    assert content.startswith("---")
    parts = content.split("---", 2)
    assert len(parts) >= 3, "README must contain frontmatter enclosed in ---"
    frontmatter = parts[1]

    short_desc = None
    for line in frontmatter.splitlines():
        line = line.strip()
        if line.startswith("short_description:"):
            short_desc = line.split(":", 1)[1].strip()
            if (short_desc.startswith('"') and short_desc.endswith('"')) or (
                short_desc.startswith("'") and short_desc.endswith("'")
            ):
                short_desc = short_desc[1:-1]
            break

    assert short_desc is not None, "README frontmatter missing 'short_description'"
    assert (
        len(short_desc) <= 60
    ), f"short_description length {len(short_desc)} exceeds 60 characters: '{short_desc}'"
    assert "RACO" in short_desc and "ICML 2026" in short_desc


def test_readme_frontmatter_python_version(project_root):
    readme_path = project_root / "README.md"
    content = readme_path.read_text(encoding="utf-8")
    assert content.startswith("---")
    parts = content.split("---", 2)
    assert len(parts) >= 3, "README must contain frontmatter enclosed in ---"
    frontmatter = parts[1]

    python_ver = None
    for line in frontmatter.splitlines():
        line = line.strip()
        if line.startswith("python_version:"):
            python_ver = line.split(":", 1)[1].strip()
            if (python_ver.startswith('"') and python_ver.endswith('"')) or (
                python_ver.startswith("'") and python_ver.endswith("'")
            ):
                python_ver = python_ver[1:-1]
            break

    assert python_ver is not None, "README frontmatter missing 'python_version'"
    assert python_ver == "3.12", f"python_version must be '3.12', got '{python_ver}'"


def test_readme_frontmatter_sdk_version(project_root):
    readme_path = project_root / "README.md"
    content = readme_path.read_text(encoding="utf-8")
    assert content.startswith("---")
    parts = content.split("---", 2)
    assert len(parts) >= 3, "README must contain frontmatter enclosed in ---"
    frontmatter = parts[1]

    sdk_ver = None
    for line in frontmatter.splitlines():
        line = line.strip()
        if line.startswith("sdk_version:"):
            sdk_ver = line.split(":", 1)[1].strip()
            if (sdk_ver.startswith('"') and sdk_ver.endswith('"')) or (
                sdk_ver.startswith("'") and sdk_ver.endswith("'")
            ):
                sdk_ver = sdk_ver[1:-1]
            break

    assert sdk_ver is not None, "README frontmatter missing 'sdk_version'"
    assert sdk_ver == "6.20.0", f"sdk_version must be '6.20.0', got '{sdk_ver}'"
