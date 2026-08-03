from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_scoring_pages_include_claim_hashes_and_substantive_text() -> None:
    page = ROOT / "pages" / "index.md"

    text = page.read_text(encoding="utf-8")

    assert len(text.strip()) >= 200
    assert "0199b3b43b308ce8469189f64e2310b12cb869a8c6255975c3c4cb7e9093f78a" in text
    assert "36d94de446993da42bb35022e284615dd122232225cf76fb0d3ccf26116e2788" in text
    assert "9e672b1894ac3fb6b00f8fa2a33a5d31355aa5663481e563c0594d337f0356b5" in text


def test_space_metadata_uses_huggingface_compatible_emoji() -> None:
    readme = ROOT / "README.md"
    text = readme.read_text(encoding="utf-8")
    frontmatter = text.split("---", 2)[1]
    metadata = dict(
        line.split(": ", 1)
        for line in frontmatter.splitlines()
        if ": " in line and not line.startswith("  - ")
    )

    assert metadata["emoji"]
    assert not metadata["emoji"].isascii()


def test_space_requirements_do_not_depend_on_local_editable_install() -> None:
    requirements = ROOT / "requirements.txt"
    lines = [
        line.strip()
        for line in requirements.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]

    assert "-e ." not in lines
    assert "." not in lines
    assert not any(line.startswith("-e ") for line in lines)


def test_space_app_bootstraps_src_imports_before_package_import() -> None:
    app = ROOT / "app.py"
    text = app.read_text(encoding="utf-8")

    bootstrap_index = text.index("sys.path.insert")
    package_import_index = text.index("from tau2_bench_repro.evidence import")

    assert bootstrap_index < package_import_index
