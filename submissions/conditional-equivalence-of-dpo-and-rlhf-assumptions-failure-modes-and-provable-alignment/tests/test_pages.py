import pytest


def test_readme_has_exact_static_space_metadata(project_root):
    readme = (project_root / "README.md").read_text("utf-8")
    assert "sdk: static" in readme
    assert "app_file: index.html" in readme
    assert "  - paper-7UEBX1KU1y" in readme
    assert "  - icml2026-repro" in readme


@pytest.mark.parametrize("page_name", ["index.html", "poster.html"])
def test_pages_use_only_committed_root_evidence(project_root, page_name):
    page = (project_root / page_name).read_text("utf-8")
    assert 'fetch("./evidence.json")' in page
    assert "https://" not in page
    assert "http://" not in page
    assert "innerHTML" not in page


def test_pages_dir_contains_substantive_judge_readable_evidence(project_root):
    pages_dir = project_root / "pages"
    assert pages_dir.is_dir()
    md_files = list(pages_dir.glob("*.md"))
    assert len(md_files) >= 1
    total_chars = sum(len(f.read_text("utf-8").strip()) for f in md_files)
    assert total_chars >= 200


def test_pages_index_cpo_gamma_grid_matches_code(project_root):
    from conditional_dpo_repro.grids import CPO_GAMMAS

    index_md = (project_root / "pages" / "index.md").read_text("utf-8")
    expected_gammas = ", ".join(f"{g:g}" for g in CPO_GAMMAS)
    assert f"\\gamma \\in \\{{{expected_gammas}\\}}" in index_md


def test_pages_state_honest_limits(project_root):
    text = (
        (project_root / "index.html").read_text("utf-8")
        + (project_root / "poster.html").read_text("utf-8")
        + (project_root / "README.md").read_text("utf-8")
    ).lower()
    assert "no language model was trained" in text
    assert "benchmark sota claim was not reproduced" in text
    assert "official verdict" in text
