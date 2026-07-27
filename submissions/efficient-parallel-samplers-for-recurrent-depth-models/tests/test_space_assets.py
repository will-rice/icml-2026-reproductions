from pathlib import Path
from recurrent_sampler_repro.evidence import run_pipeline, CLAIM_1_SHA256, CLAIM_2_SHA256


def get_project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def test_space_assets_offline_and_attributions():
    project_root = get_project_root()
    res = run_pipeline(project_root)
    html_content = res["space_html"]

    assert "fonts.googleapis.com" not in html_content
    assert "fonts.gstatic.com" not in html_content
    assert "http://" not in html_content and "https://" not in html_content

    assert CLAIM_1_SHA256 in html_content
    assert CLAIM_2_SHA256 in html_content

    assert "Apache-2.0" in html_content
    assert "CC-BY-4.0" in html_content

    # Direct static-Space download links
    assert 'href="evidence/manifest.json"' in html_content or 'href="evidence/manifest.json" download' in html_content
    assert 'href="evidence/claim-1-wavefront.json"' in html_content
    assert 'href="evidence/claim-2-theorem-audit.json"' in html_content
    assert 'href="evidence/results.json"' in html_content
    assert 'href="evidence/REPORT.md"' in html_content or 'href="REPORT.md"' in html_content

    for line in html_content.splitlines():
        assert line == line.rstrip()


def test_validated_project_root_is_the_static_space():
    project_root = get_project_root()
    run_pipeline(project_root)

    readme = (project_root / "README.md").read_text()
    assert readme.startswith("---\n")
    assert "sdk: static" in readme
    assert "app_file: index.html" in readme
    assert "icml2026-repro" in readme
    assert "paper-h7WBYYJF1Q" in readme

    for name in ("index.html", "poster.html", "REPORT.md"):
        assert (project_root / name).read_bytes() == (
            project_root / "space" / name
        ).read_bytes()
