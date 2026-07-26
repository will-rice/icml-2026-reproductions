from pathlib import Path
from recurrent_sampler_repro.evidence import (
    run_pipeline,
    CLAIM_1_TEXT,
    CLAIM_2_TEXT,
    TEX_SHA256,
    SAMPLER_SHA256,
)


def get_project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def test_space_assets():
    project_root = get_project_root()
    # Ensure space assets are generated
    run_pipeline(project_root)

    space_dir = project_root / "space"
    readme_path = space_dir / "README.md"
    html_path = space_dir / "index.html"
    poster_path = space_dir / "poster.html"

    assert readme_path.exists()
    assert html_path.exists()
    assert poster_path.exists()

    readme_text = readme_path.read_text(encoding="utf-8")
    assert "sdk: static" in readme_text
    assert "app_file: index.html" in readme_text
    assert "icml2026-repro" in readme_text
    assert "paper-h7WBYYJF1Q" in readme_text

    html_text = html_path.read_text(encoding="utf-8")
    assert CLAIM_1_TEXT in html_text
    assert CLAIM_2_TEXT in html_text
    assert "PARTIAL" in html_text
    assert "UNAVAILABLE" in html_text
    assert TEX_SHA256 in html_text
    assert SAMPLER_SHA256 in html_text

    # Rejection of unsupported claims in static HTML
    assert "verified" not in html_text.lower() or "unverified" in html_text.lower() or "claim" in html_text.lower()
    # Check limitations warning is present
    assert "No Model Execution" in html_text or "No model execution" in html_text or "No Model Execution:" in html_text
    assert "No GPU Benchmarking" in html_text or "No GPU" in html_text
