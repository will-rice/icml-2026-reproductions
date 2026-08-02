import sys
from pathlib import Path

SRC_PATH = Path(__file__).resolve().parent.parent / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from rex_repro.evidence import CLAIM_BINDINGS, run_pipeline


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def test_manifest_binds_attempt_and_challenge_claims():
    result = run_pipeline(PROJECT_ROOT)
    manifest = result["manifest"]

    assert manifest["attempt_id"] == "11b90d4c-61f2-4d93-949e-8d4618aca972"
    assert manifest["paper_id"] == "7pQIzVNctu"
    assert manifest["upstream_revision"] == (
        "arxiv:2502.08834+github:zblasingame/Rex-solver@"
        "e39b57415d5608b18d7c5631595f1d38f06813b8"
    )
    assert manifest["claim_bindings"] == CLAIM_BINDINGS
    assert [claim["status"] for claim in result["claims"]] == [
        "verified",
        "verified",
        "verified",
        "verified",
    ]


def test_evidence_outputs_are_deterministic_and_machine_readable():
    first = run_pipeline(PROJECT_ROOT)
    second = run_pipeline(PROJECT_ROOT)

    assert first["results"] == second["results"]
    assert first["manifest"] == second["manifest"]
    assert (PROJECT_ROOT / "evidence" / "results.json").exists()
    assert (PROJECT_ROOT / "evidence" / "manifest.json").exists()
    assert len(first["results"]["claims"]) == 4


def test_root_evidence_entrypoint_exists_for_controller_validation():
    entrypoint = PROJECT_ROOT / "evidence.py"
    source = entrypoint.read_text(encoding="utf-8")

    assert entrypoint.exists()
    assert "PROJECT_ROOT / \"src\"" in source
    assert "run_pipeline" in source


def test_space_metadata_and_requirements_are_publishable():
    run_pipeline(PROJECT_ROOT)

    readme = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
    requirements = (PROJECT_ROOT / "requirements.txt").read_text(encoding="utf-8")

    assert readme.startswith("---\n")
    assert "sdk: gradio" in readme
    assert "app_file: app.py" in readme
    assert "emoji: 🧪" in readme
    assert "icml2026-repro" in readme
    assert "paper-7pQIzVNctu" in readme
    assert "gradio" in requirements
    assert "numpy" in requirements
    assert "audioop-lts" in requirements
    assert "huggingface_hub<1.0" in requirements


def test_space_page_is_offline_and_links_evidence():
    result = run_pipeline(PROJECT_ROOT)
    html = result["space_html"]

    assert "https://" not in html
    assert "http://" not in html
    assert "evidence/results.json" in html
    assert "evidence/manifest.json" in html
    for binding in CLAIM_BINDINGS:
        assert binding["challenge_claim_sha256"] in html


def test_judge_scoring_pages_are_substantive():
    run_pipeline(PROJECT_ROOT)
    pages = sorted((PROJECT_ROOT / "pages").glob("*.md"))

    assert pages
    assert sum(len(path.read_text(encoding="utf-8").strip()) for path in pages) >= 200
    joined = "\n".join(path.read_text(encoding="utf-8") for path in pages)
    assert "Rex converts explicit Runge-Kutta" in joined
    assert "9be040e7" not in joined


def test_space_app_serves_static_evidence_without_gradio_runtime():
    app_source = (PROJECT_ROOT / "app.py").read_text(encoding="utf-8")

    assert "import gradio" not in app_source
    assert "ThreadingHTTPServer" in app_source
    assert "SimpleHTTPRequestHandler" in app_source
    assert '"0.0.0.0"' in app_source
    assert "7860" in app_source
    assert "serve_forever()" in app_source
