import json
from pathlib import Path
import sys


PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

import generate_evidence  # noqa: E402


EXPECTED_CLAIMS = {
    "1d9a6dacc1dc82ef48de7d737894a7483656532009cd09ee5520fac3ff8cc6cb": "MemoryBench provides a three-module framework with a task provider, user simulator, and performance monitor for testing LLM-system continual learning from feedback logs (Figure 1)",
    "8e42a5968c6221140d8c838ad8a77d03d4650c7445048c57cea565257e4109e1": "MemoryBench covers 11 public datasets across three domains, four task-format categories, and two languages (Table 2)",
    "455c286b44e701d2952fbad8875c587b9222f0a041fc7766b27466bcdfe5d2db": "The benchmark includes both declarative/procedural memory and explicit/implicit feedback categories absent from prior memory benchmarks (Table 1)",
    "020b23a747bb1deb098786bc57173b48597869889509eb725fe0e6ceaa4c1e58": "Off-policy results show that advanced memory systems such as A-Mem, Mem0, and MemoryOS do not consistently outperform simpler RAG baselines (Figure 2)",
    "62864e8988dab4b163f8fbd1ddb98a9aafbf2869b98a4d10139d2cea998143e3": "Comparisons with and without feedback show simulated user feedback can improve model performance on task-specific metrics (Table 11)",
}


def test_generate_evidence_writes_bound_claim_results(tmp_path):
    bundle = generate_evidence.build_evidence(tmp_path)

    assert bundle["paper_id"] == "If4X4W2HWx"
    assert bundle["attempt_id"] == "bbdaa7e7-bb11-4d12-8f04-f0a93e54cf8d"
    assert bundle["snapshot_id"] == "82f8fe77e2f556bf4274e2b1d4a957bf9d1784978c27d31fc3ac6e4c1954f11b"
    assert bundle["api_cost_usd"] == 0.0
    assert set(bundle["upstream_revisions"]) == {
        "arxiv",
        "github_code",
        "hf_dataset",
        "hf_results",
    }

    results = {claim["claim_sha256"]: claim for claim in bundle["claims"]}
    assert set(results) == set(EXPECTED_CLAIMS)
    for digest, text in EXPECTED_CLAIMS.items():
        claim = results[digest]
        assert claim["claim"] == text
        assert claim["status"] in {"verified", "falsified", "toy", "inconclusive"}
        assert claim["computed_observations"]
        assert claim["paper_reported_context"]
        assert claim["paper_reported_context"] != claim["computed_observations"]

    written = tmp_path / "evidence" / "bundle.json"
    assert json.loads(written.read_text()) == bundle


def test_space_assets_exist_and_reference_evidence():
    for relative in ["README.md", "app.py", "pages/00-summary.md"]:
        path = PROJECT / relative
        assert path.exists(), relative
        assert "MemoryBench" in path.read_text()

    app = (PROJECT / "app.py").read_text()
    assert "bundle.json" in app
    assert "gradio" in app.lower()


def test_space_readme_declares_required_hub_metadata():
    readme = (PROJECT / "README.md").read_text()
    assert readme.startswith("---\n")
    metadata = readme.split("---", 2)[1]
    assert "sdk: gradio" in metadata
    assert "app_file: app.py" in metadata
    assert "icml2026-repro" in metadata
    assert "paper-If4X4W2HWx" in metadata
