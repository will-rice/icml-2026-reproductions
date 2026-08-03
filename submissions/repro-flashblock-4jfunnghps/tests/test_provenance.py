import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_space_metadata_is_present_and_tagged():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")

    assert readme.startswith("---\n")
    assert "sdk: gradio" in readme
    assert "app_file: app.py" in readme
    assert "icml2026-repro" in readme
    assert "paper-4jfuNNghPS" in readme
    assert (ROOT / "requirements.txt").exists()
    pages = list((ROOT / "pages").glob("*.md"))
    assert pages
    assert sum(len(path.read_text(encoding="utf-8").strip()) for path in pages) >= 200


def test_evidence_summary_matches_fenced_attempt():
    evidence = json.loads((ROOT / "evidence_summary.json").read_text(encoding="utf-8"))

    assert evidence["paper_id"] == "4jfuNNghPS"
    assert evidence["attempt_id"] == "ee4b5986-ff11-4f99-9a93-cd8fc43eb04d"
    assert evidence["snapshot_id"] == "c68adfe585882f99e8f3dd3ed496aedc650f5b64684955045d04513816cbe106"
    assert evidence["challenge_revision"] == "81166abbeb76e5f79ff87e51061b5a0306507203"
    assert evidence["upstream_revision"] == "arxiv:2602.05305v3"
    assert evidence["space_id"] == "wrice/repro-flashblock-4jfunnghps"
    assert evidence["estimated_api_cost_usd"] == 0.0

    claim_hashes = {
        binding["challenge_claim_sha256"]
        for binding in evidence["claim_bindings"]
    }
    assert claim_hashes == {
        "749abab004dce42ccbe424cda535117dc3025a9889d0030d629555114b6a2dc5",
        "c3322b9476700a79a6ac2599ca9cd93ec9d26950e61efe364258c0026d102e2b",
        "a78798faeae29d98737f8d391f54be4cc938111ce6fb1d7f29463c75319d607d",
    }
