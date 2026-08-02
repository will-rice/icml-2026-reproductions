import json
import subprocess
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]


def test_generate_evidence_records_pinned_autobg_artifacts(tmp_path):
    output = tmp_path / "autobg_results.json"
    result = subprocess.run(
        ["uv", "run", "--project", str(PROJECT), "python", str(PROJECT / "generate_evidence.py"), "--output", str(output)],
        cwd=PROJECT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(output.read_text(encoding="utf-8"))

    assert payload["attempt_id"] == "1ff17cfb-669a-4192-90f0-c014220f7f12"
    assert payload["snapshot_id"] == "dc14d49cb209316c2a8f5cd9ff0e2ff27eacc29f5d2c4c4a2bacccca9ab6b4cc"
    assert payload["paper"]["paper_id"] == "75AYDsndHP"
    assert payload["upstream"]["code_commit"] == "21624a80504b3199b291514c37a49cccd19c8817"
    assert payload["upstream"]["robin_model_revision"] == "2813c971b63a177ad578c51c9a550c2e63e9168d"
    assert payload["upstream"]["many_peptides_revision"] == "1af9336878122eb1d62894fe2fb3ff4b801a3216"
    assert payload["upstream"]["estimated_api_cost_usd"] == 0.0

    source = payload["source_tree"]
    assert source["python_file_count"] >= 40
    assert source["config_file_count"] >= 60
    assert source["script_file_count"] >= 3
    assert source["license"] == "mit-plus-notice"

    support = payload["artifact_support"]
    assert support["autoregressive_module"] is True
    assert support["exact_log_likelihood"] is True
    assert support["importance_reweighting"] is True
    assert support["sequential_interventions"] is True
    assert support["chignolin_config"] is True
    assert support["robin_eval_script"] is True
    assert support["robin_checkpoint_lfs_size"] > 1_000_000_000

    claims = {claim["id"]: claim for claim in payload["claims"]}
    assert claims["arbg_likelihood_importance"]["status"] == "artifact_verified"
    assert claims["topology_interventions"]["status"] == "artifact_verified"
    assert claims["benchmark_improvements"]["status"] == "released_claim_values_unrecomputed"
    assert claims["robin_model"]["status"] == "metadata_verified"
    assert claims["robin_energy_reduction"]["status"] == "released_claim_values_unrecomputed"

    for claim in claims.values():
        assert claim["observed"]["source"] != "paper_prose"
        assert len(claim["challenge_claim_sha256"]) == 64

    assert any("headline" in note.lower() for note in payload["unavailable"])


def test_space_pages_include_scoring_surface():
    pages = sorted((PROJECT / "pages").glob("*.md"))
    assert len(pages) >= 2

    texts = [page.read_text(encoding="utf-8") for page in pages]
    assert sum(len(text.strip()) for text in texts) >= 200

    numeric_lines = [
        line
        for text in texts
        for line in text.splitlines()
        if any(character.isdigit() for character in line)
    ]
    assert len(numeric_lines) >= 15
