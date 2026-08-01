from pathlib import Path

from dflash_repro.core import (
    build_evidence_bundle,
    load_source_snapshot,
    summarize_source,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_pinned_source_contains_block_diffusion_mechanism_markers():
    source = load_source_snapshot()
    summary = summarize_source(source)

    assert summary["github_revision"] == "94e4abc5e0c31b67bc1a9d30f1cc34ece28a8756"
    assert summary["has_dflash_generate"]
    assert summary["has_context_feature_extraction"]
    assert summary["has_target_hidden_conditioning"]
    assert summary["has_noise_embedding_draft_block"]
    assert summary["has_noncausal_draft_attention"]
    assert summary["has_parallel_block_acceptance"]


def test_pinned_source_exposes_required_benchmark_surfaces():
    summary = summarize_source(load_source_snapshot())

    assert summary["benchmark_backends"] == ["mlx", "sglang", "transformers", "vllm"]
    assert summary["benchmark_datasets"] == ["gsm8k", "humaneval", "math500", "mbpp", "mt-bench"]
    assert summary["readme_mentions_fa4"]
    assert summary["readme_mentions_qwen3_drafts"]


def test_evidence_bundle_keeps_gpu_speedups_inconclusive():
    bundle = build_evidence_bundle(load_source_snapshot())
    statuses = {
        claim["challenge_claim_sha256"]: claim["status"]
        for claim in bundle["claims"]
    }

    assert statuses["2637cad1833ecb87f838786ba8aee2364688f5c9a645ece89d4cf1fddbb26f68"] == "verified"
    assert statuses["83c2b07156c4deb43c365e160990f0cb39c9190c14d7ba862aac88079bce1551"] == "inconclusive"
    assert statuses["03c7bbcd902aee67b70a33a633fb2d90d249e2dceeca4971cd1771969db65076"] == "inconclusive"
    assert statuses["1c22abd30516043736ec10d93a157de5fc2216ef069661335d1c04d49c919ddf"] == "inconclusive"
    assert statuses["a591774b2149b43146f956d59c5df39775bd15ff9090f19e15a8bd91d141bdd1"] == "inconclusive"
    assert statuses["4e4cbcaf9b48d19f5fb2e11c9f92b04b30bcd8d97de43a3c95f4a3a20da1561e"] == "inconclusive"
    assert bundle["reproduced_speedup_measurements"] == []
    assert bundle["reproduced_ablation_measurements"] == []


def test_space_metadata_and_report_bind_attempt():
    readme = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
    report = (PROJECT_ROOT / "pages" / "report.md").read_text(encoding="utf-8")

    assert "icml2026-repro" in readme
    assert "paper-Oz335dV48X" in readme
    assert "DFlash" in report
    assert "94e4abc5e0c31b67bc1a9d30f1cc34ece28a8756" in report
    assert "GPU speedup claims are inconclusive" in report
