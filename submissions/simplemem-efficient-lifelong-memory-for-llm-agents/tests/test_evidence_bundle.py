import importlib.util
import json
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = Path("/tmp/simplemem-upstream-codex03")
ARXIV_SOURCE = Path("/tmp/simplemem-2601.02553-src.tar")
ARXIV_PDF = Path("/tmp/simplemem-2601.02553.pdf")


def load_generate_evidence():
    spec = importlib.util.spec_from_file_location(
        "simplemem_generate_evidence",
        PROJECT / "generate_evidence.py",
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def build_bundle(tmp_path):
    module = load_generate_evidence()
    output = tmp_path / "bundle.json"
    bundle = module.build_bundle(
        source_root=SOURCE_ROOT,
        arxiv_source=ARXIV_SOURCE,
        arxiv_pdf=ARXIV_PDF,
        output=output,
    )
    assert json.loads(output.read_text()) == bundle
    return bundle


def test_bundle_records_primary_artifact_provenance(tmp_path):
    bundle = build_bundle(tmp_path)

    assert bundle["attempt_id"] == "8d3d77be-6e6a-48a0-b50e-3a078786181d"
    assert bundle["paper_id"] == "oBgLvd5YC6"
    assert bundle["challenge_snapshot_id"] == "b4f93b39c8b36e72a5c6e1cb4712583e6f78510e414749bc064f46a09ce81885"
    assert bundle["provenance"]["code"]["revision"] == "db80b6a7c591e0ea730a058e9f5fc4eb06572299"
    assert bundle["provenance"]["arxiv_source_sha256"] == "d75d00ede2529a7656b6c6030161d489630dcf03a9f538dbd20322aa6a69f08a"
    assert bundle["provenance"]["arxiv_pdf_sha256"] == "8752aa223e004ca286995bc1e8cbde8e89e67ad3aeb9ba0266f3ccab3cc11078"
    assert bundle["provenance"]["license"] == "MIT"
    assert len(bundle["claim_bindings"]) == 6
    assert bundle["claim_bindings"][0]["challenge_claim_sha256"] == "c447ee8488120566377eecb650e62b3b423204b6facb5d5847e9a6d667f6f438"


def test_source_and_arxiv_evidence_detects_pipeline_and_tables(tmp_path):
    bundle = build_bundle(tmp_path)

    source = bundle["observations"]["source_release"]
    assert source["code_revision"] == "db80b6a7c591e0ea730a058e9f5fc4eb06572299"
    assert source["tracked_file_count"] > 100
    assert source["pipeline_terms"] == {
        "semantic_structured_compression": True,
        "online_semantic_synthesis": True,
        "intent_aware_retrieval_planning": True,
    }
    assert source["component_paths_present"] == {
        "memory_builder": True,
        "hybrid_retriever": True,
        "answer_generator": True,
    }
    assert source["arxiv_table_labels_present"] == {
        "high_capacity_results": True,
        "longmemeval_full": True,
        "efficient_results": True,
        "memory_time_accuracy": True,
        "ablation": True,
    }


def test_benchmark_observations_record_api_requirements_and_no_raw_results(tmp_path):
    bundle = build_bundle(tmp_path)

    benchmark = bundle["observations"]["benchmark_release"]
    assert benchmark["locomo_runner_present"] is True
    assert benchmark["evolvemem_runner_present"] is True
    assert benchmark["requires_openai_api_key"] is True
    assert benchmark["llm_judge_present"] is True
    assert benchmark["raw_result_artifact_count"] == 0
    assert benchmark["raw_result_artifacts"] == []


def test_claim_statuses_do_not_promote_paper_reported_values(tmp_path):
    bundle = build_bundle(tmp_path)
    statuses = {claim["claim_index"]: claim["status"] for claim in bundle["claims"]}

    assert statuses[1] == "toy"
    assert statuses[2] == "unavailable"
    assert statuses[3] == "unavailable"
    assert statuses[4] == "unavailable"
    assert statuses[5] == "unavailable"
    assert statuses[6] == "unavailable"
    assert all("paper_table" not in claim.get("evidence_basis", []) for claim in bundle["claims"])
    assert all("paper_reported_value" not in claim for claim in bundle["claims"])


def test_space_report_summarizes_bundle_statuses(tmp_path):
    bundle = build_bundle(tmp_path)
    bundle_path = tmp_path / "bundle.json"
    bundle_path.write_text(json.dumps(bundle), encoding="utf-8")

    spec = importlib.util.spec_from_file_location("simplemem_app", PROJECT / "app.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    report = module.build_markdown(bundle_path)

    assert "SimpleMem: Efficient Lifelong Memory for LLM Agents" in report
    assert "Attempt `8d3d77be-6e6a-48a0-b50e-3a078786181d`" in report
    assert "| 1 | toy |" in report
    assert "| 6 | unavailable |" in report
    assert "No paper-reported table value is treated as a reproduced measurement." in report
