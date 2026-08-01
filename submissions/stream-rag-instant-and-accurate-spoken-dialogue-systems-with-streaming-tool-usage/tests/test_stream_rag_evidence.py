import importlib.util
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
GENERATOR = PROJECT_ROOT / "generate_evidence.py"


def load_generator():
    spec = importlib.util.spec_from_file_location("stream_rag_generate", GENERATOR)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_streaming_scheduler_and_negative_sampling_logic():
    module = load_generator()

    chunks = ["who", "who founded", "who founded rare", "who founded rare beauty"]
    fixed = module.fixed_interval_queries(chunks)
    triggered = module.model_triggered_queries(chunks, threshold=0.45)

    assert len(fixed["threads_started"]) == 4
    assert triggered["max_parallel_threads"] == 1
    assert triggered["queries"][-1] == "who founded rare beauty"

    label = module.negative_sampling_label(
        current_query="who founded rare beauty",
        previous_query="red bull founder",
        negative_sample=True,
    )
    assert label == "who founded rare beauty"


def test_latency_and_accuracy_arithmetic():
    module = load_generator()
    checks = module.table_checks()

    assert checks["audio_crag"]["human_queries"] == 618
    assert checks["audio_crag"]["synthetic_queries"] == 1862
    assert round(checks["open_book_first_token_multiplier"], 1) == 4.4
    assert round(checks["streaming_tool_latency_saving_percent"], 1) == 20.7
    assert checks["qwen_accuracy"]["closed_book"] == 11.1
    assert checks["qwen_accuracy"]["streaming_rag"] == 34.2


def test_evidence_bundle_claims_and_limits(tmp_path):
    module = load_generator()
    output = tmp_path / "bundle.json"
    bundle = module.build_bundle()
    module.write_bundle(bundle, output)
    saved = json.loads(output.read_text())

    assert saved["paper"]["paper_id"] == "NMMmwSbzRx"
    assert saved["upstream"]["arxiv_source_sha256"] == (
        "e40c9783bb9c6f9b5995d08fac509cce3d03102ea9e48d35b94a47d4a96725a7"
    )
    assert len(saved["claims"]) == 6
    verdicts = {claim["challenge_claim_sha256"]: claim["verdict"] for claim in saved["claims"]}
    assert verdicts["acf84c2f5e061e004ba2df9f5a873cd0a42a1c4e7b0d153e3db783f5681b134b"] == "toy"
    assert verdicts["433318d8b20019620a1a2b923503d89374c5df98cf42b3bb7655fdab6b09e7fc"] == "inconclusive"
    assert saved["limitations"]["official_code_released"] is False
