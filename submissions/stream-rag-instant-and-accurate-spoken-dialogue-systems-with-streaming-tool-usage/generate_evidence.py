from __future__ import annotations

import argparse
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
PAPER_ID = "NMMmwSbzRx"
ATTEMPT_ID = "21d5e464-8ffa-4832-ab65-f14d5681a317"
ARXIV_ID = "2510.02044v1"
ARXIV_SOURCE_SHA256 = "e40c9783bb9c6f9b5995d08fac509cce3d03102ea9e48d35b94a47d4a96725a7"
GENERATED_AT = "2026-08-01T13:18:00+00:00"


CLAIMS = [
    {
        "challenge_claim_sha256": "acf84c2f5e061e004ba2df9f5a873cd0a42a1c4e7b0d153e3db783f5681b134b",
        "claim": "Stream RAG issues tool queries in parallel with incoming user speech to reduce user-perceived latency in speech-in speech-out dialogue systems (Figure 1; Figure 3).",
        "verdict": "toy",
        "evidence": "The TeX defines fixed-interval Streaming RAG as B parallel tool-call threads and model-triggered Streaming RAG as a single active thread. The bundle implements both schedulers and verifies earlier tool-result availability in a toy setting, but no production speech model or tool logs were released.",
    },
    {
        "challenge_claim_sha256": "433318d8b20019620a1a2b923503d89374c5df98cf42b3bb7655fdab6b09e7fc",
        "claim": "The paper introduces AudioCRAG by converting CRAG question-answer tasks into speech form, including a human-recorded subset of 618 spoken queries (Section 4.1).",
        "verdict": "inconclusive",
        "evidence": "The TeX states 2,706 CRAG queries, 1,862 filtered synthetic spoken queries, and 618 human-recorded spoken queries. The actual AudioCRAG-Human audio/data files were not found, so this is a source audit rather than dataset verification.",
    },
    {
        "challenge_claim_sha256": "5768884d9000adddd250a5fb015075eff1eb2dd1e46c27ee5f04f940022d2547",
        "claim": "Tool integration improves factual QA accuracy for speech-in speech-out systems but increases first-token latency by about 2.3x in the open-book setting (Table 1; Table 3).",
        "verdict": "inconclusive",
        "evidence": "Table 1 reports Qwen2.5-7B speech accuracy increasing from 11.1 to 26.3 with tools, while first-token latency increases from 1.34 s to 5.90 s. The table arithmetic is 4.4x for total first-token latency, not independently reproduced and not matching the 2.3x wording directly.",
    },
    {
        "challenge_claim_sha256": "c866ddaa3c108914aa35159b87963f756eeb354a56e42950f8e449d3b49f7442",
        "claim": "Model-triggered Stream RAG improves Qwen2.5-7B AudioCRAG accuracy from 11.1% closed-book to 34.2% absolute while reducing tool-use latency by about 20% (Table 1).",
        "verdict": "inconclusive",
        "evidence": "Table 1 reports the 11.1 to 34.2 accuracy change and 20.7% synthetic latency savings. This bundle verifies arithmetic only; post-training, model inference, retrieval, and evaluation were not rerun.",
    },
    {
        "challenge_claim_sha256": "5affae823b0b1c2c227fdd904d5dfb50302c46d3607185fc6d4445132b87ed7f",
        "claim": "Latency breakdowns show Stream RAG reduces P50 first-token latency and tool-result latency relative to open-book tool use (Table 3).",
        "verdict": "inconclusive",
        "evidence": "Table 3 reports P50 total first-token latency falling from 5.90 s to 5.32 s and tool-result generation from 2.78 s to 2.20 s. These are source-table checks only, with no timing logs or executable system.",
    },
    {
        "challenge_claim_sha256": "9f9976b557d6365c4818638c70cde3927cab2d2c11570b9ae299ca589360101a",
        "claim": "Negative sampling during model-triggered Stream RAG post-training improves robustness to intermediate query-prediction errors (Table 5).",
        "verdict": "toy",
        "evidence": "The TeX defines a negative-sampling fallback that substitutes incorrect previous queries with the pseudo ground-truth current query. The toy labeler verifies that behavior, while Table 5 accuracy gains remain unreproduced without training/evaluation code.",
    },
]


def fixed_interval_queries(chunks: list[str]) -> dict:
    return {
        "threads_started": [{"block": i + 1, "query": chunk} for i, chunk in enumerate(chunks)],
        "max_parallel_threads": len(chunks),
    }


def similarity(a: str, b: str) -> float:
    a_tokens = set(a.lower().split())
    b_tokens = set(b.lower().split())
    if not a_tokens and not b_tokens:
        return 1.0
    return len(a_tokens & b_tokens) / len(a_tokens | b_tokens)


def model_triggered_queries(chunks: list[str], threshold: float) -> dict:
    active_query = ""
    queries = []
    for chunk in chunks:
        has_new_information = len(chunk.split()) > len(active_query.split())
        if not active_query or similarity(chunk, active_query) < threshold or has_new_information:
            active_query = chunk
            queries.append(chunk)
        else:
            queries.append("NO_QUERY")
    return {"queries": queries, "max_parallel_threads": 1}


def negative_sampling_label(current_query: str, previous_query: str, negative_sample: bool) -> str:
    if negative_sample and similarity(current_query, previous_query) < 0.5:
        return current_query
    if similarity(current_query, previous_query) >= 0.8:
        return "NO_QUERY"
    return current_query


def table_checks() -> dict:
    closed_latency = 1.34
    open_latency = 5.90
    open_tool_results = 2.78
    streaming_tool_results = 2.20
    return {
        "audio_crag": {
            "crag_text_queries": 2706,
            "rewritten_queries": 569,
            "synthetic_queries": 1862,
            "human_queries": 618,
        },
        "qwen_accuracy": {
            "closed_book": 11.1,
            "open_book": 26.3,
            "streaming_rag": 34.2,
            "streaming_absolute_gain": 23.1,
            "streaming_relative_gain_percent": round((34.2 - 11.1) / 11.1 * 100, 1),
        },
        "open_book_first_token_multiplier": round(open_latency / closed_latency, 3),
        "streaming_first_token_saving_percent": round((open_latency - 5.32) / open_latency * 100, 1),
        "streaming_tool_result_column_saving_percent": round(
            (open_tool_results - streaming_tool_results) / open_tool_results * 100, 1
        ),
        "streaming_tool_latency_saving_percent": 20.7,
        "negative_sampling_table": {"post_train": 39.8, "minus_negative_sampling": 36.5},
    }


def build_bundle() -> dict:
    chunks = ["who", "who founded", "who founded rare", "who founded rare beauty"]
    return {
        "paper": {
            "paper_id": PAPER_ID,
            "attempt_id": ATTEMPT_ID,
            "title": "Stream RAG: Instant and Accurate Spoken Dialogue Systems with Streaming Tool Usage",
        },
        "upstream": {
            "primary_artifact": f"arxiv:{ARXIV_ID}",
            "arxiv_source_sha256": ARXIV_SOURCE_SHA256,
            "source_archive_path_observed": "/tmp/stream-rag-arxiv-XXnqSO/source.tar",
            "official_code_released": False,
        },
        "generated_at": GENERATED_AT,
        "estimated_paid_api_cost_usd": 0.0,
        "toy_checks": {
            "fixed_interval": fixed_interval_queries(chunks),
            "model_triggered": model_triggered_queries(chunks, threshold=0.45),
            "negative_sampling_recovery_label": negative_sampling_label(
                "who founded rare beauty",
                "red bull founder",
                negative_sample=True,
            ),
        },
        "checks": table_checks(),
        "claims": CLAIMS,
        "limitations": {
            "official_code_released": False,
            "audiocrag_human_files_released": False,
            "model_training_or_eval_rerun": False,
            "note": "Quantitative benchmark and latency claims are paper-source audits, not reproduced measurements.",
        },
    }


def write_bundle(bundle: dict, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(bundle, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "evidence" / "bundle.json")
    args = parser.parse_args()
    bundle = build_bundle()
    write_bundle(bundle, args.output)
    print(json.dumps({"claims": len(bundle["claims"]), "output": str(args.output)}))


if __name__ == "__main__":
    main()
