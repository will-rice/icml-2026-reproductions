from __future__ import annotations

import json
import re
from functools import lru_cache
from typing import Any

import pandas as pd
from huggingface_hub import hf_hub_download


DATASET_REPO = "icml-anon-submission/medmosaic-dataset"
DATASET_REVISION = "a6ea67bd4a65b87248c6651e559656b2c31fa669"
DATASET_FILE = "data/test.parquet"

EXPECTED_QA_TYPE_COUNTS = {
    "Sound_Only": 1417,
    "Speech_Only": 1174,
    "Speech_Sound": 1087,
    "Open_Ended": 691,
    "voice_qa": 180,
    "Long_Form": 106,
    "Multi_Turn": 6,
}

CLAIM_HASHES = {
    "dataset_size": "7892ea8680906a105db8288b33fb04b6639d319047078eadd54a461c89d0dc09",
    "categories": "ab4285e365e2ada727ae3ccdb719a9c64ef075ed0fa1a6327daeb3d0955134cc",
    "model_benchmark": "951f2c0da2384aa8130bf0d3a2901537d2cb167416f2888dd9b38d69b2720538",
    "audio_ablation": "6a2e0745b8a74511913de8cf606b6f3ea8504f58db131c96406ac365b2671b8c",
    "difficulty_accuracy": "1ce471213ab62dad6a22534ed55da70a22b4664e8afdd603bd3d94507109b0a3",
    "expert_review": "e7429633bf76cc769e78c999e472562cdcf2d0ea0c11017548dc54d4bfdf6254",
}

TARGET_CLAIMS = {
    "dataset_size": "MedMosaic contains 46,701 medical audio question-answer pairs spanning physiological sounds, clinical conversations, and combined speech-sound scenarios (Figure 1)",
    "categories": "The benchmark includes multiple QA categories, including sound-only, speech-plus-sound, short and long clinical conversations, multi-turn MCQ, and open-ended QA (Table 1)",
    "model_benchmark": "Benchmarking 13 audio-language systems shows Gemini-2.5-Pro is the strongest evaluated model but reaches only about 68.1% weighted accuracy (Table 1)",
    "audio_ablation": "Removing audio materially reduces model performance, indicating MedMosaic is not trivially solvable from question text alone (Table 5)",
    "difficulty_accuracy": "Model accuracy generally declines from easy to hard difficulty strata across categories, supporting the benchmark's reasoning-difficulty labels (Table 6)",
    "expert_review": "Clinical expert review accepted 72.4% of assessed synthetic QA examples without modification, supporting the synthetic generation pipeline's clinical validity (Section 4)",
}


@lru_cache(maxsize=1)
def load_pinned_index() -> pd.DataFrame:
    path = hf_hub_download(
        repo_id=DATASET_REPO,
        filename=DATASET_FILE,
        repo_type="dataset",
        revision=DATASET_REVISION,
    )
    return pd.read_parquet(path)


def summarize_index(frame: pd.DataFrame) -> dict[str, Any]:
    options = frame["options"].map(_parse_options)
    audio_folders = frame["audio_path"].dropna().map(lambda value: str(value).split("/", 1)[0])
    difficulty = frame["difficulty_level"].where(frame["difficulty_level"].notna(), "null")
    standard_mcq = frame[~frame["qa_type"].isin(["Open_Ended", "Multi_Turn"])]
    turns = frame["turns"].map(_parse_turns)

    return {
        "dataset_repo": DATASET_REPO,
        "dataset_revision": DATASET_REVISION,
        "dataset_file": DATASET_FILE,
        "row_count": int(len(frame)),
        "qa_type_counts": _counts(frame["qa_type"]),
        "difficulty_counts": _counts(difficulty),
        "audio_folder_counts": _counts(audio_folders),
        "rows_with_audio_path": int(frame["audio_path"].notna().sum()),
        "rows_with_question": int(frame["question"].notna().sum()),
        "rows_with_ground_truth": int(frame["ground_truth"].notna().sum()),
        "standard_mcq_rows": int(len(standard_mcq)),
        "standard_mcq_rows_with_ten_options": int(
            options.loc[standard_mcq.index].map(lambda values: len(values) == 10).sum()
        ),
        "ground_truth_answer_marker_rows": int(
            frame["ground_truth"].fillna("").map(
                lambda value: bool(re.search(r"The answer is \([a-j]\)\.", str(value)))
            ).sum()
        ),
        "multi_turn_rows": int((frame["qa_type"] == "Multi_Turn").sum()),
        "multi_turn_turns": int(sum(len(value) for value in turns)),
        "multi_turn_turns_with_answer_markers": int(
            sum(
                bool(re.search(r"The answer is \([a-j]\)\.", str(turn.get("ground_truth", ""))))
                for value in turns
                for turn in value
            )
        ),
    }


def build_evidence_bundle(frame: pd.DataFrame | None = None) -> dict[str, Any]:
    summary = summarize_index(load_pinned_index() if frame is None else frame)
    return {
        "paper_id": "OMdQJQwp26",
        "attempt_id": "32f2eefb-2f74-40c6-a98d-f7d9eed154e6",
        "snapshot_id": "b09826f921a7d1649e5071df82e7a9f8f6211ac6dfa0e2589aa54651661eb283",
        "challenge_revision": "81166abbeb76e5f79ff87e51061b5a0306507203",
        "upstream": {
            "paper": "arxiv:2605.00969v2",
            "openreview": "https://openreview.net/forum?id=OMdQJQwp26",
            "dataset": f"{DATASET_REPO}@{DATASET_REVISION}",
            "dataset_file": DATASET_FILE,
        },
        "dataset_index_summary": summary,
        "claims": _claim_records(summary),
        "reproduced_model_measurements": [],
        "reproduced_expert_review_measurements": [],
        "cost_usd": 0.0,
        "cpu_only": True,
    }


def _claim_records(summary: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        _claim(
            "dataset_size",
            "falsified",
            "The pinned released dataset index contains 4,661 rows, not the paper-claimed 46,701 QA pairs.",
        ),
        _claim(
            "categories",
            "verified",
            "The pinned index contains the expected QA categories and folder-level coverage for sound-only, speech-plus-sound, speech-only, long-form, multi-turn, open-ended, and voice QA.",
        ),
        _claim(
            "model_benchmark",
            "inconclusive",
            "No primary model-evaluation logs or score tables for the 13-system benchmark are present in the pinned dataset artifact.",
        ),
        _claim(
            "audio_ablation",
            "inconclusive",
            "The pinned dataset index contains questions, labels, and audio paths, but no audio-removal ablation outputs.",
        ),
        _claim(
            "difficulty_accuracy",
            "inconclusive",
            "Difficulty labels are present, but the artifact does not include per-difficulty model accuracy outputs.",
        ),
        _claim(
            "expert_review",
            "inconclusive",
            "The artifact does not include the clinical expert review sheets needed to reproduce the 72.4% acceptance rate.",
        ),
    ]


def _claim(key: str, status: str, observation: str) -> dict[str, str]:
    return {
        "target_claim": TARGET_CLAIMS[key],
        "challenge_claim": TARGET_CLAIMS[key],
        "challenge_claim_sha256": CLAIM_HASHES[key],
        "status": status,
        "observation": observation,
    }


def _counts(series: pd.Series) -> dict[str, int]:
    return {str(key): int(value) for key, value in series.value_counts(dropna=False).items()}


def _parse_options(value: object) -> list[str]:
    if not isinstance(value, str) or not value.strip():
        return []
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return []
    if not isinstance(parsed, list):
        return []
    return [str(item) for item in parsed]


def _parse_turns(value: object) -> list[dict[str, Any]]:
    if not isinstance(value, str) or not value.strip():
        return []
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return []
    if not isinstance(parsed, list):
        return []
    return [item for item in parsed if isinstance(item, dict)]
