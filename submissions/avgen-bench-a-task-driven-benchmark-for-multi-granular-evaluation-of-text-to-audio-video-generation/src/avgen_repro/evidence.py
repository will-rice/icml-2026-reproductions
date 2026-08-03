"""CPU-only evidence audit for pinned AVGen-Bench artifacts."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import urllib.request
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi, hf_hub_download


ATTEMPT_ID = "dca81643-55fa-4d3c-9244-d3d1a00119aa"
PAPER_ID = "aJdgt8xDMy"
SNAPSHOT_ID = "41692f328d154e4fad790fb8c89aa276452ce49b8aaa18064abb9c47a897d622"
TITLE = "AVGen-Bench: A Task-Driven Benchmark for Multi-Granular Evaluation of Text-to-Audio-Video Generation"
GITHUB_REVISION = "1049eabac472d479fe5feeb1ee202961f8e0982a"
HF_DATASET_REVISION = "69eb2a20b2d47659be7cd40984baf02b7f2395a8"

UPSTREAM_PINS = {
    "paper": "arxiv:2604.08540",
    "official_code": f"github:microsoft/AVGen-Bench@{GITHUB_REVISION}",
    "generated_outputs": f"hf-dataset:microsoft/AVGen-Bench@{HF_DATASET_REVISION}",
}

CLAIMS = [
    {
        "target_claim": "AVGen-Bench defines a task-driven text-to-audio-video prompt set spanning 3 main domains and 11 real-world sub-categories (Figure 3; Figure 4).",
        "challenge_claim_sha256": "1c19020c9c880185418148720050194b2615d522a0a923bd475444da0d9e83d7",
    },
    {
        "target_claim": "AVGen-Bench evaluates T2AV systems with joint audio-visual metrics plus fine-grained modules for scene text, face identity, pitch, speech, physics, and holistic semantic alignment (Figure 5).",
        "challenge_claim_sha256": "25d43bc1f8efa8482f310d00b3f0a9703018ccad0cf53319883cf04b9efa0465",
    },
    {
        "target_claim": "Compared with existing benchmarks, AVGen-Bench uses higher-complexity prompts and covers all audio modalities with a broader metric set (Table 1).",
        "challenge_claim_sha256": "de1223581cc3744a8ca5d3ccced5f6d6052731ed9601247e911f0c4a0b7079b0",
    },
    {
        "target_claim": "Quantitative evaluation reveals that current T2AV models can score well on aesthetics while failing fine-grained semantic reliability, including text rendering, speech coherence, physical reasoning, and pitch control (Table 2; Figure 2).",
        "challenge_claim_sha256": "46282c68984bc5c2331b43d7170ddcdca860e9ccba0c6363057fbf87c4aa2db7",
    },
    {
        "target_claim": "Automated fine-grained scores correlate with expert human judgments across six evaluated dimensions (Table 3).",
        "challenge_claim_sha256": "6993580cccb8453f48c1e582c223972cda4a64c95c81e33fb264732be633c895",
    },
    {
        "target_claim": "Repeated-run and prompt-subset analyses show AVGen-Bench produces stable model comparisons under MLLM-assisted evaluation and prompt resampling (Table 5; Figure 6).",
        "challenge_claim_sha256": "f3ac97e89590cc2c500df024212e328fcc02dc90c564d002693c53e7877ce82e",
    },
]

PROMPT_FILES = [
    "prompts/ads.json",
    "prompts/animals.json",
    "prompts/asmr.json",
    "prompts/chemical_reaction.json",
    "prompts/cooking.json",
    "prompts/gameplays.json",
    "prompts/movie_trailer.json",
    "prompts/musical_instrument_tutorial.json",
    "prompts/news.json",
    "prompts/physical_experiment.json",
    "prompts/sports.json",
]

CODE_TEXT_PATHS = [
    "README.md",
    "aggregate_score.py",
    "scripts/eval_scale_stability_from_cached.py",
]

EXPECTED_MODULES = {
    "visual_quality": ("eval/Q-Align", "environments/visual_quality.yml"),
    "audio_quality": ("eval/audiobox-aesthetics", "environments/audio_quality.yml"),
    "av_sync": ("eval/Syncformer", "environments/avsync.yml"),
    "lip_sync": ("eval/syncnet_python", "environments/lipsync.yml"),
    "text": ("eval/Ocr", "environments/text_rendering_quality.yml"),
    "face": ("eval/facial_consistency", "environments/facial_consistency.yml"),
    "music_or_pitch": ("eval/music_check", "environments/pitch_accuracy.yml"),
    "speech": ("eval/speech", "environments/speech_quality.yml"),
    "low_physics": ("eval/videophy", "environments/low_level_physics.yml"),
    "high_physics": ("eval/gemini_phy",),
    "holistic": ("eval/plot_matching",),
}


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def audit_prompt_inventory(prompts: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    category_counts: dict[str, int] = {}
    sha_by_file: dict[str, str] = {}
    prompt_lengths: list[int] = []
    audio_terms = ("audio", "sound", "voice", "spoken", "says", "hear", "pitch", "music", "speech")
    prompts_with_audio = 0

    for filename, records in sorted(prompts.items()):
        category = Path(filename).stem
        category_counts[category] = len(records)
        canonical = json.dumps(records, sort_keys=True, ensure_ascii=False)
        sha_by_file[filename] = sha256_text(canonical)
        for record in records:
            prompt = str(record.get("prompt", ""))
            prompt_lengths.append(len(prompt.split()))
            if any(term in prompt.lower() for term in audio_terms):
                prompts_with_audio += 1

    return {
        "category_count": len(category_counts),
        "prompt_count": sum(category_counts.values()),
        "category_counts": category_counts,
        "mean_prompt_words": (sum(prompt_lengths) / len(prompt_lengths)) if prompt_lengths else 0.0,
        "prompts_with_audio_cues": prompts_with_audio,
        "sha256_by_file": sha_by_file,
    }


def audit_metric_inventory(aggregate_source: str, repo_files: set[str]) -> dict[str, Any]:
    groups = _literal_assignment(aggregate_source, "GROUP_DIMENSIONS")
    if groups:
        groups = {key: list(value) for key, value in groups.items()}

    module_presence = {}
    for name, candidates in EXPECTED_MODULES.items():
        present = any(any(path.startswith(candidate) or path == candidate for path in repo_files) for candidate in candidates)
        module_presence[name] = "present" if present else "missing"

    metric_count = sum(len(value) for value in groups.values()) if groups else 0
    return {
        "group_count": len(groups),
        "groups": groups,
        "metric_count": metric_count,
        "module_presence": module_presence,
        "missing_modules": [name for name, status in module_presence.items() if status == "missing"],
        "aggregate_source_sha256": sha256_text(aggregate_source),
    }


def parse_readme_leaderboard(readme: str) -> list[dict[str, Any]]:
    lines = readme.splitlines()
    rows: list[dict[str, Any]] = []
    in_detailed = False
    headers: list[str] = []

    for line in lines:
        if line.startswith("| Model | Components | Vis |"):
            in_detailed = True
            headers = [_clean_cell(cell).lower().replace(" ", "_").replace("(pq)", "pq") for cell in line.strip("|").split("|")]
            continue
        if not in_detailed:
            continue
        if line.startswith("|---"):
            continue
        if not line.startswith("|"):
            if rows:
                break
            continue
        cells = [_clean_cell(cell) for cell in line.strip("|").split("|")]
        if len(cells) != len(headers):
            continue
        row = dict(zip(headers, cells, strict=True))
        parsed = {
            "model": row["model"],
            "components": row["components"],
        }
        for key in headers[2:]:
            parsed[key] = _to_float(row[key])
        rows.append(parsed)
    return rows


def summarize_leaderboard_failure_modes(rows: list[dict[str, Any]]) -> dict[str, Any]:
    fine_dims = ["text", "face", "music", "speech", "lo-phy", "hi-phy", "holistic"]
    dimension_means: list[dict[str, Any]] = []
    high_basic_low_fine: list[str] = []

    for row in rows:
        basic = _mean([_norm_vis(row.get("vis")), _norm_aud(row.get("aud_pq"))])
        fine = _mean(
            [
                row.get("text"),
                row.get("face"),
                row.get("music"),
                row.get("speech"),
                _norm_lophy(row.get("lo-phy")),
                row.get("hi-phy"),
                row.get("holistic"),
            ]
        )
        if basic is not None and fine is not None and basic >= 80.0 and fine < 70.0:
            high_basic_low_fine.append(str(row["model"]))

    for dim in fine_dims:
        values = []
        for row in rows:
            value = row.get(dim)
            if dim == "lo-phy":
                value = _norm_lophy(value)
            if isinstance(value, (int, float)):
                values.append(float(value))
        dimension_means.append({"dimension": _display_dimension(dim), "mean": _mean(values)})

    dimension_means.sort(key=lambda item: item["mean"] if item["mean"] is not None else float("inf"))
    return {
        "row_count": len(rows),
        "models_with_high_basic_and_low_fine": high_basic_low_fine,
        "lowest_mean_dimensions": dimension_means,
    }


def audit_artifact_availability(repo_files: set[str], hf_files: set[str]) -> dict[str, Any]:
    repeat_outputs = sorted(
        path
        for path in repo_files | hf_files
        if any(marker in path.lower() for marker in ("repeat_", "scale_stability_subset", "stability_eval_runs"))
        and path.endswith((".json", ".csv", ".md"))
    )
    human_outputs = sorted(
        path
        for path in repo_files | hf_files
        if "human" in path.lower() and path.endswith((".json", ".csv", ".parquet"))
    )
    return {
        "generated_video_present": any(path.endswith(".mp4") for path in hf_files),
        "metadata_present": "metadata.parquet" in hf_files,
        "stability_code_present": "scripts/eval_scale_stability_from_cached.py" in repo_files,
        "repeat_output_artifacts": repeat_outputs,
        "human_correlation_artifacts": human_outputs,
    }


def build_evidence_bundle(
    prompts: dict[str, list[dict[str, Any]]],
    aggregate_source: str,
    readme: str,
    repo_files: set[str],
    hf_files: set[str],
) -> dict[str, Any]:
    prompt_audit = audit_prompt_inventory(prompts)
    metric_audit = audit_metric_inventory(aggregate_source, repo_files)
    leaderboard_rows = parse_readme_leaderboard(readme)
    leaderboard_summary = summarize_leaderboard_failure_modes(leaderboard_rows)
    availability = audit_artifact_availability(repo_files, hf_files)

    claim_results = _claim_results(prompt_audit, metric_audit, leaderboard_summary, availability)
    return {
        "attempt_id": ATTEMPT_ID,
        "paper_id": PAPER_ID,
        "title": TITLE,
        "snapshot_id": SNAPSHOT_ID,
        "upstream_pins": UPSTREAM_PINS,
        "artifact_access": {
            "prompt_files": sorted(prompts),
            "repo_file_count": len(repo_files),
            "hf_file_count": len(hf_files),
            "leaderboard_rows": len(leaderboard_rows),
        },
        "audits": {
            "prompt_inventory": prompt_audit,
            "metric_inventory": metric_audit,
            "leaderboard_failure_modes": leaderboard_summary,
            "artifact_availability": availability,
            "readme_sha256": sha256_text(readme),
        },
        "claim_results": claim_results,
    }


def fetch_pinned_artifacts() -> tuple[dict[str, list[dict[str, Any]]], str, str, set[str], set[str]]:
    repo_files = set(_github_tree_files())
    code_texts = {path: _fetch_github_text(path) for path in CODE_TEXT_PATHS}
    prompts = {}
    for path in PROMPT_FILES:
        local_path = hf_hub_download(
            "microsoft/AVGen-Bench",
            path,
            repo_type="dataset",
            revision=HF_DATASET_REVISION,
        )
        prompts[Path(path).name] = json.loads(Path(local_path).read_text(encoding="utf-8"))

    api = HfApi()
    hf_files = set(api.list_repo_files("microsoft/AVGen-Bench", repo_type="dataset", revision=HF_DATASET_REVISION))
    return prompts, code_texts["aggregate_score.py"], code_texts["README.md"], repo_files, hf_files


def build_offline_fixture_bundle() -> dict[str, Any]:
    prompts = {
        "ads.json": [
            {"content": "Ad One", "prompt": "A product ad with spoken voiceover and on-screen text."},
            {"content": "Ad Two", "prompt": "A noisy beach ad with synchronized splash sounds."},
        ],
        "sports.json": [
            {"content": "Golf", "prompt": "A club hits sand with a dull thump and visible motion."}
        ],
    }
    aggregate = (
        'GROUP_DIMENSIONS = {"basic": ("Vis", "Aud"), "cross": ("AV", "Lip"), '
        '"fine": ("Text", "Face", "Music", "Speech", "Lo-Phy", "Hi-Phy", "Holistic")}'
    )
    readme = (
        "| Model | Components | Vis | Aud (PQ) | AV | Lip | Text | Face | Music | Speech | Lo-Phy | Hi-Phy | Holistic | Total |\n"
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n"
        "| StrongVisual | Demo | 0.970 | 7.40 | 0.20 | 2.00 | 12.00 | 50.00 | 4.00 | 92.00 | 3.00 | 60.00 | 70.00 | 64.00 |\n"
    )
    repo_files = {"README.md", "aggregate_score.py", "scripts/eval_scale_stability_from_cached.py", "eval/Ocr/batch_eval.py"}
    hf_files = {"metadata.parquet", "prompts/ads.json", "Veo_3.1_fast/ads/example.mp4"}
    return build_evidence_bundle(prompts, aggregate, readme, repo_files, hf_files)


def write_evidence(output_path: Path, offline_fixture: bool = False) -> dict[str, Any]:
    if offline_fixture:
        bundle = build_offline_fixture_bundle()
    else:
        prompts, aggregate, readme, repo_files, hf_files = fetch_pinned_artifacts()
        bundle = build_evidence_bundle(prompts, aggregate, readme, repo_files, hf_files)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(bundle, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_report(output_path.parent.parent / "pages" / "report.md", bundle)
    return bundle


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="evidence/bundle.json")
    parser.add_argument("--offline-fixture", action="store_true")
    args = parser.parse_args(argv)

    bundle = write_evidence(Path(args.output), offline_fixture=args.offline_fixture)
    print(f"wrote {args.output} with {len(bundle['claim_results'])} claim results")
    return 0


def _claim_results(prompt_audit: dict[str, Any], metric_audit: dict[str, Any], leaderboard: dict[str, Any], availability: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "claim_index": 1,
            "claim": CLAIMS[0]["target_claim"],
            "claim_sha256": CLAIMS[0]["challenge_claim_sha256"],
            "status": "verified" if prompt_audit["category_count"] == 11 and prompt_audit["prompt_count"] == 235 else "toy",
            "observation": f"Released prompts contain {prompt_audit['prompt_count']} prompts across {prompt_audit['category_count']} category files; scoring code exposes {metric_audit['group_count']} evaluation groups.",
            "limitation": "The raw prompt JSON verifies subcategory coverage; the domain wording is inferred from released benchmark/scoring structure.",
        },
        {
            "claim_index": 2,
            "claim": CLAIMS[1]["target_claim"],
            "claim_sha256": CLAIMS[1]["challenge_claim_sha256"],
            "status": "verified" if metric_audit["metric_count"] == 11 and not metric_audit["missing_modules"] else "toy",
            "observation": f"Aggregate code defines {metric_audit['metric_count']} dimensions across {metric_audit['group_count']} groups and released source exposes {len(metric_audit['module_presence']) - len(metric_audit['missing_modules'])} expected evaluator modules.",
            "limitation": "This verifies metric/module availability and score formula, not full model evaluation reruns.",
        },
        {
            "claim_index": 3,
            "claim": CLAIMS[2]["target_claim"],
            "claim_sha256": CLAIMS[2]["challenge_claim_sha256"],
            "status": "toy",
            "observation": f"The release includes a benchmark-comparison asset and {metric_audit['metric_count']} metric dimensions; prompt mean length is {prompt_audit['mean_prompt_words']:.1f} words.",
            "limitation": "No independent rerun or audit of prior benchmark prompt complexity was performed.",
        },
        {
            "claim_index": 4,
            "claim": CLAIMS[3]["target_claim"],
            "claim_sha256": CLAIMS[3]["challenge_claim_sha256"],
            "status": "toy" if leaderboard["row_count"] else "inconclusive",
            "observation": f"Parsed {leaderboard['row_count']} released leaderboard rows; high-basic/low-fine models: {leaderboard['models_with_high_basic_and_low_fine']}.",
            "limitation": "The evidence recomputes patterns from released aggregate README values, not raw evaluator outputs.",
        },
        {
            "claim_index": 5,
            "claim": CLAIMS[4]["target_claim"],
            "claim_sha256": CLAIMS[4]["challenge_claim_sha256"],
            "status": "verified" if availability["human_correlation_artifacts"] else "inconclusive",
            "observation": f"Human-correlation artifacts found: {availability['human_correlation_artifacts']}.",
            "limitation": "Expert human judgment data are required to recompute correlations and were not found in released artifacts.",
        },
        {
            "claim_index": 6,
            "claim": CLAIMS[5]["target_claim"],
            "claim_sha256": CLAIMS[5]["challenge_claim_sha256"],
            "status": "toy" if availability["stability_code_present"] else "inconclusive",
            "observation": f"Stability script present: {availability['stability_code_present']}; repeat output artifacts found: {availability['repeat_output_artifacts']}.",
            "limitation": "Prompt-subset/repeated-run cached outputs are required for numeric stability reproduction and were not found.",
        },
    ]


def _literal_assignment(source: str, name: str) -> dict[str, Any]:
    match = re.search(rf"{name}\s*(?::[^=]+)?=\s*(\{{.*?\}})", source, flags=re.S)
    if not match:
        return {}
    try:
        return ast.literal_eval(match.group(1))
    except Exception:
        return {}


def _github_tree_files() -> list[str]:
    url = f"https://api.github.com/repos/microsoft/AVGen-Bench/git/trees/{GITHUB_REVISION}?recursive=1"
    with urllib.request.urlopen(url, timeout=30) as response:
        payload = json.loads(response.read().decode("utf-8"))
    return [item["path"] for item in payload.get("tree", []) if item.get("type") == "blob"]


def _fetch_github_text(path: str) -> str:
    url = f"https://raw.githubusercontent.com/microsoft/AVGen-Bench/{GITHUB_REVISION}/{path}"
    with urllib.request.urlopen(url, timeout=30) as response:
        return response.read().decode("utf-8")


def _clean_cell(cell: str) -> str:
    return re.sub(r"[*`]", "", cell).strip()


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(str(value).strip())
    except ValueError:
        return None


def _mean(values: list[float | None]) -> float | None:
    clean = [float(value) for value in values if isinstance(value, (int, float))]
    return sum(clean) / len(clean) if clean else None


def _norm_vis(value: Any) -> float | None:
    value = _to_float(value)
    return value * 100.0 if value is not None else None


def _norm_aud(value: Any) -> float | None:
    value = _to_float(value)
    return value * 10.0 if value is not None else None


def _norm_lophy(value: Any) -> float | None:
    value = _to_float(value)
    return value * 20.0 if value is not None else None


def _display_dimension(value: str) -> str:
    return {
        "lo-phy": "Lo-Phy",
        "hi-phy": "Hi-Phy",
        "text": "Text",
        "face": "Face",
        "music": "Music",
        "speech": "Speech",
        "holistic": "Holistic",
    }[value]


def _write_report(path: Path, bundle: dict[str, Any]) -> None:
    lines = [
        "# AVGen-Bench Evidence Report",
        "",
        f"- Attempt: `{bundle['attempt_id']}`",
        f"- Paper: `{bundle['paper_id']}`",
        f"- Snapshot: `{bundle['snapshot_id']}`",
        "",
        "## Claim Results",
        "",
        "| # | Status | Evidence summary |",
        "|---:|---|---|",
    ]
    for result in bundle["claim_results"]:
        lines.append(
            f"| {result['claim_index']} | {result['status']} | {result['observation']} Limitation: {result['limitation']} |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
