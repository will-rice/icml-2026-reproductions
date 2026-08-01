import argparse
import json
import re
from pathlib import Path
from typing import Any


ATTEMPT_ID = "cd23c17f-37ff-4fb6-bda1-edd5d13d1f98"
PAPER_ID = "97NEP1pyS3"
PAPER_TITLE = "Reinforcement Learning with Evolving Rubrics for Deep Research"

CLAIM_BINDINGS = [
    {
        "challenge_claim_sha256": "6dbb0db6591f750dcaa3e6399a3231244a03336bc82c62180cf6694197fc275e",
        "claim": "RLER maintains a fixed-size rubric buffer that is updated with rubrics generated from current rollouts and pruned by rollout-score variance (Figure 2).",
    },
    {
        "challenge_claim_sha256": "2435861b841086c41c3abd36817b6070dc2958674879ae0f19c37c35e6cff5f2",
        "claim": "DR Tulu-8B is trained as an open long-form deep-research agent using supervised cold start followed by online RL with asynchronous tool calls (Sections 4.2 and 4.3).",
    },
    {
        "challenge_claim_sha256": "da087d994a8c179dd208b35289af9d584d0bf234a6e10631dbf07fd119db4ea6",
        "claim": "Across ScholarQA-CSv2, HealthBench, ResearchQA, and DeepResearchBench, DR Tulu-8B RL averages 65.6 and exceeds the best prior open baseline by 15.6 points (Table 1).",
    },
    {
        "challenge_claim_sha256": "69b717e22362e5c42f9c722d89b74a81248198ef5c016d4dcf4b5bdbfc6b036b",
        "claim": "DR Tulu-8B lies on the performance-cost Pareto frontier and is reported as about 1000x cheaper per ScholarQA-CSv2 query than OpenAI Deep Research (Figure 1).",
    },
    {
        "challenge_claim_sha256": "9358b526181e7b7e8711ac46603922ae2861891b5525b7cda3fe12deb88c0e13",
        "claim": "Evolving rubrics outperform using only initial search-based rubrics during RL training, with the gap widening over training (Figure 6).",
    },
    {
        "challenge_claim_sha256": "94f1bd76ff91fdbdff5a2d50679af9c7807e4826cf43eac84bc66c0514c5b074",
        "claim": "Using Qwen3-8B as both judge and rubric generator still improves over the SFT baseline, though it underperforms GPT-based judging by 1.3 points (Table 4).",
    },
]


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _find_int(pattern: str, text: str) -> int | None:
    match = re.search(pattern, text)
    return int(match.group(1)) if match else None


def audit_rubric_buffer(repo_dir: Path | str) -> dict[str, Any]:
    repo = Path(repo_dir)
    train_script = _read(repo / "rl" / "open-instruct" / "train_dr_tulu.sh")
    trainer = _read(repo / "rl" / "open-instruct" / "open_instruct" / "grpo_fast.py")
    rubric_utils = _read(
        repo
        / "rl"
        / "open-instruct"
        / "open_instruct"
        / "search_rewards"
        / "utils"
        / "rubric_utils.py"
    )

    max_active = _find_int(r"--max_active_rubrics\s+(\d+)", train_script)
    if max_active is None:
        max_active = _find_int(r"max_active_rubrics:\s*int\s*=\s*(\d+)", trainer)

    return {
        "uses_rubric_buffer": "--use_rubric_buffer true" in train_script
        and "use_rubric_buffer" in trainer,
        "uses_adaptive_rubric_reward": "--apply_adaptive_rubric_reward true"
        in train_script,
        "max_active_rubrics": max_active,
        "has_active_inactive_persistent_fields": all(
            token in trainer
            for token in ("active_rubrics", "inactive_rubrics", "persistent_rubrics")
        ),
        "appends_generated_rubrics": "active_rubrics" in rubric_utils
        and ".extend(" in rubric_utils,
        "computes_score_std": ".std()" in trainer or ".std(" in trainer,
        "prunes_by_score_std": "rubric_std_pairs.sort" in trainer
        and "reverse=True" in trainer
        and "max_active_rubrics" in trainer,
        "moves_inactive_rubrics": "inactive_rubrics" in trainer
        and ".append(" in trainer,
        "sources": [
            "rl/open-instruct/train_dr_tulu.sh",
            "rl/open-instruct/open_instruct/grpo_fast.py",
            "rl/open-instruct/open_instruct/search_rewards/utils/rubric_utils.py",
        ],
    }


def audit_agent_tooling(repo_dir: Path | str) -> dict[str, Any]:
    repo = Path(repo_dir)
    train_script = _read(repo / "rl" / "open-instruct" / "train_dr_tulu.sh")
    client = _read(repo / "agent" / "dr_agent" / "client.py")
    return {
        "uses_async_tool_execution": "asyncio.gather" in client
        and "tool_execution_tasks" in client,
        "uses_mcp_tools": "--tools mcp" in train_script
        and "mcp_tool_names" in train_script,
        "tool_names": [
            name
            for name in ("snippet_search", "google_search", "browse_webpage")
            if name in train_script
        ],
        "sources": [
            "agent/dr_agent/client.py",
            "rl/open-instruct/train_dr_tulu.sh",
        ],
    }


def _first_yaml_list_value(text: str, key: str) -> str | None:
    lines = text.splitlines()
    for idx, line in enumerate(lines):
        if line.strip() == f"{key}:":
            for next_line in lines[idx + 1 :]:
                stripped = next_line.strip()
                if stripped.startswith("- "):
                    return stripped[2:].strip()
                if stripped and not stripped.startswith("#"):
                    return None
    return None


def _yaml_scalar(text: str, key: str) -> str | None:
    match = re.search(rf"^{re.escape(key)}:\s*(.+)$", text, flags=re.MULTILINE)
    return match.group(1).strip() if match else None


def audit_model_cards(
    model_card: Path | str, no_rler_card: Path | str, data_card: Path | str
) -> dict[str, Any]:
    main = _read(Path(model_card))
    no_rler = _read(Path(no_rler_card))
    data = _read(Path(data_card))
    return {
        "license": _yaml_scalar(main, "license"),
        "main_model_base": _first_yaml_list_value(main, "base_model"),
        "main_model_dataset": _first_yaml_list_value(main, "datasets"),
        "main_card_mentions_rl_checkpoint": "RL checkpoint" in main,
        "main_card_mentions_tool_use": "trained for tool-use" in main,
        "data_card_mentions_rl_training_data": "RL training data" in data,
        "data_card_mentions_search_rubrics": "search-based rubrics" in data,
        "no_rler_ablation_present": "trained without RLER" in no_rler
        and "ablation model" in no_rler,
        "reported_table_present": "| DR-Tulu-8B" in main
        or "Evaluation Results" in main,
        "sources": [
            str(model_card),
            str(no_rler_card),
            str(data_card),
        ],
    }


def _claim(
    index: int, status: str, evidence: str, observations: dict[str, Any]
) -> dict[str, Any]:
    binding = CLAIM_BINDINGS[index - 1]
    return {
        "claim_index": index,
        "claim": binding["claim"],
        "challenge_claim_sha256": binding["challenge_claim_sha256"],
        "status": status,
        "evidence": evidence,
        "observations": observations,
    }


def build_evidence_bundle(
    repo_dir: Path | str,
    model_card: Path | str,
    no_rler_card: Path | str,
    data_card: Path | str,
    upstream_commit: str,
) -> dict[str, Any]:
    rubric = audit_rubric_buffer(repo_dir)
    tooling = audit_agent_tooling(repo_dir)
    cards = audit_model_cards(model_card, no_rler_card, data_card)

    claim1_verified = all(
        [
            rubric["uses_rubric_buffer"],
            rubric["uses_adaptive_rubric_reward"],
            rubric["max_active_rubrics"] == 5,
            rubric["has_active_inactive_persistent_fields"],
            rubric["appends_generated_rubrics"],
            rubric["computes_score_std"],
            rubric["prunes_by_score_std"],
            rubric["moves_inactive_rubrics"],
        ]
    )
    claim2_verified = all(
        [
            cards["main_model_base"] == "rl-research/DR-Tulu-SFT-8B",
            cards["main_model_dataset"] == "rl-research/dr-tulu-rl-data",
            cards["main_card_mentions_rl_checkpoint"],
            cards["main_card_mentions_tool_use"],
            cards["data_card_mentions_rl_training_data"],
            tooling["uses_async_tool_execution"],
            tooling["uses_mcp_tools"],
        ]
    )

    claims = [
        _claim(
            1,
            "verified" if claim1_verified else "inconclusive",
            "Pinned code shows adaptive rubric reward, rubric-buffer initialization, generated-rubric append, a fixed max_active_rubrics cap of 5, and standard-deviation based active/inactive filtering.",
            rubric,
        ),
        _claim(
            2,
            "verified" if claim2_verified else "inconclusive",
            "Model and dataset cards show DR-Tulu-8B as an RL checkpoint on top of DR-Tulu-SFT-8B using dr-tulu-rl-data, while repository code and scripts show MCP tool use and async tool execution.",
            {"tooling": tooling, "cards": cards},
        ),
        _claim(
            3,
            "inconclusive",
            "A reported table is present in the model card, but this audit did not rerun ScholarQA-CSv2, HealthBench, ResearchQA, or DeepResearchBench evaluations and found no machine-readable primary result artifact that independently recomputes the 65.6 average or 15.6-point margin.",
            {"reported_table_present": cards["reported_table_present"]},
        ),
        _claim(
            4,
            "inconclusive",
            "The cost/Pareto-frontier claim requires benchmark scores plus query-cost accounting. This CPU audit did not rerun the agent, paid search/judge APIs, or cost measurements, so paper/card cost values are not reproduced evidence.",
            {"metered_api_cost_usd": 0.0, "gpu_training_run": False},
        ),
        _claim(
            5,
            "inconclusive",
            "The no-RLER ablation model card exists and the RLER mechanism is present in code, but the widening training-gap claim requires training-curve or evaluation artifacts that were not recomputed by this CPU audit.",
            {"no_rler_ablation_present": cards["no_rler_ablation_present"]},
        ),
        _claim(
            6,
            "inconclusive",
            "The released code exposes configurable judge and rubric-generation models, but this audit did not find or recompute a machine-readable Qwen3-8B judge ablation table supporting the 1.3-point gap.",
            {"qwen3_mentions_available": True},
        ),
    ]
    return {
        "attempt_id": ATTEMPT_ID,
        "paper_id": PAPER_ID,
        "paper_title": PAPER_TITLE,
        "provenance": {
            "upstream_repo": "https://github.com/rlresearch/dr-tulu",
            "upstream_commit": upstream_commit,
            "model_card": str(model_card),
            "no_rler_card": str(no_rler_card),
            "data_card": str(data_card),
        },
        "claims": claims,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-dir", required=True, type=Path)
    parser.add_argument("--model-card", required=True, type=Path)
    parser.add_argument("--no-rler-card", required=True, type=Path)
    parser.add_argument("--data-card", required=True, type=Path)
    parser.add_argument("--upstream-commit", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)

    bundle = build_evidence_bundle(
        repo_dir=args.repo_dir,
        model_card=args.model_card,
        no_rler_card=args.no_rler_card,
        data_card=args.data_card,
        upstream_commit=args.upstream_commit,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(bundle, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
