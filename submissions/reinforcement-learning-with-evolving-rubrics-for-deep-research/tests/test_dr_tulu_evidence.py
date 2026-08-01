import json
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from generate_evidence import (  # noqa: E402
    CLAIM_BINDINGS,
    audit_agent_tooling,
    audit_model_cards,
    audit_rubric_buffer,
    build_evidence_bundle,
    main,
)


def write_fixture_tree(tmp_path: Path) -> dict[str, Path]:
    repo = tmp_path / "repo"
    rl = repo / "rl" / "open-instruct"
    trainer_dir = rl / "open_instruct"
    reward_dir = trainer_dir / "search_rewards" / "utils"
    agent_dir = repo / "agent" / "dr_agent"
    reward_dir.mkdir(parents=True)
    agent_dir.mkdir(parents=True)

    (rl / "train_dr_tulu.sh").write_text(
        "\n".join(
            [
                "model_name=rl-research/DR-Tulu-SFT-8B",
                "dataset_list=\"rl-research/dr-tulu-rl-data 1.0\"",
                "--apply_adaptive_rubric_reward true \\",
                "--use_rubric_buffer true \\",
                "--use_static_rubrics_as_persistent_rubrics true \\",
                "--max_active_rubrics 5 \\",
                "--tools mcp \\",
                "--mcp_tool_names 'snippet_search,google_search,browse_webpage' \\",
            ]
        ),
        encoding="utf-8",
    )
    (trainer_dir / "grpo_fast.py").write_text(
        "\n".join(
            [
                "use_rubric_buffer: bool = False",
                "max_active_rubrics: int = 5",
                "rubric_buffer[gt['query']] = {'active_rubrics': [], 'inactive_rubrics': [], 'persistent_rubrics': gt['rubrics']}",
                "rubric_key_stats[rubric_key]['std'] = rewards.std()",
                "if stats['std'] == 0: rubrics_to_deactivate.append((query, rubric))",
                "rubric_std_pairs.sort(key=lambda x: x[1], reverse=True)",
                "rubric_std_pairs[:args.max_active_rubrics]",
            ]
        ),
        encoding="utf-8",
    )
    (reward_dir / "rubric_utils.py").write_text(
        "rubric_buffer[query]['active_rubrics'].extend(new_active_rubrics)\n",
        encoding="utf-8",
    )
    (agent_dir / "client.py").write_text(
        "\n".join(
            [
                "async def _generate_with_tools_native(self):",
                "    tool_execution_tasks.append(tool(function_args))",
                "    tool_outputs = await asyncio.gather(*tool_execution_tasks, return_exceptions=True)",
                "async def _call_litellm_with_tools(self): pass",
            ]
        ),
        encoding="utf-8",
    )

    model_card = tmp_path / "DR-Tulu-8B.md"
    model_card.write_text(
        "\n".join(
            [
                "---",
                "license: apache-2.0",
                "datasets:",
                "- rl-research/dr-tulu-rl-data",
                "base_model:",
                "- rl-research/DR-Tulu-SFT-8B",
                "---",
                "This is the RL checkpoint of DR Tulu, an open deep research agent trained on top of rl-research/DR-Tulu-SFT-8B.",
                "This model has undergone RL training on this dataset.",
                "This model has been trained for tool-use using the dr-agent-lib framework.",
                "| DR-Tulu-8B | 88.3 | 52.8 | 75.7 | 45.4 | 63.7 |",
            ]
        ),
        encoding="utf-8",
    )
    no_rler_card = tmp_path / "DR-Tulu-No-RLER-8B.md"
    no_rler_card.write_text(
        "This is the RL checkpoint trained on top of rl-research/DR-Tulu-SFT-8B. This model is trained without RLER and is an ablation model.\n",
        encoding="utf-8",
    )
    data_card = tmp_path / "dr-tulu-rl-data.md"
    data_card.write_text(
        "This dataset contains the RL training data for DR Tulu, containing prompts and search-based rubrics generated from OpenScholar and SearchArena prompts.\n",
        encoding="utf-8",
    )
    return {
        "repo": repo,
        "model_card": model_card,
        "no_rler_card": no_rler_card,
        "data_card": data_card,
    }


def test_rubric_buffer_audit_detects_fixed_size_update_and_std_pruning(tmp_path):
    paths = write_fixture_tree(tmp_path)

    audit = audit_rubric_buffer(paths["repo"])

    assert audit["uses_rubric_buffer"] is True
    assert audit["max_active_rubrics"] == 5
    assert audit["appends_generated_rubrics"] is True
    assert audit["prunes_by_score_std"] is True
    assert audit["moves_inactive_rubrics"] is True


def test_agent_and_model_card_audits_detect_training_lineage_and_async_tools(tmp_path):
    paths = write_fixture_tree(tmp_path)

    tooling = audit_agent_tooling(paths["repo"])
    cards = audit_model_cards(paths["model_card"], paths["no_rler_card"], paths["data_card"])

    assert tooling["uses_async_tool_execution"] is True
    assert tooling["uses_mcp_tools"] is True
    assert cards["main_model_base"] == "rl-research/DR-Tulu-SFT-8B"
    assert cards["main_model_dataset"] == "rl-research/dr-tulu-rl-data"
    assert cards["no_rler_ablation_present"] is True
    assert cards["license"] == "apache-2.0"


def test_bundle_keeps_reported_benchmark_tables_separate_from_reproduced_measurements(tmp_path):
    paths = write_fixture_tree(tmp_path)

    bundle = build_evidence_bundle(
        repo_dir=paths["repo"],
        model_card=paths["model_card"],
        no_rler_card=paths["no_rler_card"],
        data_card=paths["data_card"],
        upstream_commit="9d7b0371c085e9311ddec483ed39768c0bd9fe99",
    )

    assert [claim["challenge_claim_sha256"] for claim in bundle["claims"]] == [
        binding["challenge_claim_sha256"] for binding in CLAIM_BINDINGS
    ]
    statuses = {claim["claim_index"]: claim["status"] for claim in bundle["claims"]}
    assert statuses[1] == "verified"
    assert statuses[2] == "verified"
    assert statuses[3] == "inconclusive"
    assert statuses[4] == "inconclusive"
    assert statuses[5] == "inconclusive"
    assert statuses[6] == "inconclusive"
    assert "reported table" in bundle["claims"][2]["evidence"].lower()
    assert bundle["provenance"]["upstream_commit"] == "9d7b0371c085e9311ddec483ed39768c0bd9fe99"


def test_cli_writes_deterministic_json_with_final_newline(tmp_path):
    paths = write_fixture_tree(tmp_path)
    out = tmp_path / "bundle.json"
    args = [
        "--repo-dir",
        str(paths["repo"]),
        "--model-card",
        str(paths["model_card"]),
        "--no-rler-card",
        str(paths["no_rler_card"]),
        "--data-card",
        str(paths["data_card"]),
        "--upstream-commit",
        "9d7b0371c085e9311ddec483ed39768c0bd9fe99",
        "--output",
        str(out),
    ]

    assert main(args) == 0
    first = out.read_bytes()
    assert first.endswith(b"\n")
    assert json.loads(first)["attempt_id"] == "cd23c17f-37ff-4fb6-bda1-edd5d13d1f98"
    assert main(args) == 0
    assert out.read_bytes() == first
