from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path


EXPECTED_COMMIT = "1d244f5dca42944b67a379b44bfeb9f5748f189d"
PAPER_ID = "OC2z7iSQKa"
PAPER_TITLE = "$\\tau^2$-Bench: Evaluating Conversational Agents in a Dual-Control Environment"
UPSTREAM_REPO = "https://github.com/sierra-research/tau2-bench.git"

SOURCE_FILES = [
    "LICENSE",
    "src/tau2/domains/telecom/environment.py",
    "src/tau2/domains/telecom/tools.py",
    "src/tau2/domains/telecom/user_tools.py",
    "src/tau2/domains/telecom/tasks/create_tasks.py",
    "data/tau2/domains/telecom/split_tasks.json",
    "data/tau2/domains/telecom/tasks.json",
    "data/tau2/domains/telecom/tasks_full.json",
]


def resolve_upstream_root(project_root: Path) -> Path:
    vendor_root = project_root / "vendor" / "tau2-bench"
    if vendor_root.exists():
        return vendor_root

    cache_root = Path("/tmp") / "icml-tau2-bench-upstream" / EXPECTED_COMMIT
    upstream_root = cache_root / "tau2-bench"
    if not upstream_root.exists():
        cache_root.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["git", "clone", UPSTREAM_REPO, str(upstream_root)],
            check=True,
        )
    subprocess.run(
        ["git", "-C", str(upstream_root), "checkout", "--detach", EXPECTED_COMMIT],
        check=True,
        stdout=subprocess.DEVNULL,
    )
    return upstream_root


def build_evidence(upstream_root: Path) -> dict:
    upstream_root = upstream_root.resolve()
    commit = _git_commit(upstream_root)
    source_hashes = {
        relative_path: _sha256(upstream_root / relative_path)
        for relative_path in SOURCE_FILES
    }
    split_tasks = _load_json(upstream_root / "data/tau2/domains/telecom/split_tasks.json")
    tasks_full = _load_json(upstream_root / "data/tau2/domains/telecom/tasks_full.json")

    agent_tool_count = _tool_count(upstream_root / "src/tau2/domains/telecom/tools.py")
    user_tool_count = _tool_count(upstream_root / "src/tau2/domains/telecom/user_tools.py")
    base_task_count = len(split_tasks["base"])
    full_task_count = len(tasks_full)

    environment_source = _read_text(
        upstream_root / "src/tau2/domains/telecom/environment.py"
    )
    generator_source = _read_text(
        upstream_root / "src/tau2/domains/telecom/tasks/create_tasks.py"
    )

    dual_control_observations = {
        "has_agent_tools": "TelecomTools" in environment_source,
        "has_user_tools": "TelecomUserTools" in environment_source,
        "has_agent_db": "TelecomDB" in environment_source,
        "has_user_db": "TelecomUserDB" in environment_source,
        "syncs_shared_state": all(
            token in environment_source
            for token in [
                "self.user_tools.db.surroundings",
                "self.tools._get_line_by_phone",
                "roaming_allowed",
                "_set_bill_to_paid",
            ]
        ),
    }
    generator_observations = {
        "uses_intent_task_managers": all(
            token in generator_source
            for token in [
                "mobile_data_task_manager.create_tasks",
                "service_issues_task_manager.create_tasks",
                "mms_issue_task_manager.create_tasks",
            ]
        ),
        "bins_by_intent_subtasks_persona": "tasks_by_bins[(task[\"intent\"], task[\"num_subtasks\"], task[\"persona\"])]" in generator_source,
        "serializes_task_outputs": "json.dump([t.model_dump() for t in sampled_tasks]" in generator_source,
    }

    return {
        "paper_id": PAPER_ID,
        "paper_title": PAPER_TITLE,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "upstream": {
            "revision": f"arxiv:2506.07982+github:sierra-research/tau2-bench@{EXPECTED_COMMIT}",
            "commit": commit,
            "commit_matches_expected": commit == EXPECTED_COMMIT,
            "license": "MIT",
            "source_hashes": source_hashes,
        },
        "claims": [
            {
                "claim_id": "dual_control_shared_state",
                "challenge_claim_sha256": "0199b3b43b308ce8469189f64e2310b12cb869a8c6255975c3c4cb7e9093f78a",
                "claim": "tau2-bench introduces a telecom dual-control environment where both the conversational agent and simulated user can act with tools in a shared dynamic state (Figure 1).",
                "status": "verified" if all(dual_control_observations.values()) else "unverified",
                "observations": dual_control_observations,
                "provenance": "Static audit of pinned TelecomEnvironment source.",
            },
            {
                "claim_id": "telecom_artifact_counts",
                "challenge_claim_sha256": "36d94de446993da42bb35022e284615dd122232225cf76fb0d3ccf26116e2788",
                "claim": "The telecom domain is modeled with separate agent and user databases, 13 agent tools, 30 user tools, and 114 sampled tasks from a 2,285-task full set (Table 1).",
                "status": "verified"
                if {
                    "agent_tool_count": agent_tool_count,
                    "user_tool_count": user_tool_count,
                    "base_task_count": base_task_count,
                    "full_task_count": full_task_count,
                }
                == {
                    "agent_tool_count": 13,
                    "user_tool_count": 30,
                    "base_task_count": 114,
                    "full_task_count": 2285,
                }
                else "unverified",
                "observations": {
                    "agent_tool_count": agent_tool_count,
                    "user_tool_count": user_tool_count,
                    "base_task_count": base_task_count,
                    "full_task_count": full_task_count,
                },
                "provenance": "Decorator counts and JSON split/full task lengths from pinned source.",
            },
            {
                "claim_id": "compositional_task_generator",
                "challenge_claim_sha256": "9e672b1894ac3fb6b00f8fa2a33a5d31355aa5663481e563c0594d337f0356b5",
                "claim": "The benchmark uses a compositional task generator that creates verifiable telecom tasks from intent and subtask components (Section 3.2).",
                "status": "verified" if all(generator_observations.values()) else "unverified",
                "observations": generator_observations,
                "provenance": "Static audit of pinned telecom task generator source.",
            },
        ],
    }


def write_evidence(upstream_root: Path, output_path: Path) -> dict:
    evidence = build_evidence(upstream_root)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
    return evidence


def _git_commit(path: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _load_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tool_count(path: Path) -> int:
    return _read_text(path).count("@is_tool")
