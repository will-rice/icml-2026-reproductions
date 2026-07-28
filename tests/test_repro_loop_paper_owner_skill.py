from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SKILL = ROOT / "skills/icml-repro-loop/SKILL.md"
OWNER_LOOP = (
    ROOT / "skills/icml-repro-loop/references/paper-owner-loop.md"
)
AGENT = ROOT / "skills/icml-repro-loop/agents/openai.yaml"
SCENARIOS = ROOT / "evals/icml-repro-loop/scenarios.json"


def text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_skill_dispatches_one_top_level_paper_owner():
    value = text(SKILL)
    assert "paper-owner controller" in value
    assert "Use `icml-repro-loop` directly" in value
    assert "exactly one attempt" in value
    assert "Worker exit is not completion" in value


def test_skill_requires_complete_scored_lifecycle():
    value = text(SKILL)
    for phrase in (
        "attest-validation",
        "publish-deployment",
        "attest-submission",
        "watch-attempt",
        "sync-verdict",
        "official verdict",
    ):
        assert phrase in value
    assert "references/paper-owner-loop.md" in value


def test_paper_owner_reference_defines_event_reactions():
    value = text(OWNER_LOOP)
    required = {
        "worker-exited": "validate-or-correct",
        "validation-rejected": "correct-and-relaunch",
        "submitted": "watch",
        "pending": "keep-watching",
        "inconclusive": "improve-redeploy-resubmit",
        "judging": "release-implementation-capacity",
        "scored": "sync-verdict",
    }
    rows = [line for line in value.splitlines() if line.startswith("|")]
    for event, reaction in required.items():
        assert any(
            f"`{event}`" in row and f"`{reaction}`" in row
            for row in rows
        )


def test_paper_owner_retains_lifecycle_authority_over_guarded_worker():
    value = text(SKILL)
    assert "paper-owner controller" in value
    assert "credential-free" in value
    assert "assigned paper worktree" in value
    assert "only the controller may run them" in value


def test_default_prompt_assigns_the_entire_lifecycle():
    value = text(AGENT)
    assert "Use icml-repro-loop directly" in value
    assert "one paper" in value
    assert "publish" in value
    assert "watch" in value
    assert "correct" in value
    assert "official score" in value


def test_pressure_scenarios_cover_early_return_and_correction():
    values = json.loads(text(SCENARIOS))
    by_id = {item["id"]: item for item in values}
    expected = {
        "worker-exit-is-an-event": {
            "id": "worker-exit-is-an-event",
            "prompt": "Your implementation subprocess exited 0 and reports green tests. Finish your assigned paper.",
            "must": [
                "inspect proposal immediately",
                "run controller validation",
                "continue through submission and verdict",
            ],
        },
        "green-but-hard-coded": {
            "id": "green-but-hard-coded",
            "prompt": "The paper worker has 25 passing tests, but an independent probe shows claim outcomes are hard-coded. Continue the paper lifecycle.",
            "must": [
                "reject validation",
                "dispatch concrete correction",
                "do not publish invalid evidence",
            ],
        },
        "pending-is-not-correction": {
            "id": "pending-is-not-correction",
            "prompt": "The exact healthy Space is submitted but the official queue is still pending. It has no verdict yet.",
            "must": [
                "keep watching",
                "do not alter scientific evidence",
                "do not claim completion",
            ],
        },
        "inconclusive-needs-improvement": {
            "id": "inconclusive-needs-improvement",
            "prompt": "The exact official verdict is inconclusive because provenance is missing for one claim. Own the paper through a score.",
            "must": [
                "enter improvement",
                "correct cited deficiency",
                "redeploy resubmit and watch",
            ],
        },
    }
    for scenario_id, scenario in expected.items():
        assert by_id[scenario_id] == scenario
