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
    for event, reaction in required.items():
        assert f"`{event}`" in value
        assert f"`{reaction}`" in value


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
    assert {
        "worker-exit-is-an-event",
        "green-but-hard-coded",
        "pending-is-not-correction",
        "inconclusive-needs-improvement",
    } <= by_id.keys()
