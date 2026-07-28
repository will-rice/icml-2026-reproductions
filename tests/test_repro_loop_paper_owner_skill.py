from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SKILL = ROOT / "skills/icml-repro-loop/SKILL.md"
OWNER_LOOP = (
    ROOT / "skills/icml-repro-loop/references/paper-owner-loop.md"
)
AGENT = ROOT / "skills/icml-repro-loop/agents/openai.yaml"
CHECKLIST = ROOT / "skills/icml-repro-loop/references/submission-checklist.md"
SCENARIOS = ROOT / "evals/icml-repro-loop/scenarios.json"

EXPECTED_DEFAULT_PROMPT = (
    "Use icml-repro-loop directly and keep running its paper-owner loop."
)


def text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_skill_dispatches_one_persistent_controller_capable_paper_owner():
    value = text(SKILL)
    assert "persistent paper-owner worker" in value
    assert "trusted controller" in value
    assert "one current paper at a time" in value
    assert "repeat" in value
    assert "implementation subprocess" in value
    assert "subprocess is not the dispatched worker" in value


def test_skill_repeats_only_after_score_or_recoverable_blocker():
    skill = " ".join(text(SKILL).split())
    owner_loop = " ".join(text(OWNER_LOOP).split())

    assert "exact official verdict" in skill
    assert "select the next paper" in skill
    assert "remain dedicated" in skill
    assert "submitted or judging" in skill
    assert "release the blocked attempt" in skill
    assert "same or another worker" in owner_loop
    assert "fresh fencing token" in owner_loop
    assert "next-paper selection uses a fresh assessed immutable snapshot" in owner_loop
    assert (
        "blocked-attempt reclamation uses a fresh assessed immutable snapshot"
        in owner_loop
    )


def test_skill_gives_controller_credentials_only_to_paper_owner():
    skill = " ".join(text(SKILL).split())

    assert "paper-owner worker may publish" in skill
    assert "controller credentials" in skill
    assert (
        "controller credentials never enter Git, evidence, logs, or subordinate "
        "environments"
    ) in skill
    assert "subordinate implementation subprocess" in skill
    assert "credential-free" in skill


def test_checklist_requires_release_event_before_next_iteration():
    checklist = text(CHECKLIST)

    assert "release-paper" in checklist
    assert "paper-owner-released" in checklist
    assert "must not select while `submitted` or `judging`" in checklist
    assert "blocked attempt remains reclaimable" in checklist


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
    flat_value = " ".join(value.split())
    required = {
        "worker-exited": "validate-or-correct",
        "validation-rejected": "correct-and-relaunch",
        "submitted": "remain-dedicated",
        "pending": "keep-watching",
        "inconclusive": "improve-redeploy-resubmit",
        "judging": "remain-dedicated",
        "scored": "sync-verdict",
    }
    rows = [line for line in value.splitlines() if line.startswith("|")]
    for event, reaction in required.items():
        assert any(
            f"`{event}`" in row and f"`{reaction}`" in row
            for row in rows
        )
    assert "submitted/judging are dedicated states and do not release" in flat_value
    assert "watch; do not select another paper" in flat_value
    assert "release after exact `sync-verdict`" in flat_value
    assert "persist, release reclaimably, then `claim-next`" in flat_value


def test_paper_owner_contract_separates_validation_event_from_phase_and_verdict_improvement():
    skill = text(SKILL)
    reference = text(OWNER_LOOP)
    flat_reference = " ".join(reference.split())

    assert "`validation-rejected` is an event" in reference
    assert "not a phase" in reference
    assert "attempt remains `implementing`" in reference
    assert "correction contract" in reference
    assert "normal fenced `run-worker`" in reference
    assert "There is no `--work-kind` flag" in flat_reference
    assert "derives `implementation` or `correction` from the attempt phase" in flat_reference
    assert "`sync-verdict --improvement-reason REASON`" in flat_reference
    assert "enters `improving`" in flat_reference
    assert "`validation-rejected` is an event, not a phase" in skill


def test_paper_owner_retains_lifecycle_authority_over_guarded_worker():
    value = text(SKILL)
    assert "paper-owner controller" in value
    assert "credential-free" in value
    assert "assigned paper worktree" in value
    assert "only the controller may run them" in value


def test_default_prompt_assigns_the_entire_lifecycle():
    value = text(AGENT)
    prompt_line = next(
        line.strip() for line in value.splitlines()
        if line.strip().startswith("default_prompt:")
    )
    assert prompt_line == f'default_prompt: "{EXPECTED_DEFAULT_PROMPT}"'


def test_completion_gate_requires_all_paper_owner_outcomes():
    value = text(CHECKLIST)
    expected_bullets = (
        "The directly dispatched persistent paper-owner worker owns one current\n"
        "  paper at a time and repeats only after exact verdict import or a\n"
        "  genuine persisted blocker.",
        "An implementation-worker exit triggered immediate diff review and fresh\n"
        "  controller validation without a user status prompt.",
        "A rejected validation produced exact correction findings and a guarded\n"
        "  relaunch on the same attempt.",
        "The paper owner continued through `publish-deployment`,\n"
        "  `attest-submission`, `watch-attempt`, and `sync-verdict`.",
        "Pending queue state was watched rather than treated as evidence failure.",
        "`submitted` and `judging` remained dedicated to their paper and did not\n"
        "  free owner capacity; release occurred only after exact verdict import\n"
        "  or a genuine persisted blocker.",
    )
    assert "## Paper-Owner Completion Gate" in value
    for bullet in expected_bullets:
        assert f"- [ ] {bullet}" in value


def test_completion_gate_preserves_ordered_event_handoff_without_status_waits():
    value = text(CHECKLIST)
    expected_handoff = """```text
run-worker
  -> inspect worker-exited telemetry
  -> attest-validation OR correction run-worker
  -> publish-deployment
  -> refresh-live + attest-submission
  -> watch-attempt + record-poll
  -> improvement loop OR sync-verdict
```"""
    assert expected_handoff in value
    assert "No arrow in this handoff is driven by a user status question." in value


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
        "score-then-repeat": {
            "id": "score-then-repeat",
            "prompt": "Your exact official verdict was imported successfully. Continue running icml-repro-loop.",
            "must": [
                "release scored paper",
                "select next eligible paper",
                "retain exact verdict history",
            ],
        },
        "judging-remains-dedicated": {
            "id": "judging-remains-dedicated",
            "prompt": "Your paper is judging and another high-value paper is unclaimed. Continue the loop.",
            "must": [
                "remain dedicated to judging paper",
                "do not select another paper",
                "keep bounded verdict watch",
            ],
        },
        "block-release-reclaim": {
            "id": "block-release-reclaim",
            "prompt": "An external blocker prevents useful progress on the current paper. Continue autonomously.",
            "must": [
                "persist and notify blocker",
                "release without abandoning attempt",
                "allow later fenced reclamation",
                "select another paper",
            ],
        },
    }
    for scenario_id, scenario in expected.items():
        assert by_id[scenario_id] == scenario
