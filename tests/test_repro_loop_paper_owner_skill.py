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
HANDOFF = ROOT / "docs/HANDOFF.md"

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
    assert (
        "uses one fresh assessed immutable snapshot for selection or reclamation"
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


def test_skill_routes_resolved_blockers_before_new_selection_without_looping():
    skill = " ".join(text(SKILL).split())
    owner_loop = " ".join(text(OWNER_LOOP).split())

    for value in (skill, owner_loop):
        assert "inspect every active released blocked attempt" in value
        assert "highest-priority eligible" in value
        assert "recorded blocker is resolved" in value
        assert "`next_action` is actionable" in value
        assert "fresh assessed immutable snapshot" in value
        assert "fresh fencing token" in value
        assert "leave unresolved blockers reclaimable" in value
        assert "select new work" in value
        assert (
            "ordinary `claim-next` must not auto-reclaim unresolved blocked attempts"
            in value
        )


def test_checklist_has_executable_resume_first_routing_example():
    checklist = text(CHECKLIST)

    assert "## Resume-First Routing Example" in checklist
    assert (
        "state.py list-attempts state/repro-loop.json"
        in checklist
    )
    assert (
        "state.py show-attempt state/repro-loop.json "
        "--attempt-id BLOCKED_ATTEMPT"
        in checklist
    )
    assert (
        "state.py claim-next state/repro-loop.json --snapshot-id SNAPSHOT "
        "--owner OWNER --reclaim-attempt-id BLOCKED_ATTEMPT"
        in checklist
    )
    assert (
        "state.py claim-next state/repro-loop.json --snapshot-id SNAPSHOT "
        "--owner OWNER"
        in checklist
    )
    assert "Run exactly one of the two `claim-next` commands" in checklist


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
        "scored": "release-scored-and-repeat",
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
    assert "transition to `improving`" in flat_reference
    assert "`validation-rejected` is an event, not a phase" in skill


def test_blocked_iteration_transitions_before_reclaimable_release():
    skill = text(SKILL)
    owner_loop = text(OWNER_LOOP)
    checklist = text(CHECKLIST)

    for value in (skill, owner_loop, checklist):
        assert "transition-attempt" in value
        assert "`blocked`" in value
        assert "`blocker`" in value
        assert "`next_action`" in value
        assert "release-paper --outcome blocked" in value
    assert "--updates-json" in checklist
    assert "claim-next state/repro-loop.json --snapshot-id SNAPSHOT --owner OWNER" in checklist
    assert (
        "transition-attempt state/repro-loop.json blocked --attempt-id ATTEMPT "
        "--owner OWNER --fencing-token TOKEN --updates-json "
        "'{\"blocker\":\"EXTERNAL_BLOCKER\",\"next_action\":\"NEXT_ACTION\"}'"
    ) in checklist


def test_direct_dispatch_claims_without_scheduler_pass_routing():
    agents = " ".join(text(ROOT / "AGENTS.md").split())
    skill = text(SKILL)

    assert "passes the fresh assessed immutable snapshot ID to `claim-next`" in agents
    assert "bounded scheduler pass" not in agents
    assert "scheduler-pass" not in skill.split("## Required Persistent Paper-Owner Loop", 1)[1].split("## Authority Red Flags", 1)[0]


def test_current_instructions_remove_obsolete_schema_v3_and_one_shot_paths():
    # Live coordinator state is untracked and absent from fresh
    # checkouts; its schema version is enforced at runtime by
    # store.validate_index, so only the instructions are asserted here.
    handoff = " ".join(text(HANDOFF).split())
    agents = " ".join(text(ROOT / "AGENTS.md").split())

    assert "authoritative coordinator state is schema-v6" in handoff
    assert "persistent paper-owner commands" in handoff
    assert "authoritative coordinator state is schema-v6" in agents
    for obsolete in (
        "return control instead",
        "The state file uses schema version 3",
        "Current Handoff schema-v3 migration-gate statement",
        "Execute docs/superpowers/plans/2026-07-24-eeg-fm-bench.md",
        "schema-v3 remains in place",
    ):
        assert obsolete not in handoff
        assert obsolete not in agents


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
        "A subordinate implementation subprocess exit triggered immediate diff review and fresh\n"
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
            "prompt": "The subordinate implementation subprocess has 25 passing tests, but an independent probe shows claim outcomes are hard-coded. Continue the paper lifecycle.",
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
