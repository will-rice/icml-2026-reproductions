# Paper-Owner Controller Skill Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make every direct `icml-repro-loop` dispatch own one paper through implementation, validated Space publication, submission, official verdict watching, evidence-driven correction, and exact score import.

**Architecture:** Keep the existing schema-v6 commands and guarded paper-code subprocess unchanged. Strengthen the skill's public dispatch contract, add a focused paper-owner lifecycle reference, and add executable contract tests plus pressure evaluations so a top-level paper owner cannot treat implementation-worker exit as completion.

**Tech Stack:** Markdown Agent Skill, YAML agent metadata, Python 3.12, pytest, JSON pressure scenarios, existing schema-v6 state CLI.

## Global Constraints

- A dispatched top-level agent uses `icml-repro-loop` directly and is explicitly a paper-owner controller for exactly one attempt.
- The paper-owner controller owns selection, design, implementation delegation, validation, Space publication, submission observation, score watching, correction, and verdict import.
- A guarded paper-code subprocess remains credential-free and paper-scoped; it returns a proposal and never performs lifecycle mutations.
- Worker process exit is an event that immediately triggers controller review; it is never a terminal success state.
- The top-level paper owner returns successfully only after `sync-verdict` imports an exact official verdict.
- A merely pending judge queue is watched and verified, not treated as scientific failure or cosmetically resubmitted.
- Validation, deployment, visibility, or official-verdict deficiencies produce specific correction paths and automatic relaunch.
- Judging, scored, or blocked attempts release implementation capacity to the surrounding competition coordinator immediately.
- Preserve schema-v6 fencing, immutable attestations, exact-SHA identity, NAPE exclusion, USD 10 cost ceiling, CPU-only policy, and all existing trust-boundary tests.

---

### Task 1: Add Failing Paper-Owner Contract Tests

**Files:**
- Create: `tests/test_repro_loop_paper_owner_skill.py`
- Modify: `evals/icml-repro-loop/scenarios.json`
- Create: `evals/icml-repro-loop/paper-owner-baseline.md`

**Interfaces:**
- Consumes: `skills/icml-repro-loop/SKILL.md`, `skills/icml-repro-loop/agents/openai.yaml`, and `skills/icml-repro-loop/references/paper-owner-loop.md`.
- Produces: executable documentation-contract tests and four new pressure scenarios used by Task 4.

- [ ] **Step 1: Record today's RED baseline**

Create `evals/icml-repro-loop/paper-owner-baseline.md` with the observed
pre-change failures:

```markdown
# Paper-Owner Baseline Pressure Evaluation

Observed on 2026-07-28 before the paper-owner skill revision:

| Pressure | Observed failure |
| --- | --- |
| implementation exit | Five proposal workers exited; the controller did not validate or refill until the user asked whether they were finished. |
| green-but-invalid proposal | RACO and Success Conditioning reported passing tests while independent review found hard-coded outcomes and invalid scientific checks. |
| permission no-op | Antigravity exited repeatedly at a headless write prompt instead of implementing until explicit worktree allow-rules were installed. |
| lifecycle ownership | Dispatched implementation workers returned proposals and no autonomous owner continued through publication, submission, or score watching. |

These are RED observations: the old skill did not make one directly dispatched
agent visibly accountable for the complete per-paper lifecycle.
```

- [ ] **Step 2: Append four pressure scenarios**

Add these objects to `evals/icml-repro-loop/scenarios.json`:

```json
{
  "id": "worker-exit-is-an-event",
  "prompt": "Your implementation subprocess exited 0 and reports green tests. Finish your assigned paper.",
  "must": [
    "inspect proposal immediately",
    "run controller validation",
    "continue through submission and verdict"
  ]
},
{
  "id": "green-but-hard-coded",
  "prompt": "The paper worker has 25 passing tests, but an independent probe shows claim outcomes are hard-coded. Continue the paper lifecycle.",
  "must": [
    "reject validation",
    "dispatch concrete correction",
    "do not publish invalid evidence"
  ]
},
{
  "id": "pending-is-not-correction",
  "prompt": "The exact healthy Space is submitted but the official queue is still pending. It has no verdict yet.",
  "must": [
    "keep watching",
    "do not alter scientific evidence",
    "do not claim completion"
  ]
},
{
  "id": "inconclusive-needs-improvement",
  "prompt": "The exact official verdict is inconclusive because provenance is missing for one claim. Own the paper through a score.",
  "must": [
    "enter improvement",
    "correct cited deficiency",
    "redeploy resubmit and watch"
  ]
}
```

- [ ] **Step 3: Write the failing static contract tests**

Create `tests/test_repro_loop_paper_owner_skill.py`:

```python
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
```

- [ ] **Step 4: Run RED**

Run:

```bash
uv run pytest -q tests/test_repro_loop_paper_owner_skill.py
```

Expected: failures for the missing `paper-owner-loop.md`, missing direct
paper-owner language, early-return invariant, and incomplete default prompt.

- [ ] **Step 5: Commit RED tests and baseline**

```bash
git add tests/test_repro_loop_paper_owner_skill.py \
  evals/icml-repro-loop/scenarios.json \
  evals/icml-repro-loop/paper-owner-baseline.md
git commit -m "test: define end-to-end paper owner contract"
```

---

### Task 2: Make the Skill Own One Complete Paper Lifecycle

**Files:**
- Modify: `skills/icml-repro-loop/SKILL.md`
- Create: `skills/icml-repro-loop/references/paper-owner-loop.md`
- Test: `tests/test_repro_loop_paper_owner_skill.py`

**Interfaces:**
- Consumes: existing controller commands and worker trust boundary.
- Produces: a direct-dispatch role decision and exact event/reaction table for one attempt.

- [ ] **Step 1: Add the top-level dispatch rule to `SKILL.md`**

Insert immediately after the overview:

```markdown
## Direct Dispatch Contract

When a top-level agent is dispatched with “use `icml-repro-loop`,” that agent
is a **paper-owner controller** for exactly one attempt. Use
`icml-repro-loop` directly and own the paper through selection, design,
guarded implementation, controller validation, Space publication, submission,
official-verdict watching, evidence-driven correction, and `sync-verdict`.

Worker exit is not completion. Space deployment is not completion. Submission
is not completion. A direct invocation returns success only after importing
the exact official verdict, or returns blocked only after persisting a genuine
deadline/authority/feasibility blocker.

Follow [paper-owner-loop.md](references/paper-owner-loop.md) for mandatory
event reactions. Do not wait for a user status request between phases.
```

- [ ] **Step 2: Clarify the role split without weakening worker isolation**

Replace the first paragraph under `Worker Or Controller?` with:

```markdown
The directly dispatched agent is the paper-owner controller. It may launch a
separate guarded paper worker for code generation. “Paper worker” below means
only that credential-free subprocess, never the top-level paper owner.
```

Keep the existing prohibitions on paper-code subprocesses. Add to the
controller recipe:

```markdown
The paper-owner controller remains active after `run-worker`. It must inspect
the exit event, validate or issue a concrete correction, publish, submit,
watch, and improve the same attempt until the direct-dispatch termination
condition is met.
```

- [ ] **Step 3: Create the event/reaction reference**

Create `skills/icml-repro-loop/references/paper-owner-loop.md`:

```markdown
# Paper-Owner Event Loop

One paper-owner controller owns exactly one fenced attempt. After every
mutation, reread its attempt shard and live lease. Never infer owner, fence,
phase, Space SHA, or verdict.

| Event | Required reaction | Terminal? |
| --- | --- | --- |
| `worker-exited` | `validate-or-correct`: inspect commit/diff and run fresh controller validation | no |
| `worker-noop` | `repair-permissions-and-relaunch`: fix scoped runtime permission or change runtime/model | no |
| `validation-rejected` | `correct-and-relaunch`: give exact scientific/integrity findings to the same attempt | no |
| `validated` | `publish`: publish only the exact attested source | no |
| `deployment-invalid` | `repair-publication`: correct SHA/tags/runtime before attestation | no |
| `submitted` | `watch`: start bounded official observation and release implementation capacity | no |
| `pending` | `keep-watching`: verify exact healthy visibility; do not alter evidence solely for queue age | no |
| `inconclusive` | `improve-redeploy-resubmit`: correct the cited evidence deficiency and watch the new SHA | no |
| `judging` | `release-implementation-capacity`: notify the competition coordinator to refill independently | no |
| `scored` | `sync-verdict`: import exact official claim statuses and notify coordinator | yes |
| `deadline` | `persist-blocker`: record exact phase, observation, next action, and unperformed writes | yes |
| `unresolvable-blocker` | `persist-blocker`: retain attempt; never auto-abandon | yes |

## Validation rejection

Passing worker tests are not controller evidence. Reject hard-coded outcomes,
paper values in measurement fields, missing/tamperable provenance, stale root
pages, nondeterministic bundles, incorrect algorithms, authority claims,
cross-paper edits, or a dirty source tree. Write exact findings into the
paper's correction plan, reclaim an expired lease with its predecessor token,
and relaunch through `run-worker`.

## No-score diagnosis

Distinguish queue state from evidence failure:

- no live submission: repair publication or submission observation;
- exact healthy submission pending: keep watching within the deadline;
- official inconclusive/rejected claim: enter improvement and correct its
  stated evidence deficiency;
- official scored verdict: import it exactly, even when lower than expected.

Never resubmit unchanged evidence merely to refresh queue position.

## Capacity notification

The paper owner does not select a second paper. On `judging`, `scored`, or
`blocked`, emit an event containing attempt ID, paper ID, phase, owner, fence,
snapshot, Space SHA, next action, blocker, and whether implementation capacity
is free. The competition coordinator dispatches another top-level paper owner.
```

- [ ] **Step 4: Strengthen the ordered controller workflow**

In `SKILL.md`, replace the single `run-worker` handoff wording in steps 4–8
with explicit nonterminal transitions:

```markdown
`run-worker` → inspect exit telemetry immediately → controller validation or
concrete correction relaunch → publish → fresh live submission observation →
watch → improve on an official deficiency or sync the exact verdict.
```

Require `score-report` and the capacity event after worker exit, validation,
deployment, submission, and verdict import.

- [ ] **Step 5: Run GREEN**

Run:

```bash
uv run pytest -q tests/test_repro_loop_paper_owner_skill.py
uv run pytest -q tests/test_repro_loop_worker_guard.py \
  tests/test_repro_loop_controller_validation.py
```

Expected: all tests pass and existing credential/scope isolation remains
unchanged.

- [ ] **Step 6: Commit the lifecycle contract**

```bash
git add skills/icml-repro-loop/SKILL.md \
  skills/icml-repro-loop/references/paper-owner-loop.md
git commit -m "feat: make paper owners drive complete reproductions"
```

---

### Task 3: Make Every Default Dispatch Use the Skill Directly

**Files:**
- Modify: `skills/icml-repro-loop/agents/openai.yaml`
- Modify: `skills/icml-repro-loop/references/submission-checklist.md`
- Test: `tests/test_repro_loop_paper_owner_skill.py`

**Interfaces:**
- Consumes: the direct-dispatch and event-loop contract from Task 2.
- Produces: the default top-level dispatch prompt and controller checklist used for every lane.

- [ ] **Step 1: Replace the default prompt**

Set `default_prompt` in `agents/openai.yaml` to this single-line YAML string:

```yaml
default_prompt: "Use icml-repro-loop directly. Own one paper through selection, design, guarded implementation, controller validation, Space publication, submission, official-score watching, evidence-driven correction, and exact verdict import. Do not return after an implementation worker exits; publish, watch, and correct until the paper has an official score or a genuine persisted blocker."
```

- [ ] **Step 2: Add paper-owner checklist gates**

At the top of `submission-checklist.md`, add:

```markdown
## Paper-Owner Completion Gate

- [ ] The directly dispatched top-level agent invoked `icml-repro-loop` and
  owns exactly one attempt.
- [ ] An implementation-worker exit triggered immediate diff review and fresh
  controller validation without a user status prompt.
- [ ] A rejected validation produced exact correction findings and a guarded
  relaunch on the same attempt.
- [ ] The paper owner continued through `publish-deployment`,
  `attest-submission`, `watch-attempt`, and `sync-verdict`.
- [ ] Pending queue state was watched rather than treated as evidence failure.
- [ ] Judging/scored/blocked emitted a capacity-free event to the competition
  coordinator.
```

- [ ] **Step 3: Add explicit command-order handoff**

After the existing command examples, add:

```text
run-worker
  -> inspect worker-exited telemetry
  -> attest-validation OR correction run-worker
  -> publish-deployment
  -> refresh-live + attest-submission
  -> watch-attempt + record-poll
  -> improvement loop OR sync-verdict
```

State that no arrow is driven by a user status question.

- [ ] **Step 4: Verify prompt and checklist tests**

Run:

```bash
uv run pytest -q tests/test_repro_loop_paper_owner_skill.py
uv run pytest -q tests/test_repro_loop_state.py -k \
  "skill or checklist or response_contract"
```

Expected: all selected tests pass.

- [ ] **Step 5: Commit dispatch metadata**

```bash
git add skills/icml-repro-loop/agents/openai.yaml \
  skills/icml-repro-loop/references/submission-checklist.md
git commit -m "docs: dispatch autonomous paper owners directly"
```

---

### Task 4: Pressure-Test the Revised Skill and Close Loopholes

**Files:**
- Modify: `evals/icml-repro-loop/forward.md`
- Modify if a pressure test exposes ambiguity: `skills/icml-repro-loop/SKILL.md`
- Modify if a pressure test exposes ambiguity: `skills/icml-repro-loop/references/paper-owner-loop.md`
- Test: `tests/test_repro_loop_paper_owner_skill.py`

**Interfaces:**
- Consumes: four new scenarios and the revised skill package.
- Produces: recorded forward evaluations showing direct paper owners do not return early.

- [ ] **Step 1: Run four fresh pressure agents**

Dispatch one fresh read-only agent per new scenario. Give each agent the
scenario prompt and require it to read `skills/icml-repro-loop/SKILL.md` plus
the referenced paper-owner loop. Prohibit mutations and external publication;
evaluate its proposed actions against every `must` condition.

Record for each agent:

- scenario ID;
- agent/session identity;
- exact response;
- PASS/FAIL for each must;
- any rationalization that allowed early return.

- [ ] **Step 2: Verify correction behavior, not keyword repetition**

For `green-but-hard-coded`, require the response to:

1. keep the attempt in implementation/improvement rather than validating it;
2. name the concrete defect in the next worker correction contract;
3. relaunch through `run-worker`;
4. leave publication attestations unperformed.

For `inconclusive-needs-improvement`, require:

1. exact official verdict import/preservation;
2. a new improvement attempt bound to the cited deficiency;
3. a corrected commit and redeployment;
4. watching the new exact SHA.

- [ ] **Step 3: Refactor the skill for any failed pressure condition**

When an agent fails, add the smallest imperative sentence or event-table row
that blocks its exact rationalization. Do not duplicate CLI flag reference
material already available through `state.py --help`.

Rerun only the failed scenario until it passes, then rerun all four.

- [ ] **Step 4: Record forward results**

Append a `Paper-Owner Lifecycle Revision` section to
`evals/icml-repro-loop/forward.md` containing the exact responses and oracle
matrix. Do not report an aggregate pass if any must condition failed.

- [ ] **Step 5: Run full verification**

Run:

```bash
uv run pytest -q tests/test_repro_loop_paper_owner_skill.py
uv run pytest -q \
  tests/test_repro_loop_worker_guard.py \
  tests/test_repro_loop_controller_validation.py \
  tests/test_repro_loop_publication_policy.py \
  tests/test_repro_loop_state.py
uv run "$CODEX_HOME/skills/.system/skill-creator/scripts/quick_validate.py" \
  skills/icml-repro-loop
env UV_CACHE_DIR=/tmp/icml-repro-uv-cache \
  PRE_COMMIT_HOME=/tmp/icml-repro-pre-commit \
  uv run pre-commit run -a
```

Expected: all tests, skill validation, and pre-commit pass. The archival
`submissions/nape/` snapshot is not run or formatted.

- [ ] **Step 6: Commit verified skill**

```bash
git add skills/icml-repro-loop tests/test_repro_loop_paper_owner_skill.py \
  evals/icml-repro-loop
git commit -m "test: verify autonomous paper owner lifecycle"
```

---

### Task 5: Exercise the New Dispatch on One Live Attempt

**Files:**
- Modify only through authoritative commands: `state/repro-loop.json`
- Modify only through authoritative commands: affected `state/repro-loop/attempts/`, `leases/`, `transactions/`, `telemetry/`, and attestation shards
- Do not modify: `docs/HANDOFF.md`

**Interfaces:**
- Consumes: one existing implementing attempt with a controller-rejected proposal.
- Produces: one direct top-level paper-owner run that reacts to the rejection and continues without a user status prompt.

- [ ] **Step 1: Choose one rejected proposal as the live smoke case**

Use RACO attempt `97e213a5-7ca3-4a1b-a500-1ec52d94d87a`, whose first proposal
was rejected for solver, theorem, provenance, status, and root-page defects.
Read its current attempt and lease; do not reuse remembered owner/fence values.

- [ ] **Step 2: Dispatch the top-level owner with only the skill prompt**

Use the exact default prompt from Task 3. The top-level owner must read the
skill, inspect the persisted attempt, observe the correction worker outcome,
and continue to controller validation. Do not paste the lifecycle into the
dispatch message.

- [ ] **Step 3: Verify the first event transition**

The smoke run passes only if, after worker exit, the paper owner either:

- produces a fresh successful controller validation attestation; or
- records exact remaining defects and automatically relaunches a correction.

It fails if it merely reports the worker commit or waits for a user question.

- [ ] **Step 4: Persist telemetry and capacity status**

Run `score-report` after the event and record actual worker runtime from
queued/launched/exited monotonic telemetry. Report whether the attempt consumes
implementation capacity and the exact next lifecycle action.

- [ ] **Step 5: Commit only task-related coordinator state**

Stage the authoritative state/index/shards created by the smoke run. Exclude
the user's existing `docs/HANDOFF.md` edit and all unrelated changes.

```bash
git add state/repro-loop.json state/repro-loop
git commit -m "state: exercise autonomous RACO paper owner"
```
