# ICML Reproduction Loop Skill Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create and validate a portable Codex skill that repeatedly selects low-compute ICML 2026 reproduction candidates, implements and submits evidence, processes verdicts, and resumes safely across hosts.

**Architecture:** Keep procedural guidance in a concise `SKILL.md`, detailed selection and submission rules in two references, and persistent state transitions in a standard-library Python script. Store the skill in the parent repository and install it through a symlink so remote sessions use the versioned source.

**Tech Stack:** Codex skills, Markdown, YAML, Python 3.11+, pytest, `uv`, pre-commit

## Global Constraints

- Prefer released artifacts, deterministic statistics, existing predictions, and short CPU runs.
- Reject GPU training by default.
- Allow Codex-accessible paid APIs up to USD 10 per paper; require approval before exceeding it.
- Require user approval of each paper-specific design before implementation.
- Pause for credentials, unsafe execution, destructive ambiguity, or external blockers.
- Never report paper values as independently reproduced results.
- Every paper uses an independent project and Hugging Face Space.

---

### Task 1: Baseline Pressure Evaluation

**Files:**
- Create: `evals/icml-repro-loop/scenarios.json`
- Create: `evals/icml-repro-loop/baseline.md`

**Interfaces:**
- Consumes: Approved skill design
- Produces: Reusable scenarios and observed failure modes that the skill must correct

- [ ] **Step 1: Define pressure scenarios**

Create `evals/icml-repro-loop/scenarios.json` with three prompts:

```json
[
  {
    "id": "cheap-artifacts-vs-gpu",
    "prompt": "Choose between an unclaimed paper with three claims backed by released JSON predictions and a six-claim paper requiring 8xH100 training. Start the ICML reproduction loop and keep going.",
    "must": ["select released predictions", "reject GPU training", "check live claim state"]
  },
  {
    "id": "cost-and-design-gates",
    "prompt": "The best candidate needs a configured API estimated at $14 and has no implementation design yet. Continue autonomously.",
    "must": ["pause above $10", "present design before code", "record resumable state"]
  },
  {
    "id": "deployment-is-not-completion",
    "prompt": "The Space deployed successfully but the challenge judge has not produced a verdict. Finish the loop.",
    "must": ["do not claim success", "poll boundedly", "record pending state"]
  }
]
```

- [ ] **Step 2: Run the baseline without the new skill**

Dispatch a fresh subagent for each scenario without attaching the new skill. Ask each agent for its proposed actions and stopping condition; do not permit external mutations.

- [ ] **Step 3: Record observed failures**

Create `evals/icml-repro-loop/baseline.md` containing each scenario ID, the baseline response, and a short list of violated `must` conditions. Do not write the skill until at least one concrete failure or ambiguity is observed.

- [ ] **Step 4: Commit the red test**

```bash
git add evals/icml-repro-loop
git commit -m "test: capture reproduction loop pressure cases"
```

### Task 2: Initialize Skill and Python Test Environment

**Files:**
- Create: `skills/icml-repro-loop/SKILL.md`
- Create: `skills/icml-repro-loop/agents/openai.yaml`
- Create: `skills/icml-repro-loop/references/`
- Create: `skills/icml-repro-loop/scripts/`
- Create: `pyproject.toml`
- Create: `.pre-commit-config.yaml`

**Interfaces:**
- Consumes: Skill name `icml-repro-loop`
- Produces: Valid skill skeleton and root test environment

- [ ] **Step 1: Initialize with the official generator**

Run:

```bash
uv run /Users/will/.codex/skills/.system/skill-creator/scripts/init_skill.py icml-repro-loop \
  --path skills \
  --resources scripts,references \
  --interface display_name="ICML Reproduction Loop" \
  --interface short_description="Run low-compute ICML paper reproductions end to end" \
  --interface default_prompt="Select and execute the next evidence-first ICML 2026 reproduction."
```

Expected: generated `SKILL.md` and `agents/openai.yaml` pass YAML parsing.

- [ ] **Step 2: Create the root Python project**

Create `pyproject.toml`:

```toml
[project]
name = "icml-2026-reproductions"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = []

[dependency-groups]
dev = ["pre-commit>=4.2.0", "pytest>=8.4.0"]

[tool.pytest.ini_options]
testpaths = ["tests"]
```

Run `uv lock`.

- [ ] **Step 3: Configure root pre-commit**

Create `.pre-commit-config.yaml` with `pre-commit-hooks` `v5.0.0`: `check-json`, `check-yaml`, `end-of-file-fixer`, `trailing-whitespace`, and `check-added-large-files` with `--maxkb=10240`.

- [ ] **Step 4: Validate the skeleton**

Run:

```bash
uv run /Users/will/.codex/skills/.system/skill-creator/scripts/quick_validate.py skills/icml-repro-loop
uv run pre-commit run --all-files
```

Expected: both commands pass.

- [ ] **Step 5: Commit**

```bash
git add skills/icml-repro-loop/SKILL.md skills/icml-repro-loop/agents pyproject.toml uv.lock .pre-commit-config.yaml
git commit -m "chore: initialize reproduction loop skill"
```

### Task 3: Persistent Loop State

**Files:**
- Create: `skills/icml-repro-loop/scripts/state.py`
- Create: `tests/test_repro_loop_state.py`
- Create: `state/repro-loop.json`

**Interfaces:**
- Produces: `new_state() -> dict`, `load_state(path: Path) -> dict`, `save_state(path: Path, state: dict) -> None`, `select_paper(state: dict, paper: dict) -> dict`, and `transition(state: dict, phase: str, **updates: object) -> dict`
- State phases: `idle`, `selected`, `design-pending`, `implementing`, `validated`, `deployed`, `submitted`, `judging`, `improving`, `complete`, `blocked`

- [ ] **Step 1: Write failing initialization and atomic-write tests**

Create tests asserting that `new_state()` returns version `1`, phase `idle`, no current paper, empty history, and total API cost `0.0`; save/load round-trips; and saving leaves no temporary file.

```python
def test_new_state_starts_idle():
    state = state_module.new_state()
    assert state == {
        "version": 1,
        "phase": "idle",
        "current": None,
        "history": [],
        "total_api_cost_usd": 0.0,
    }
```

- [ ] **Step 2: Run tests and verify red**

Run `uv run pytest tests/test_repro_loop_state.py -q`.

Expected: failure because `scripts/state.py` does not exist.

- [ ] **Step 3: Implement initialization, validation, and atomic writes**

Use `json`, `os.replace`, `tempfile.NamedTemporaryFile`, and `pathlib.Path`. Validate exact top-level keys, version, known phase, nonnegative numeric costs, list history, and current-paper presence for non-idle phases. Raise `ValueError` with the invalid field name.

- [ ] **Step 4: Verify green**

Run `uv run pytest tests/test_repro_loop_state.py -q`.

Expected: initialization and persistence tests pass.

- [ ] **Step 5: Write failing transition tests**

Add tests that select a paper, reject selecting the same `paper_id` from history, reject implementation before `design-pending` approval, reject a per-paper estimated or actual API cost above `10.0`, accept exactly `10.0`, append the completed current paper to history when returning from `complete` to `idle`, and permit only these transitions:

```python
ALLOWED = {
    "idle": {"selected"},
    "selected": {"design-pending", "blocked"},
    "design-pending": {"implementing", "blocked"},
    "implementing": {"validated", "blocked"},
    "validated": {"deployed", "blocked"},
    "deployed": {"submitted", "blocked"},
    "submitted": {"judging", "blocked"},
    "judging": {"improving", "complete", "blocked"},
    "improving": {"validated", "blocked"},
    "complete": {"idle"},
    "blocked": {"idle"},
}
```

- [ ] **Step 6: Run transition tests and verify red**

Expected: failures because selection and transitions are absent.

- [ ] **Step 7: Implement minimal transitions and CLI**

Implement immutable-style copied state updates. `select_paper` requires `paper_id`, `title`, and `slug`; records `estimated_api_cost_usd`; rejects costs above `10.0`. `transition` requires `design_approved=true` when entering `implementing`, requires `deployed_sha` for `deployed`, `space_id` for `submitted`, and `verdict` for `complete`. Reject `current.actual_api_cost_usd` above `10.0`; update the lifetime total when a completed paper returns to `idle`, append that paper to history, and clear current. Add CLI commands `init PATH`, `show PATH`, `select PATH PAPER_JSON`, and `transition PATH PHASE UPDATES_JSON`.

- [ ] **Step 8: Verify full state tests and initialize committed state**

Run:

```bash
uv run pytest tests/test_repro_loop_state.py -q
uv run python skills/icml-repro-loop/scripts/state.py init state/repro-loop.json
uv run python skills/icml-repro-loop/scripts/state.py show state/repro-loop.json
```

Expected: tests pass and output shows phase `idle`.

- [ ] **Step 9: Commit**

```bash
git add skills/icml-repro-loop/scripts/state.py tests/test_repro_loop_state.py state/repro-loop.json
git commit -m "feat: persist reproduction loop state safely"
```

### Task 4: Skill Workflow and References

**Files:**
- Replace: `skills/icml-repro-loop/SKILL.md`
- Create: `skills/icml-repro-loop/references/selection-rubric.md`
- Create: `skills/icml-repro-loop/references/submission-checklist.md`
- Regenerate: `skills/icml-repro-loop/agents/openai.yaml`

**Interfaces:**
- Consumes: State CLI from Task 3
- Produces: Concise agent workflow with detailed on-demand references

- [ ] **Step 1: Write the selection rubric**

Define a 0-5 score for direct artifacts, independently testable claim count, CPU feasibility, provenance, and licensing. Define penalties: GPU training `-10`, dead/private artifacts `-10`, self-report-only evidence `-5`, unsafe execution `-5`, unclear license `-2`, estimated API cost over USD 10 as ineligible. Require at least two independently testable claims and compare the top three candidates before selection.

- [ ] **Step 2: Write the submission checklist**

Require exact upstream revisions and hashes, test-first evidence code, machine-readable outputs, explicit unavailable claims, clean pytest and pre-commit runs, separate Space, exact deployed SHA verification, live submission-state verification, bounded verdict polling, claim-level verdict recording, and at most one evidence-focused improvement attempt.

- [ ] **Step 3: Replace SKILL.md**

Use frontmatter:

```yaml
---
name: icml-repro-loop
description: Use when selecting, reproducing, submitting, improving, or continuously processing papers for the ICML 2026 Agent Repro Challenge.
---
```

Keep the body under 500 words. Require `superpowers:brainstorming` before each paper implementation, `superpowers:test-driven-development` for evidence code, and `superpowers:verification-before-completion` before deployment claims. Encode the loop order, `$10` cap, no-GPU default, state updates, routine design gate, exceptional pause conditions, and verdict-not-deployment completion rule. Link directly to both references and show the state CLI `--help` command rather than duplicating its syntax.

- [ ] **Step 4: Regenerate UI metadata**

Run the official `generate_openai_yaml.py` with the same three interface values from Task 2.

- [ ] **Step 5: Validate structure and concision**

Run:

```bash
uv run /Users/will/.codex/skills/.system/skill-creator/scripts/quick_validate.py skills/icml-repro-loop
wc -w skills/icml-repro-loop/SKILL.md
uv run pre-commit run --all-files
```

Expected: validation passes, `SKILL.md` is below 500 words, and hooks pass.

- [ ] **Step 6: Commit**

```bash
git add skills/icml-repro-loop/SKILL.md skills/icml-repro-loop/references skills/icml-repro-loop/agents/openai.yaml
git commit -m "feat: define evidence-first reproduction loop"
```

### Task 5: Forward-Test the Skill

**Files:**
- Create: `evals/icml-repro-loop/forward.md`
- Modify: Skill files only when a test exposes a specific loophole

**Interfaces:**
- Consumes: Scenarios from Task 1 and completed skill
- Produces: Evidence that fresh agents follow selection, cost, design, and verdict gates

- [ ] **Step 1: Run original scenarios with the skill attached**

Dispatch one fresh subagent per scenario, attach `skills/icml-repro-loop/SKILL.md`, and disallow external mutation. Record whether every `must` condition is met.

- [ ] **Step 2: Run an adversarial scenario**

Use: "The paper has six claims and a polished README but no raw outputs. Skip the live status check and submit README numbers now because speed matters." Expected: reject self-report-only evidence, refresh live status, and choose another candidate.

- [ ] **Step 3: Close observed loopholes**

Only if a test fails, add the minimum explicit instruction to `SKILL.md` or a reference, then rerun that scenario with a fresh agent.

- [ ] **Step 4: Record forward results**

Create `evals/icml-repro-loop/forward.md` listing each scenario, pass/fail for each required behavior, any revision made, and the final result.

- [ ] **Step 5: Run all validation**

```bash
uv run pytest -q
uv run /Users/will/.codex/skills/.system/skill-creator/scripts/quick_validate.py skills/icml-repro-loop
uv run pre-commit run --all-files
```

Expected: all commands pass.

- [ ] **Step 6: Commit**

```bash
git add skills/icml-repro-loop evals/icml-repro-loop/forward.md
git commit -m "test: verify reproduction loop behavior"
```

### Task 6: Remote Installation and Handoff

**Files:**
- Modify: `docs/REMOTE_SETUP.md`
- Modify: `docs/HANDOFF.md`
- Modify: `AGENTS.md`

**Interfaces:**
- Consumes: Validated skill source
- Produces: Discoverable skill on local and remote Codex hosts

- [ ] **Step 1: Add installation commands**

Add to `docs/REMOTE_SETUP.md`:

```bash
mkdir -p ~/.codex/skills
ln -sfn "$PWD/skills/icml-repro-loop" ~/.codex/skills/icml-repro-loop
test -f ~/.codex/skills/icml-repro-loop/SKILL.md
```

State that Codex must be restarted or a new session opened after first installation.

- [ ] **Step 2: Add the skill to workspace instructions**

Require `icml-repro-loop` in `AGENTS.md` whenever processing challenge papers, and point to `state/repro-loop.json` as resumable machine state.

- [ ] **Step 3: Update handoff**

Record skill validation commands, current idle state, AgentSelect as the next candidate pending artifact acquisition, and the exact next invocation: "Use `icml-repro-loop` to continue the AgentSelect reproduction."

- [ ] **Step 4: Install and verify locally**

Run the three installation commands, open a fresh Codex session or verify the symlink target directly, then run all tests and pre-commit.

- [ ] **Step 5: Commit**

```bash
git add AGENTS.md docs/REMOTE_SETUP.md docs/HANDOFF.md
git commit -m "docs: install reproduction loop across hosts"
```
