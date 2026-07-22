# Current Handoff

## Loop State

- `state/repro-loop.json` is authoritative and currently records `idle` with
  no current paper, no history, one recorded rejection, and USD 0.00 total API
  cost.
- The state file uses schema version 3. Selection requires explicit
  `estimated_api_cost_usd`, immutable `upstream_revision`, and at least two
  unique `target_claims`. Each judging entry starts a new poll round; verdict
  history is authoritative and the final `verdict` mirrors its last record.
- The source skill is `skills/icml-repro-loop/`; install it according to
  `docs/REMOTE_SETUP.md` before using a new host.

## Published Parent Repository

- Public repository: https://github.com/will-rice/icml-2026-reproductions
- Verified baseline revision: `02c8379d22859de0966127dfb0793b7dbc2cb7b8`.
- Clone-verified baseline at check time:
  `e1a0c4a24b92805e383d6cdbe7181db7ca6c62b3`.
- A fresh shallow clone of that baseline succeeded and contained `SKILL.md`,
  `REMOTE_SETUP.md`, and `HANDOFF.md`. Subsequent documentation commits may
  advance `main`.

Verify the current remote and local heads with:

```bash
git ls-remote origin refs/heads/main
git rev-parse HEAD
```

This document cannot contain its own commit SHA because its contents
participate in that commit's hash.

## Candidate Record

- AgentSelect, OpenReview `4M5Kj2UqaM` / arXiv `2603.03761`, is already judged
  and recorded as rejected in `state/repro-loop.json`. It must not be selected
  or resumed. Its historical artifacts were the
  [official repository](https://github.com/Ancientshi/AgentSelect) and the
  [full dataset](https://drive.google.com/drive/folders/1wAzaUxOzPrwuF4s_iRT4NlRqV8gbLKe6?usp=sharing).
- WF-Bench was the latest read-only candidate. It is not yet selected and has
  not received a paper-specific implementation design approval.

## Next Action

Use `icml-repro-loop` to refresh the live challenge status and re-evaluate
WF-Bench. Do not select AgentSelect.

Before selecting WF-Bench, perform a fresh live status, primary-artifact,
license, provenance, and CPU-feasibility check, compare the top eligible
candidates as required by the selection rubric, and persist each ineligible
candidate while idle with:

```bash
uv run python skills/icml-repro-loop/scripts/state.py reject state/repro-loop.json CANDIDATE_JSON
```

`reject` records the candidate decision without a phase transition. If WF-Bench
remains eligible and is selected, use
`superpowers:brainstorming`, persist `design-pending`, present a paper-specific
design, and wait for explicit user approval before writing evidence code.

Selection JSON now includes the target claim names:

```bash
uv run python skills/icml-repro-loop/scripts/state.py select state/repro-loop.json '{"paper_id":"PAPER_ID","title":"TITLE","slug":"paper-slug","estimated_api_cost_usd":0.0,"upstream_revision":"REVISION","target_claims":["claim-1","claim-2"]}'
```

Use the judging, improvement, and completion JSON examples in
`skills/icml-repro-loop/references/submission-checklist.md`; improvement and
completion verdicts must cover exactly the selected target claims.

Blocked transitions require a nonempty `blocker` and record `blocked_from`.
Resume by transitioning back to that phase. Do not archive a blocked attempt
unless the user explicitly directs `{"abandon": true}`; abandonment records the
attempt in history and accounts its actual API cost.

## Validation Commands

```bash
CODEX_HOME=${CODEX_HOME:-$HOME/.codex}
uv sync --frozen
uv run pytest -q
uv run "$CODEX_HOME/skills/.system/skill-creator/scripts/quick_validate.py" skills/icml-repro-loop
uv run pre-commit run -a
```
