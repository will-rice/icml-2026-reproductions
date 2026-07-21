# Current Handoff

## Loop State

- `state/repro-loop.json` is authoritative and currently records `idle` with
  no current paper, no history, one recorded rejection, and USD 0.00 total API
  cost.
- The source skill is `skills/icml-repro-loop/`; install it according to
  `docs/REMOTE_SETUP.md` before using a new host.

## Published Parent Repository

- Public repository: https://github.com/will-rice/icml-2026-reproductions
- Verified baseline revision: `02c8379d22859de0966127dfb0793b7dbc2cb7b8`.
- Remote `main` head: `e1a0c4a24b92805e383d6cdbe7181db7ca6c62b3`.
- A fresh shallow clone succeeded and contained `SKILL.md`, `REMOTE_SETUP.md`,
  and `HANDOFF.md`.

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

## Validation Commands

```bash
CODEX_HOME=${CODEX_HOME:-$HOME/.codex}
uv sync --frozen
uv run pytest -q
uv run "$CODEX_HOME/skills/.system/skill-creator/scripts/quick_validate.py" skills/icml-repro-loop
uv run pre-commit run -a
```
