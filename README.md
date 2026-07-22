# ICML 2026 Reproductions

This repository is the parent workspace for independently reproducible ICML
2026 Agent Repro Challenge submissions. Each paper lives in its own project
under `submissions/`; the parent does not provide shared runtime code.

Root-authored content is licensed under the [MIT License](LICENSE). Bundled
NAPE components retain their own licenses; see
[`submissions/nape/LICENSE`](submissions/nape/LICENSE) and
[`submissions/nape/external/NAPE/LICENSE`](submissions/nape/external/NAPE/LICENSE).

## Layout

- `submissions/<paper>/`: independent Python project with its own lockfile,
  tests, evidence bundle, validation commands, and Hugging Face Space source.
- `skills/icml-repro-loop/`: versioned source for the reproduction-loop skill.
- `state/repro-loop.json`: resumable reproduction-loop state.
- `docs/HANDOFF.md`: current research state and next action.
- `docs/REMOTE_SETUP.md`: host setup, authentication, skill installation, and
  verification instructions.

Each challenge entry is deployed to a separate Hugging Face Space because
challenge metadata and judging are specific to that Space.

## NAPE Snapshot

[`submissions/nape/`](submissions/nape/) is an immutable convenience snapshot
of the canonical [NAPE reproduction repository](https://github.com/will-rice/icml-2026-repro).
It is not an independently runnable submission: its archive intentionally
contains no nested Git metadata. The canonical repository remains authoritative
and is not replaced by this copy. See
[`submissions/nape/UPSTREAM.md`](submissions/nape/UPSTREAM.md) for the immutable
source revisions and import method.

Verify the parent workspace from the repository root:

```bash
CODEX_HOME=${CODEX_HOME:-$HOME/.codex}
uv sync --frozen
uv run pytest -q
uv run "$CODEX_HOME/skills/.system/skill-creator/scripts/quick_validate.py" skills/icml-repro-loop
uv run pre-commit run -a
```

Run NAPE evidence commands in the
[canonical repository](https://github.com/will-rice/icml-2026-repro), where its
Git provenance is available. At the imported canonical revision, `uv run
pytest -q` passed 119 tests.

## Reproduction Skill

The versioned skill source is at `skills/icml-repro-loop/`. Install it on a
Codex host by following [`docs/REMOTE_SETUP.md`](docs/REMOTE_SETUP.md), which
creates the `$CODEX_HOME/skills/icml-repro-loop` symlink and verifies it.

## Submission Status

AgentSelect was screened as a candidate but is not implemented here because it
became judged. `docs/HANDOFF.md` is the authoritative record for the current
candidate and future work.
