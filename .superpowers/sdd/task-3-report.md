# Task 3 Report: Persistent Loop State

## Scope

Implemented persistent ICML reproduction-loop state in
`skills/icml-repro-loop/scripts/state.py`, with its state-machine tests in
`tests/test_repro_loop_state.py`, and initialized
`state/repro-loop.json` to a committed idle state.

Public interfaces implemented exactly as specified:

- `new_state() -> dict`
- `load_state(path: Path) -> dict`
- `save_state(path: Path, state: dict) -> None`
- `select_paper(state: dict, paper: dict) -> dict`
- `transition(state: dict, phase: str, **updates: object) -> dict`

## Implementation

- Persists JSON atomically with `NamedTemporaryFile` and `os.replace`, removing
  temporary files after success or failure.
- Validates exact top-level keys, schema version, phases, lifetime cost, history,
  and current-paper presence.
- Enforces the required transition graph, selection identity fields, unique
  completed `paper_id` values, design approval, deployment SHA, space ID, verdict,
  and the $10.00 estimated/actual per-paper API-cost limits.
- Returns deep-copied state updates and records a completed paper and its actual
  cost when it returns to `idle`.
- Provides `init`, `show`, `select`, and `transition` CLI commands. `init` creates
  the target parent directory when needed.

## Red-Green TDD Evidence

1. Initialization, persistence, atomic-write cleanup, and schema-validation tests:
   - Red: `uv run pytest tests/test_repro_loop_state.py -q` produced `8 failed`.
     Every failure was the expected `FileNotFoundError` for the absent
     `scripts/state.py` module.
   - Green: the same command produced `8 passed in 0.12s` after implementing
     initialization, validation, and atomic persistence.

2. Selection, transition graph, lifecycle bookkeeping, cost limits, and CLI tests:
   - Red: the state suite produced `8 passed, 137 failed`; the intended failures
     were missing `select_paper` and `transition` interfaces and missing CLI output.
   - Green correction: the initial implementation run produced `144 passed,
     1 failed`; the sole failure exposed a test fixture that attempted the forbidden
     `improving -> complete` transition. The fixture was corrected to reach
     `complete` from `judging`.
   - Green: the corrected suite produced `145 passed in 0.37s`.

3. CLI initialization in a missing parent directory:
   - Red: the state suite produced `145 passed, 1 failed` because `init` could not
     create a nonexistent parent directory.
   - Green: `save_state` creates `path.parent`; the state suite produced
     `146 passed in 0.45s`.

4. Required command verification:
   - `uv run python skills/icml-repro-loop/scripts/state.py init state/repro-loop.json`
     completed successfully.
   - `uv run python skills/icml-repro-loop/scripts/state.py show state/repro-loop.json`
     emitted version `1`, phase `idle`, null current paper, empty history, and
     total API cost `0.0`.

## Final Verification

- `uv run pytest -q`: `146 passed in 0.49s`.
- `uv run pre-commit run -a`: all hooks passed on the second run. The first run
  added EOF newlines to the existing tracked `references/.gitkeep` and
  `scripts/.gitkeep` files; those hook-required changes are included.
- `git diff --check`: no whitespace errors before final report creation.

## Self-Review

Reviewed the implementation against every Task 3 requirement and the public
interfaces. No unresolved defects or requirement gaps were found. The only
post-green code change was a behavior-preserving simplification of the required
phase-artifact guards, followed by the complete pytest and pre-commit runs above.

## Review Fixes

### Red-Green Evidence

- Red: after adding focused review regression tests, `uv run pytest
  tests/test_repro_loop_state.py -q` produced `147 passed, 21 failed in 0.68s`.
  The failures covered missing derived project paths; non-finite cost acceptance in
  save, load, and transition paths; mutable estimates and identity fields; stale
  artifact acceptance; discarded blocked attempts; CLI overwrite; and duplicate
  project paths and submission space IDs.
- Fix: centralized finite, nonnegative, per-paper $10.00 cost validation for
  current and historical records; used `allow_nan=False` for JSON output; derived
  `submissions/<slug>` project paths; rejected immutable-field and estimate
  changes; required artifact values in the transition updates; archived blocked
  attempts like completed attempts; refused existing CLI init targets; and enforced
  historical project-path and space-ID uniqueness.
- Green: the review regression suite produced `168 passed in 0.62s`.

### Final Verification After Review Fixes

- `uv run pytest -q`: `168 passed in 0.59s`.
- `uv run pre-commit run --all-files`: every configured hook passed.

## Second Review Fix Wave

### Red-Green Evidence

- Red: after adding monotonic-cost, persisted identity, canonical-path, and exact
  type regression tests, `uv run pytest tests/test_repro_loop_state.py -q`
  produced `169 passed, 16 failed in 0.69s`. Failures covered decreasing actual
  cost, later space-ID mutation, persisted duplicate project paths and space IDs,
  invalid or traversing slugs and paths, boolean versions, and list phases.
- Fix: actual API cost updates now reject decreases; a set `space_id` is immutable;
  persisted records require lowercase hyphenated slugs and exact
  `submissions/<slug>` paths; nonempty project paths and present space IDs are
  unique across history and current state; and version/phase types are validated
  before membership checks. Invalid persisted values now raise field-specific
  `ValueError`s rather than leaking `TypeError`.
- Green: the focused state suite produced `185 passed in 0.81s`.

### Final Verification After Second Review Fix Wave

- `uv run pytest -q`: `185 passed in 0.59s`.
- `uv run pre-commit run --all-files`: every configured hook passed.
