# Task 4 Report: Full-Release Future Adaptation Evidence

## Status

DONE

## Changed Files

- `src/icml_2026_repro/future_adaptation_evidence.py`
  - Added the 52-trajectory official evaluator/simulator sweep.
  - Added the deterministic residual-patch fixture.
  - Added the composed future-adaptation evidence builder.
  - Enforced exact pinned NAPE HEAD and clean worktree checks in every public evidence producer.
- `src/icml_2026_repro/online_trace.py`
  - Renamed `build_claim_2_evidence` to `build_online_trace_evidence` without changing trace behavior.
- `src/icml_2026_repro/cli.py`
  - Updated the online trace builder import and call after the rename.
- `src/icml_2026_repro/__init__.py`
  - Updated the public online trace export after the rename.
- `tests/test_future_adaptation_evidence.py`
  - Added release summary, target preservation, residual patch, malformed input, empty release, unequal final state, determinism, composition, and checkout integrity coverage.
- `tests/test_online_trace.py`
  - Updated tests to use the renamed online trace builder.
- `tests/test_cli.py`
  - Updated CLI test patches to use the renamed online trace builder.

## Commits

- `333fd9f452f28374a19e9b5b4cbca271843b9ef7` - `feat: exercise NAPE future adaptation across the release`
- The report is committed separately after this document is written.

## TDD Evidence

### Red

Command:

```text
uv run pytest tests/test_future_adaptation_evidence.py -q
```

Output before implementation:

```text
ERROR collecting tests/test_future_adaptation_evidence.py
ImportError: cannot import name 'future_adaptation_evidence' from 'icml_2026_repro'
1 error in 0.37s
```

The test module could not collect because the required production module did not exist.

### Green

Command:

```text
uv run pytest tests/test_future_adaptation_evidence.py -q
```

Output:

```text
.............                                                            [100%]
13 passed in 15.52s
```

The asserted release summary is `52/50/52/0/52/0`; the residual fixture reports one missing operation and preserves the target.

## Verification

Required focused and upstream future-edit command:

```text
uv run pytest tests/test_future_adaptation_evidence.py tests/test_online_trace.py external/NAPE/tests/test_future_edits.py -q
```

Output:

```text
..........................................................               [100%]
58 passed in 16.37s
```

First pre-commit run:

```text
uv run pre-commit run -a
```

Output summary:

```text
AST, whitespace, prettier, ruff, and pytest passed.
ty failed with 7 diagnostics: three object comparison types, three invariant evaluator list arguments, and one uncast test evidence value.
```

The implementation was corrected with a typed trajectory row and explicit union-list/test casts; no ignores were added.

Final pre-commit run:

```text
uv run pre-commit run -a
```

Output:

```text
check python ast.........................................................Passed
fix end of files.........................................................Passed
trim trailing whitespace.................................................Passed
check for merge conflicts................................................Passed
fix requirements.txt.................................(no files to check)Skipped
prettier-format..........................................................Passed
ruff format..............................................................Passed
ruff (legacy alias)......................................................Passed
ty.......................................................................Passed
pytest...................................................................Passed
```

After writing this report, a further full pre-commit run reformatted
`tests/test_future_adaptation_evidence.py` and exited 1 because hooks changed the file;
all type checks and tests still passed. The hooks were rerun after accepting that
format-only delta, producing the same all-passed output above with exit code 0.

Diff integrity command:

```text
git diff --check
```

Output: empty, exit code 0.

## Self-Review

- Confirmed each release trajectory uses the sorted file order, exact next operation, deterministic `ZZ999` false positive, cached initial/target states, required evaluator arguments, simulator indices `1/1`, and final zero FP/FN/mismatch comparison.
- Confirmed per-trajectory counts come directly from `operations_removed`, `inverse_ops_added`, and `metadata["missing_ops_count"]`.
- Confirmed only `NotImplementedError` enters the explicit unsupported-operation skip path; malformed or short trajectories fail with the source filename.
- Confirmed all three new public evidence producers independently verify pinned HEAD and clean NAPE worktree, while the composed builder also calls the independently verified online trace.
- Confirmed the online trace implementation changed only its public function name and docstring; its trace logic is untouched.
- Confirmed no stale runtime or test imports reference `build_claim_2_evidence`.

## Concerns

None.

## Important Findings Follow-Up

### Changes

- Added top-level and release-sweep `counting_definitions`, `case_counts`, and
  `denominators` evidence.
- Defined eligible, executed, and skipped cases; the deterministic mutation; and
  removal, inverse insertion, residual patch, and target-preserved numerators.
- Defined all four mechanism denominators as executed cases and explicitly excluded
  skipped cases.
- Restricted skip conversion to `UnsupportedOperationError`, which is created only
  when symbolic parsing or pre-evaluation operation application raises
  `NotImplementedError`.
- Added filename/stage context to downstream failures while preserving downstream
  `NotImplementedError` as that exception type.

### TDD Evidence

Initial follow-up red command:

```text
uv run pytest tests/test_future_adaptation_evidence.py -q
```

Output:

```text
F...FFFFF..F......                                                       [100%]
7 failed, 11 passed in 16.85s
```

The failures covered the missing counting contract, broad downstream skip handling,
and missing filename context for unexpected state-building failures.

The downstream propagation test was then strengthened to preserve exception type.

```text
uv run pytest tests/test_future_adaptation_evidence.py -q -k downstream_not_implemented
```

Red output:

```text
FFF                                                                      [100%]
3 failed, 15 deselected in 0.68s
```

Green output after preserving downstream `NotImplementedError`:

```text
...                                                                      [100%]
3 passed, 15 deselected in 0.60s
```

### Verification

Required focused and upstream command:

```text
uv run pytest tests/test_future_adaptation_evidence.py tests/test_online_trace.py external/NAPE/tests/test_future_edits.py -q
```

Final output:

```text
...............................................................          [100%]
63 passed in 17.87s
```

The first follow-up pre-commit run reformatted two files and exited 1; all other
hooks, including `ty` and `pytest`, passed. After accepting the formatter output,
the final command was:

```text
uv run pre-commit run -a
```

Final output:

```text
check python ast.........................................................Passed
fix end of files.........................................................Passed
trim trailing whitespace.................................................Passed
check for merge conflicts................................................Passed
fix requirements.txt.................................(no files to check)Skipped
prettier-format..........................................................Passed
ruff format..............................................................Passed
ruff (legacy alias)......................................................Passed
ty.......................................................................Passed
pytest...................................................................Passed
```

### Follow-Up Self-Review

- Confirmed the composed evidence repeats the counting contract at top level and
  retains it inside `release_sweep`, so consumers do not need external methodology.
- Confirmed the four reported mechanism numerators use executed trajectories as
  their explicit denominator; release, eligible, executed, and skipped counts are
  separately reported.
- Confirmed malformed/short inputs still fail, explicitly unsupported symbolic
  parsing or pre-evaluation application alone may skip, and skipped rows contribute
  to none of the four mechanism denominators.
- Confirmed evaluator, simulator, rebuilt-state, comparator, and target mismatch
  failures abort the audit with trajectory context; downstream `NotImplementedError`
  remains `NotImplementedError`.
- Confirmed the pinned release remains `52/50/52/0/52/0` with denominators of 52.

### Follow-Up Concerns

None.

## Final Important Finding Follow-Up

### Changes

- Kept release-sweep residual occurrence at zero cases with 52 executed release
  trajectories as its denominator.
- Made the residual correction fixture a separate `residual_correction` mechanism
  result with its own counting definitions and
  `deterministic_fixture_mechanism_proof` scope.
- Added fixture counts for one eligible case, one executed case, zero skipped cases,
  one residual-patch case, and one target-preserved case.
- Added explicit denominator populations: execution uses eligible cases; residual
  correction and target preservation use executed cases.

### TDD Evidence

Initial red command:

```text
uv run pytest tests/test_future_adaptation_evidence.py -q
```

Output:

```text
.F................                                                       [100%]
1 failed, 17 passed in 17.17s
```

The residual fixture failed on the missing `mechanism` field while the release sweep
continued to pass its zero-of-52 assertions.

The denominator population contract was then strengthened with a focused red test:

```text
uv run pytest tests/test_future_adaptation_evidence.py -q -k residual_patch_fixture_synthesizes
```

Output:

```text
F                                                                        [100%]
1 failed, 17 deselected in 0.39s
```

The failure was the missing `denominator_definitions` field.

Focused green output before the denominator-name strengthening was:

```text
..................                                                       [100%]
18 passed in 16.68s
```

### Verification

Required focused and upstream command:

```text
uv run pytest tests/test_future_adaptation_evidence.py tests/test_online_trace.py external/NAPE/tests/test_future_edits.py -q
```

Final output:

```text
...............................................................          [100%]
63 passed in 17.50s
```

The first pre-commit run reformatted two files and reported two strict test typing
errors; all behavioral tests passed. The release summary and denominator values were
explicitly cast, then the final command was:

```text
uv run pre-commit run -a
```

Final output:

```text
check python ast.........................................................Passed
fix end of files.........................................................Passed
trim trailing whitespace.................................................Passed
check for merge conflicts................................................Passed
fix requirements.txt.................................(no files to check)Skipped
prettier-format..........................................................Passed
ruff format..............................................................Passed
ruff (legacy alias)......................................................Passed
ty.......................................................................Passed
pytest...................................................................Passed
```

### Final Self-Review

- Confirmed release-sweep `summary["residual_patch_cases"]` is 0 and
  `denominators["residual_patch_cases"]` is 52.
- Confirmed the fixture is not added to release trajectories, release case counts, or
  release denominators.
- Confirmed fixture execution is reported as one executed case over one eligible case,
  with zero skips.
- Confirmed residual correction and target preservation are each one case over the
  fixture's one executed case.
- Confirmed the fixture retains the exact four-operation ground truth, two-operation
  prediction, one synthesized missing operation, and `target_preserved=True`.

### Final Concerns

None.
