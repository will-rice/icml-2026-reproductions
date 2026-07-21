# Task 5 Report: Judge-Aligned Evidence Bundle

## Status

Complete. The CLI now emits the six-artifact judge-aligned bundle using the
reviewed Claims 1-3 builders. Claims 4-6 remain explicitly not replicated.

## Commit

- `81230d2 feat: publish judge-aligned NAPE evidence bundle`

Commit output:

```text
[score-improvement-worktree 81230d2] feat: publish judge-aligned NAPE evidence bundle
 11 files changed, 842 insertions(+), 225 deletions(-)
 delete mode 100644 repro_bundle/claim_1_audit.json
 create mode 100644 repro_bundle/claim_1_benchmark.json
 create mode 100644 repro_bundle/claim_2_predictability.json
 delete mode 100644 repro_bundle/claim_2_trace.json
 create mode 100644 repro_bundle/claim_3_future_adaptation.json
 create mode 100644 repro_bundle/claims_4_6_status.json
```

## Changed Files

- `src/icml_2026_repro/cli.py`
  - Replaced legacy Claim 1 audit and Claim 2 trace builders with the reviewed
    benchmark, predictability, and future-adaptation builders.
  - Changed `BUNDLE_ARTIFACT_NAMES` to the exact six-artifact contract.
  - Builds all three reports before writing any artifact and retains a private
    temporary directory for Claim 3 recorder files.
  - Added the exact Claims 4-6 not-replicated status object.
  - Updated the environment schema to v2 and added the focused evidence command.
  - Rewrote the portable bundle guide with the reviewed evidence semantics.
- `tests/test_cli.py`
  - Added the six-artifact contract, exact builder-output assertions, exact
    Claims 4-6 status assertion, portability assertion, v2 schema assertion,
    and required evidence-guide text.
  - Retargeted macOS alias, symlink, hard-link, race, and staging tests to the
    three reviewed builders and `BUNDLE_ARTIFACT_NAMES`.
- `README.md`
  - Documents Claims 1-3 evidence, explicit Claims 4-6 status, setup, focused
    and full validation commands, official validation, and canonical Space URL.
- `repro_bundle/README.md`
- `repro_bundle/environment.json`
- Added `repro_bundle/claim_1_benchmark.json`.
- Added `repro_bundle/claim_2_predictability.json`.
- Added `repro_bundle/claim_3_future_adaptation.json`.
- Added `repro_bundle/claims_4_6_status.json`.
- Removed `repro_bundle/claim_1_audit.json`.
- Removed `repro_bundle/claim_2_trace.json`.

## TDD Evidence

Tests were changed before production code. The RED run was:

```bash
uv run pytest tests/test_cli.py -q
```

```text
F.........FFF.FFF                                                        [100%]
7 failed, 10 passed in 0.75s
```

The failures showed the expected legacy contract: `claim_1_audit.json` and
`claim_2_trace.json` were present instead of the new Claim 1-3 and status files,
the manifest was v1, and race/staging errors named the old Claim 1 artifact.

After implementing the minimum bundle revision, the GREEN run was:

```text
.....................                                                    [100%]
21 passed in 0.43s
```

The final focused run after formatting and fixture cleanup was:

```text
.....................                                                    [100%]
21 passed in 1.16s
```

## Regeneration

First and second generation command:

```bash
uv run nape-repro
```

Both runs produced:

```text
Large fingerprint (1105 > 1000 threshold), processing may be slow
Large fingerprint (1501 > 1000 threshold), processing may be slow
Wrote reproduction bundle to repro_bundle
```

Snapshot and comparison:

```bash
cp -R repro_bundle /private/tmp/nape-repro-first
uv run nape-repro
diff -ru /private/tmp/nape-repro-first repro_bundle
```

`cp` and the second generation exited 0. `diff` exited 0 and printed nothing.
The checked-in output contains exactly:

```text
README.md
claim_1_benchmark.json
claim_2_predictability.json
claim_3_future_adaptation.json
claims_4_6_status.json
environment.json
```

An additional contract audit parsed every JSON artifact, checked the exact file
set, and found no `/Users/` or `/private/` path in the bundle:

```text
six artifacts; all JSON valid; no workspace absolute paths
```

## Quality Gates

Final root suite:

```bash
uv run pytest -q
```

```text
........................................................................ [ 69%]
................................                                         [100%]
104 passed in 18.27s
```

Final upstream suite:

```bash
uv run pytest external/NAPE/tests -q
```

```text
........................................................................ [ 29%]
........................................................................ [ 59%]
........................................................................ [ 88%]
............................                                             [100%]
244 passed in 0.30s
```

Pre-commit initially reformatted `tests/test_cli.py` and fixed one Ruff finding,
so that run exited 1 as expected for modifying hooks. It was rerun immediately,
then run once more after the final fixture cleanup. Final output:

```bash
uv run pre-commit run -a
```

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

Explicit type check:

```bash
uv run ty check src tests
```

```text
All checks passed!
```

Whitespace/error check:

```bash
git diff --check
```

Exited 0 with no output. `git diff --cached --check` also exited 0 with no
output before commit.

## Self-Review

- Confirmed `build_bundle()` calls only the three reviewed public interfaces.
- Confirmed all Claims 1-3 evidence is built before the first artifact write.
- Confirmed Claim 3 still receives a private temporary recorder directory.
- Confirmed the status artifact exactly repeats the required status and reason
  for each of Claims 4, 5, and 6.
- Confirmed the generated guide includes `52`, `11,907`, `35-821`, `229`, `164`,
  `68.04%`, `50/52`, `52/52`, one residual fixture, released oracle-output
  scope, and explicit Claims 4-6 status.
- Confirmed no legacy artifact files, v1 schema, or legacy builder references
  remain in the generated bundle.
- Confirmed the existing no-follow directory traversal, symlink preflight,
  hard-link replacement, post-preflight symlink race, atomic staging, and
  staging-cleanup behavior remains covered by passing tests.
- Confirmed the commit includes only the 11 scoped source, test, documentation,
  and generated artifact changes specified by Task 5.

## Concerns

- Evidence generation emits two non-failing large-fingerprint performance
  warnings from the evaluator. They are deterministic and do not affect output.
- No functional or test concerns remain.

## Important Findings Fixes

### Status And Commit

Both Important findings are resolved in implementation commit:

- `eb010a2 fix: enforce exact reproduction bundle contents`

```text
[score-improvement-worktree eb010a2] fix: enforce exact reproduction bundle contents
 3 files changed, 167 insertions(+), 5 deletions(-)
```

### Changed Files

- `src/icml_2026_repro/cli.py`
  - Defines the two known legacy artifact names.
  - Inventories the reused output directory through its already-open FD.
  - Rejects every unknown entry before calling an evidence builder.
  - Permits only regular legacy files and never follows links while validating.
  - Revalidates and removes only known legacy names through the open directory
    FD after all six new artifacts have been written successfully.
  - Adds the zero-paid-call statement and canonical Space URL to the generated
    bundle guide.
- `tests/test_cli.py`
  - Covers successful upgrade from both regular legacy files.
  - Covers pre-builder rejection and preservation of an unknown sentinel.
  - Covers both legacy names as symlinks and directories.
  - Covers preservation of both legacy files when a new artifact write fails.
  - Asserts the zero-paid-call statement and canonical Space URL.
- `repro_bundle/README.md`
  - Regenerated with the zero-paid-call statement and canonical Space URL.
- `.superpowers/sdd/task-5-report.md`
  - Appends this exact finding-resolution evidence and self-review.

### TDD Evidence

The new regression tests were written before the implementation change.

RED command:

```bash
uv run pytest tests/test_cli.py -q
```

```text
FFFFFFF.....................                                             [100%]
7 failed, 21 passed in 1.72s
```

The seven failures independently showed the missing guide text, both retained
legacy files, the unexpected sentinel reaching Claim 1, and both legacy names
reaching Claim 1 as either symlinks or directories.

GREEN command:

```bash
uv run pytest tests/test_cli.py -q
```

```text
............................                                             [100%]
28 passed in 0.36s
```

### Deterministic Regeneration

Both generation runs:

```bash
uv run nape-repro
```

```text
Large fingerprint (1105 > 1000 threshold), processing may be slow
Large fingerprint (1501 > 1000 threshold), processing may be slow
Wrote reproduction bundle to repro_bundle
```

Snapshot and comparison:

```bash
cp -R repro_bundle /private/tmp/nape-repro-task5-fixes-first
uv run nape-repro
diff -ru /private/tmp/nape-repro-task5-fixes-first repro_bundle
```

The snapshot, second generation, and diff all exited 0. The diff printed
nothing. The regenerated directory contains exactly the six current artifacts.

### Final Quality Gates

Focused CLI suite:

```bash
uv run pytest tests/test_cli.py -q
```

```text
............................                                             [100%]
28 passed in 0.36s
```

Root suite:

```bash
uv run pytest -q
```

```text
........................................................................ [ 64%]
.......................................                                  [100%]
111 passed in 18.33s
```

Pinned upstream suite:

```bash
uv run pytest external/NAPE/tests -q
```

```text
........................................................................ [ 29%]
........................................................................ [ 59%]
........................................................................ [ 88%]
............................                                             [100%]
244 passed in 0.30s
```

Pre-commit:

```bash
uv run pre-commit run -a
```

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

Explicit type check:

```bash
uv run ty check src tests
```

```text
All checks passed!
```

```bash
git diff --check
```

Exited 0 with no output.

### Findings Self-Review

- The preflight inventories names with `os.listdir(output_directory_fd)` and
  rejects an unknown entry before any evidence builder can run.
- Unknown files are never unlinked. Only names captured from the two-name
  legacy allowlist can reach cleanup.
- Legacy validation uses `os.stat(..., dir_fd=output_directory_fd,
  follow_symlinks=False)` before generation and immediately before cleanup.
- Legacy cleanup uses `os.unlink(..., dir_fd=output_directory_fd)` and therefore
  never resolves a caller-controlled path or follows a symlink target.
- All six new artifacts are built and written through the existing atomic
  staging path before legacy cleanup begins.
- A simulated first-artifact write failure leaves both regular legacy files and
  their contents unchanged and removes the staging file.
- Existing output-directory symlink, ancestor symlink, macOS alias, artifact
  symlink, hard-link, post-preflight race, and staging-failure tests remain green.
- The generated and checked-in bundle guide contains the literal phrase
  `zero paid API calls` and the canonical Space URL.

### Findings Concerns

- Evidence generation still emits the two deterministic, non-failing
  large-fingerprint performance warnings recorded above.
- No functional or test concerns remain.
