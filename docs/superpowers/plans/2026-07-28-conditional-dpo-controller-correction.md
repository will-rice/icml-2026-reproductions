# Conditional DPO Controller Correction Contract

## Authority and scope

This is a controller-authored correction contract for attempt
`933665ed-b7ed-4d73-9b07-35704660a184`, paper `7UEBX1KU1y`, in phase
`implementing`. Modify and commit only:

`submissions/conditional-equivalence-of-dpo-and-rlhf-assumptions-failure-modes-and-provable-alignment/`

Do not edit coordinator state, this contract, the approved design, another
submission, credentials, Hub resources, or controller documents. Return the
new commit, exact tests/commands, evidence paths and hashes, and concerns as a
proposal only.

## Controller findings and established root causes

1. `validate_evidence` ignores `schema_path`; the JSON file advertises a closed
   schema but source and lane-detail objects are open, and runtime validation
   accepts unknown fields. Root cause: the initial implementation performed
   only a few hand-written identity checks and never consumed the schema.
2. `evidence/validation.json` says `full_test_count: 27`, while the clean
   proposal runs 31 tests before these corrections. Root cause: static
   validation metadata was not regenerated from the final test run.
3. Every lane assigns its local outcome with a literal. Most are consistent
   with current observations, but the CPO lane says `consistent` despite its
   own audit finding the exact constrained-RLHF objective unbounded in all 135
   positive-gamma cases. Root cause: outcome labels are not derived from
   recomputed observations.
4. `sources/paper.json` records a false title and `CC-BY-4.0`. The pinned
   arXiv v1 primary record instead gives title
   `Conditional Equivalence of DPO and RLHF: Implicit Assumption, Failure Modes, and Provable Alignment`
   and `arXiv.org perpetual non-exclusive license`. Root cause: challenge
   metadata was copied into primary-source provenance and the license was
   guessed. Also add the pinned HTML URL
   `https://arxiv.org/html/2605.20834v1`.

The exact finite reduction in the approved design correctly exposes the paper
Equation 10 objective: positive `gamma * delta` is unbounded as the preferred
probability approaches one, while Equation 17's held-fixed reference-margin
population loss still has the separately recomputed stationary identity. The
CPO claim therefore has separable supporting and contradicting observations
and must derive `mixed`, not `consistent`.

## Mandatory red-green sequence

For each defect, write the smallest regression test first and run it alone.
Capture that it fails for the expected reason before changing production data
or code. Then make the minimal fix and rerun the focused test and the full
project suite.

Required new behavior:

1. Runtime validation actually reads the supplied schema and rejects:
   an unknown top-level field, an unknown source field, an unknown claim field,
   and an unknown nested lane-detail/case field. It also rejects a missing
   required field, the wrong type (including booleans where numbers are
   expected), non-finite values, wrong identities, wrong claim order/hash, and
   invalid outcomes. The committed schema must set
   `additionalProperties: false` on every project-owned object, including the
   source object and all five lane result/case/witness shapes. A small
   standard-library recursive validator for the schema subset used here is
   acceptable; do not add a runtime dependency.
2. Each targeted lane outcome is derived from the computed aggregate
   conditions rather than assigned unconditionally. Add direct tests for the
   derivation helpers, including a CPO result with 45 certified gamma-zero
   optima, 135 positive-gamma unbounded cases, and passing Equation 17 checks
   deriving `mixed`. Preserve the exact computed observations.
3. Correct the pinned primary-source title, license, and HTML URL and add exact
   immutable regression assertions. Do not vendor the paper.
4. Regenerate `evidence.json` twice and require byte identity. Confirm the CPO
   claim is `mixed`; regenerate presentation-derived values if necessary.
   Recompute `evidence/validation.json` only after the final full project test
   run, with the exact final passed-test count and evidence SHA-256.

## Required verification before proposal return

Run at least:

```bash
env UV_CACHE_DIR=/tmp/conditional-dpo-uv-cache uv run --project submissions/conditional-equivalence-of-dpo-and-rlhf-assumptions-failure-modes-and-provable-alignment python -m pytest submissions/conditional-equivalence-of-dpo-and-rlhf-assumptions-failure-modes-and-provable-alignment/tests -q
env UV_CACHE_DIR=/tmp/conditional-dpo-uv-cache uv run --project submissions/conditional-equivalence-of-dpo-and-rlhf-assumptions-failure-modes-and-provable-alignment conditional-dpo-repro generate --project-root submissions/conditional-equivalence-of-dpo-and-rlhf-assumptions-failure-modes-and-provable-alignment --output /tmp/conditional-dpo-corrected-1.json
env UV_CACHE_DIR=/tmp/conditional-dpo-uv-cache uv run --project submissions/conditional-equivalence-of-dpo-and-rlhf-assumptions-failure-modes-and-provable-alignment conditional-dpo-repro generate --project-root submissions/conditional-equivalence-of-dpo-and-rlhf-assumptions-failure-modes-and-provable-alignment --output /tmp/conditional-dpo-corrected-2.json
cmp /tmp/conditional-dpo-corrected-1.json /tmp/conditional-dpo-corrected-2.json
git diff --check
```

Commit only the paper project. No controller validation, deployment,
submission, polling, verdict import, branch integration, or authority claim is
authorized.

## Controller review finding after proposal `44d8cfc`

The first correction proposal closes the original four findings, but controller
review rejects validation for one exact stale presentation value:

- `pages/index.md` says the 180-case CPO grid uses
  `gamma in {0, 0.05, 0.1, 0.25}`.
- `src/conditional_dpo_repro/grids.py` and the committed `evidence.json`
  actually use `gamma in {0, 0.01, 0.05, 0.1}`.

Write a failing page regression test first that binds the judge-readable
summary to the executable four-value gamma grid. Observe RED, replace only the
stale summary values, observe GREEN, regenerate evidence only if its bytes
would change, run the full project suite and `git diff --check`, and commit
only the paper project. Return the new commit as another proposal.
