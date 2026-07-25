# Evidence And Submission Checklist

Use this checklist for every paper. A checked item needs an artifact, command result, or live observation; intention is not evidence.

## Upstream And Claims

- [ ] Record the exact paper identifier and revision.
- [ ] Pin every upstream repository to an exact commit SHA, not a branch or mutable tag.
- [ ] Record dataset, checkpoint, prediction, archive, and release identifiers plus cryptographic hashes for downloaded files.
- [ ] Record source URLs, acquisition commands, licenses, and which upstream artifact supports each target claim.
- [ ] Define each claim's expected observation and independent test before implementation.

## Evidence Implementation

- [ ] Persist a paper-specific design and author through fenced `record-design`. Record approval by a different reviewer through `review-design`; rejection revises only that attempt.
- [ ] Write each evidence test first and observe the expected failure before implementing it.
- [ ] Keep inputs, code-computed outputs, and paper-reported context distinguishable. Never label paper-reported values as reproduced; only code-computed outputs can support reproduction.
- [ ] Emit deterministic machine-readable claim results, such as JSON or CSV, with claim IDs, observations, tolerances, provenance, and status.
- [ ] Mark inaccessible artifacts or untestable claims explicitly as `unavailable`; do not replace them with README values, screenshots, or assertions.
- [ ] Keep evidence generation independently executable from a clean environment.

## Local Validation

- [ ] Re-run the complete evidence pipeline from pinned inputs.
- [ ] Confirm machine-readable outputs parse and agree with the human-readable report.
- [ ] Run the submission project's full pytest suite with no failures.
- [ ] Run root `uv run pytest -q` and `uv run pre-commit run --all-files` cleanly.
- [ ] Review the diff for credentials, mutable URLs, generated caches, unrelated changes, and uncommitted work.

## Space Deployment

- [ ] Use a separate Hugging Face Space for this paper; do not add it to another paper's Space.
- [ ] Commit the exact validated source and evidence configuration before deployment.
- [ ] Record the local source commit and the Space repository revision.
- [ ] Query the deployed Space after build and verify its exact SHA equals the intended deployed commit. A successful build or healthy UI alone is insufficient.
- [ ] Exercise the live Space's evidence path and verify its machine-readable output.

## Challenge Submission

- [ ] Immediately before submitting, run `refresh-live --assessments-json PATH` and record the assessment hash and immutable snapshot ID. If challenge revision drift aborts refresh, regenerate assessments from a new raw refresh. No other state command may access the network.
- [ ] Stop if the paper became claimed, queued, judging, or otherwise ineligible.
- [ ] Submit the verified Space revision and record the submission ID, Space ID, deployed SHA, and timestamp.
- [ ] Refresh live challenge state after submission and verify the submission appears in the expected state. Do not infer acceptance from the submit request alone.

## Verdict Handling

- [ ] Call `watch-attempt` with an explicit attempt, owner, fence, finite positive `poll_limit`, and aware `poll_deadline`.
- [ ] Persist each observation with fenced `record-poll`. At either limit
  without a verdict, persist the blocker and next action in that attempt shard,
  and let the scheduler refill its capacity.
- [ ] Store a verdict dictionary with a nonempty `claims` list. Every item must contain exactly nonempty `claim` and `status` fields; status is `verified`, `partial`, `inconclusive`, `contradicted`, or `unavailable`.
- [ ] Preserve judge details and distinguish challenge verdicts from the reproduction's own measurements.
- [ ] Extract a concrete selection or evidence lesson for future candidates.

## Improvement And Completion

- [ ] Improve only the current paper, at most once, by transitioning `judging` -> `improving` with a nonempty `improvement_reason` when its concrete verdict defect is fixable within the CPU, USD 10, licensing, and safety gates.
- [ ] Keep the attempt evidence-focused; do not broaden it into new training or an unrelated reproduction.
- [ ] Re-run test-first evidence, local validation, exact-SHA deployment verification, live submission verification, and bounded verdict polling after the change.
- [ ] Record both verdicts in the authoritative `verdicts` list with improvement attempt/reason metadata. Keep final `verdict` equal to the verdict payload in the last history record. If no eligible fix exists or one attempt has occurred, record the lesson and stop improving.
- [ ] Mark the paper complete only after a verdict is received and all claim-level outcomes are recorded. Improvement must occur before `complete` -> `idle` archives the paper; deployment or submission alone never completes the loop.
- [ ] A blocked attempt resumes only to its recorded `blocked_from` phase. Never autonomously abandon it; only an explicit user-directed `abandon=true` may archive/cost-account it to `idle`.

## State CLI Examples

Every attempt mutation below also takes `--owner OWNER --fencing-token TOKEN`.
Record a design before independent review:

```bash
uv run python skills/icml-repro-loop/scripts/state.py record-design state/repro-loop.json --attempt-id ATTEMPT --owner OWNER --fencing-token TOKEN --author AUTHOR --design-path PATH
uv run python skills/icml-repro-loop/scripts/state.py review-design state/repro-loop.json --attempt-id ATTEMPT --owner OWNER --fencing-token TOKEN --reviewer REVIEWER --decision approved
```

Start a bounded judgment:

```bash
uv run python skills/icml-repro-loop/scripts/state.py watch-attempt state/repro-loop.json --attempt-id ATTEMPT --owner OWNER --fencing-token TOKEN --poll-limit 12 --poll-deadline 2026-07-25T18:00:00+00:00
```

Record an exact-source verdict, then transition one fixable attempt:

```bash
uv run python skills/icml-repro-loop/scripts/state.py record-verdict state/repro-loop.json --attempt-id ATTEMPT --owner OWNER --fencing-token TOKEN --raw-verdict RAW_JSON --normalized-verdict VERDICT_JSON --source-revision EXACT_SHA
uv run python skills/icml-repro-loop/scripts/state.py transition-attempt state/repro-loop.json improving --attempt-id ATTEMPT --owner OWNER --fencing-token TOKEN --updates-json '{"improvement_reason":"Add missing claim-1 provenance"}'
```

Complete with the final exact-claim verdict:

```bash
uv run python skills/icml-repro-loop/scripts/state.py transition-attempt state/repro-loop.json complete --attempt-id ATTEMPT --owner OWNER --fencing-token TOKEN
```
