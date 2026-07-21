# Evidence And Submission Checklist

Use this checklist for every paper. A checked item needs an artifact, command result, or live observation; intention is not evidence.

## Upstream And Claims

- [ ] Record the exact paper identifier and revision.
- [ ] Pin every upstream repository to an exact commit SHA, not a branch or mutable tag.
- [ ] Record dataset, checkpoint, prediction, archive, and release identifiers plus cryptographic hashes for downloaded files.
- [ ] Record source URLs, acquisition commands, licenses, and which upstream artifact supports each target claim.
- [ ] Define each claim's expected observation and independent test before implementation.

## Evidence Implementation

- [ ] Obtain approval for the paper-specific design before writing implementation code.
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

- [ ] Immediately before submitting, refresh the live paper, claim, queue, judging, and prior-verdict state.
- [ ] Stop if the paper became claimed, queued, judging, or otherwise ineligible.
- [ ] Submit the verified Space revision and record the submission ID, Space ID, deployed SHA, and timestamp.
- [ ] Refresh live challenge state after submission and verify the submission appears in the expected state. Do not infer acceptance from the submit request alone.

## Verdict Handling

- [ ] Before polling, define a finite deadline or maximum attempts and a bounded interval or backoff.
- [ ] Persist every poll time, status, and external ID with same-phase state updates. On timeout or outage, keep judging pending, record the next check, and do not claim completion.
- [ ] Store the received verdict and a result for every claim: `verified`, `partial`, `inconclusive`, `contradicted`, or `unavailable`.
- [ ] Preserve judge details and distinguish challenge verdicts from the reproduction's own measurements.
- [ ] Extract a concrete selection or evidence lesson for future candidates.

## Improvement And Completion

- [ ] Improve only the current paper, at most once, by transitioning `judging` -> `improving` when its concrete verdict defect is fixable within the CPU, USD 10, licensing, and safety gates.
- [ ] Keep the attempt evidence-focused; do not broaden it into new training or an unrelated reproduction.
- [ ] Re-run test-first evidence, local validation, exact-SHA deployment verification, live submission verification, and bounded verdict polling after the change.
- [ ] Record both verdicts and the improvement count. If no eligible fix exists or one attempt has already occurred, record the lesson and stop improving. Never reselect a judged or historical paper.
- [ ] Mark the paper complete only after a verdict is received and all claim-level outcomes are recorded. Improvement must occur before `complete` -> `idle` archives the paper; deployment or submission alone never completes the loop.
