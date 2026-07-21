# NAPE Score Improvement Design

## Objective

Update the canonical NAPE reproduction Space to address the six claims used by
the automated Logbook Judge. The current submission scored 1/12 because it was
organized around two challenge-card claims rather than the judge's anchored
paper claims.

The revision uses only released artifacts and CPU execution. It targets complete
evidence for Claims 1-3 and explicitly marks Claims 4-6 as unreplicated. A score
increase is expected but not guaranteed because the judge decides whether an
audit of released oracle outputs constitutes a full reproduction.

## Evidence Strategy

### Claim 1: Benchmark and construction pipeline

Audit all 52 pinned trajectory files and their matching raw-artifact directories.
Compute the operation-count distribution directly from the trajectory JSON:

- trajectory count;
- total operations;
- minimum and maximum sequence length;
- arithmetic mean; and
- median.

Require exact agreement with the paper values: 52 trajectories, 11,907
operations, range 35-821, mean 229 after paper-style rounding, and median 164.
Verify one-to-one trajectory IDs across `data/trajectories` and `data/raw`, and
require every raw directory to contain the released workbook, sheet image,
operation sequence, and predictability output.

Audit the pinned source tree for executable implementations of the symbolic
sequencing, region annotation, and LLM refinement stages. Report human annotation
as release provenance rather than claiming it can be independently rerun from
the repository.

### Claim 2: Empirical predictability ceiling

Parse all 52 pinned `predictable_state.json` files and validate their schema and
internal arithmetic. Recompute:

- total predictable properties;
- total final-state properties;
- weighted global coverage;
- per-trajectory mean and median coverage; and
- number of trajectories above 50% coverage.

Expected released-result audit: 126,940 / 186,574 = 68.04%, mean 65.99%, median
66.34%, and 44/52 trajectories above 50%.

The report will state precisely that this independently recomputes the published
aggregate from released oracle outputs; it does not repeat the original paid
frontier-model generations.

### Claim 3: Online future adaptation

Exercise the pinned official `FutureEditsManager`, state builder, comparator, and
orchestrator against every released trajectory. Produce aggregate evidence that
separately demonstrates:

- removal of already-satisfied future operations;
- insertion of inverse operations for false positives;
- synthesis of residual corrections by the final patch step; and
- equality between the adapted final state and the original target state.

Use deterministic prediction mutations derived from each trajectory, with no
model calls. Cases that cannot express a mechanism must be reported as skipped
with a reason; the evidence must include denominators and may only claim a
mechanism when at least one valid case executes and reaches the target state.

Retain a small orchestrator trace for prediction ordering and acceptance/rejection,
but do not present that fixture as the primary scale evidence.

### Claims 4-6: Model-performance results

Do not fabricate or proxy the named GPT and SmolLM results. Add explicit pages
that mark these claims unreplicated, explain the missing model outputs/API budget,
and provide exact commands/configuration prerequisites for a future run.

## Code and Artifact Structure

Add focused public audit interfaces under `src/icml_2026_repro` for dataset
statistics, predictability aggregation, and full-release future-adaptation
evidence. Extend `nape-repro` to write one JSON artifact per judge claim plus an
updated environment manifest and artifact guide.

All evidence must include the pinned NAPE revision, input paths relative to the
repository, counting definitions, observed values, and an evidence-scope field
that distinguishes recomputation from provenance-only statements.

## Logbook Revision

Replace the two-claim organization with six claim pages matching the judge's
claim text. Claims 1-3 will include fresh Trackio command cells and concise
outcome-first interpretation. Claims 4-6 will state `not replicated` without
proposing unsupported verdicts.

Update the executive summary, conclusion, and poster to show:

- current score: 1/12;
- evidence added for Claims 1-3;
- exact full-release statistics;
- zero paid API cost; and
- explicit limitations for released-output audits and unrun model benchmarks.

Publish over the existing canonical Space so the first judged logbook remains
the scoring entry.

## Validation and Failure Handling

Every parser fails on malformed schema, duplicate/missing trajectory IDs,
inconsistent counts, dirty or mismatched NAPE checkout, non-finite metrics, or
an adapted sequence that does not reach the target state.

Tests cover known aggregate values, malformed inputs, mechanism-specific future
updates, target-state preservation, deterministic reruns, and portable bundle
serialization. Before publication, run root tests, all 244 upstream tests,
pre-commit, type checking, bundle regeneration, the official logbook validator,
and local page smoke tests.

After publication, verify the canonical Space SHA and public content. Rejudging
may be asynchronous; check the public verdict dataset for the new SHA rather
than assuming publication immediately changes the score.
