# Reproduction Design: How much can language models memorize?

Attempt: `8de87cc9-1d39-49a6-b552-4b4dd7e67e0e`
Paper: `bA6BgSbaUi`
Snapshot: `14c573bd1871a2e5f02ad8928e975ba055537c9a8648b5da1669769464aacdb4`
Upstream: `arxiv:2505.24832v3`

## Target Claims

1. GPT-style transformers trained on uniform random data show a memorization
   capacity plateau of about 3.6 bits per parameter.
2. Capacity estimates across model widths and depths support roughly linear
   bits-per-parameter scaling, with precision effects treated as paper context
   unless independently measured.

## Evidence Plan

Build a CPU-only toy reproduction of the paper's random-token memorization
protocol. Train tiny deterministic GPT-style causal transformers on seeded
uniform token sequences, compute train negative log likelihood in bits, and
derive memorized bits per parameter from the entropy reduction relative to the
uniform baseline. Sweep at least two model sizes and two dataset sizes so the
evidence can distinguish memorization-capacity behavior from a single run.

The evidence bundle must keep paper-reported 3.6 bits per parameter as context
only. Reproduced measurements come only from local training runs. Full
500K-to-1.5B parameter sweeps, text-data memorization, double-descent, and
membership-inference scaling are out of CPU scope and must remain unclaimed.

## Validation

The submission must provide deterministic tests for entropy accounting,
seeded sequence generation, model-size sweep coverage, evidence bundle schema,
and claim statuses. Controller validation will run evidence generation,
paper-scoped pytest, root pytest, skill validation, and pre-commit from a clean
worktree before deployment.
