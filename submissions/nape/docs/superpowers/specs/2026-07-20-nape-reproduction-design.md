# NAPE Challenge Reproduction Design

## Goal

Produce a public ICML 2026 Agent Reproducibility Challenge logbook for
"A Benchmark and Framework for Evaluating Next Action Predictions in
Spreadsheets" that gives independently reproducible verdicts for both selected
claims. The submission targets four judged points without paid model inference.

## Claims and Verdict Standard

### Claim 1

> Benchmark generates 58 symbolic action sequences consisting of 13K actions
> from publicly available spreadsheets.

Audit the artifacts released by the paper authors rather than estimating from
the prose. Count trajectory files and their actions in the official NAPE GitHub
repository, then independently inspect the linked `Tej-a55/napeval` Hugging
Face dataset. Record immutable revisions where available and distinguish a
paper claim from the contents of the released reproduction artifacts.

The verdict is:

- verified only if the released artifacts contain 58 sequences and support the
  stated approximately 13,000 actions;
- falsified if the official released artifacts consistently contain a
  materially different number of sequences or actions;
- inconclusive if the linked sources disagree or required data are unavailable.

### Claim 2

> Online evaluation methodology predicts actions after each user action,
> accepts or rejects prediction, and updates future actions.

Exercise the official evaluator with a deterministic fixture that covers the
online state transition: prediction after an observed action, acceptance or
rejection against the trajectory, and the resulting next prediction point.
Retain the upstream test suite as broader corroborating evidence. No LLM solver
is required because this claim concerns evaluator behavior, not model quality.

The verdict is verified only when the trace demonstrates all stated transitions
and the relevant official tests pass.

## Components

### Reproduction Package

Add a small `src` package with public functions for inspecting NAPE trajectory
JSON and summarizing the released benchmark. The package returns structured
Python values; a command entry point serializes the evidence to JSON and logs a
concise human-readable summary. Paths to the vendored upstream repository and
the output directory remain repository constants because this submission has a
single fixed target.

Use `will-rice/agent-harness-template` as the structural base and the shared
tooling conventions from `will-rice/ml-template`: `uv_build`, src layout,
Google-style docstrings, strict Ruff rules, `ty`, pytest, the v6 pre-commit
hooks, Apache-2.0 licensing, and the Python package CI workflow. Do not inherit
unrelated model-training dependencies such as Torch, Lightning, or W&B.

### Tests

Use functional pytest tests with minimal trajectory fixtures. Tests define the
expected counting behavior and evidence schema before implementation. A
separate integration test exercises the official online evaluator using a tiny
fixture and checks each transition named in claim 2.

### Evidence Bundle

The reproduction command writes a bundle containing:

- `claim_1_audit.json` with source revisions, sequence count, action count,
  counting definition, and comparison with the claimed values;
- `claim_2_trace.json` with observed actions, predictions, acceptance decisions,
  and updated evaluator positions;
- a concise environment manifest and exact commands needed to rerun the checks.

Generated evidence is deterministic and contains no credentials or machine-
specific absolute paths.

### Trackio Logbook

Keep the challenge-generated structure and populate its four pages:

- Executive summary: outcome first, both verdicts, compute, wall time, and cost.
- Claim 1: source revisions, method, observed counts, limitations, and verdict.
- Claim 2: evaluator trace, upstream `244 passed` run, limitations, and verdict.
- Conclusion: reproduction commands and attached evidence bundle.

Command outputs are added through `trackio logbook run`; the final bundle is
attached as a Trackio artifact. The generated logbook assets are not manually
restyled.

## Data Flow

1. Read the pinned official NAPE checkout and linked Hugging Face dataset
   metadata.
2. Parse trajectory JSON with structured JSON APIs and calculate claim 1
   evidence.
3. Run the deterministic official-evaluator scenario and capture claim 2 state
   transitions.
4. Serialize both results into the evidence bundle.
5. Run tests and reproduction commands through Trackio so outputs become part
   of the corresponding claim pages.
6. Validate the logbook with the challenge validator.
7. Publish to
   `wrice/repro-a-benchmark-and-framework-for-evaluating-next-action-predictions-in-spreadsheets`.
8. Confirm the Space is running, inspect logs, and load the public logbook.

## Error Handling

Malformed trajectories, missing required fields, missing official data, or
unexpected evaluator behavior fail with concise errors. The audit does not use
fallback counts or silently substitute another dataset. Network-derived
metadata is cached in the evidence bundle with its source URL and revision so a
reviewer can distinguish live lookups from local deterministic checks.

## Verification

Implementation is complete only after all of the following pass in a fresh run:

- focused reproduction tests;
- the full upstream NAPE test suite;
- the deterministic evidence-generation command;
- `uv run pre-commit run -a`;
- `uv run ty check`;
- the official challenge logbook validator;
- Hugging Face Space runtime and public-page smoke checks.

The expected score is four points if judges accept a verified or falsified
verdict for each claim. The design does not treat publication itself as evidence
that either verdict is correct.
