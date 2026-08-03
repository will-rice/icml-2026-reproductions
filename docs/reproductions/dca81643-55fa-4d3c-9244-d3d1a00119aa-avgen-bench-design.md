# AVGen-Bench Reproduction Design

- Attempt: `dca81643-55fa-4d3c-9244-d3d1a00119aa`
- Paper: `aJdgt8xDMy`
- Snapshot: `41692f328d154e4fad790fb8c89aa276452ce49b8aaa18064abb9c47a897d622`
- Owner: `codex-paper-owner-05`
- Title: `AVGen-Bench: A Task-Driven Benchmark for Multi-Granular Evaluation of Text-to-Audio-Video Generation`

## Upstream Pins

- Paper: `arxiv:2604.08540`
- Official code: `github:microsoft/AVGen-Bench@1049eabac472d479fe5feeb1ee202961f8e0982a`
- Generated-output dataset: `hf-dataset:microsoft/AVGen-Bench@69eb2a20b2d47659be7cd40984baf02b7f2395a8`

## Scope

This reproduction is a CPU-only artifact audit. It does not regenerate T2AV videos, rerun heavyweight video/audio evaluators, or reuse paper-reported values as reproduced measurements. It recomputes what the released artifacts make directly checkable:

1. Prompt taxonomy, prompt counts, and prompt complexity from pinned prompt JSON files.
2. Evaluation module inventory and score aggregation structure from pinned source files.
3. Leaderboard arithmetic and failure-pattern summaries from the official released README table.
4. Availability checks for human-correlation and repeated-run artifacts.

Claims that depend on unavailable raw evaluation outputs, expert-human ratings, or repeated-run caches are marked `inconclusive` or `toy`, with the missing evidence recorded explicitly.

## Implementation Plan

- Create a standalone submission under `submissions/avgen-bench-a-task-driven-benchmark-for-multi-granular-evaluation-of-text-to-audio-video-generation`.
- Write tests first for prompt inventory, metric inventory, leaderboard parsing, claim binding, and unsupported-artifact handling.
- Implement `avgen_repro.evidence` with deterministic parsers for local fixture text plus network-backed fetches from pinned GitHub and HF dataset revisions.
- Generate `evidence/bundle.json` and `pages/report.md` with commands, pins, hashes, observations, and per-claim verdicts.
- Validate with the submission pytest suite, root quick validation where feasible, and `uv run pre-commit run -a`.

## Expected Evidence By Claim

- Claim 1: verify 11 released prompt categories and 235 prompt records; mark the separate "3 main domains" wording as only structurally supported by scoring/evaluation groups, not raw prompt metadata.
- Claim 2: verify metric modules and aggregate dimensions for visual quality, audio quality, AV sync, lip sync, text, face, music/pitch-related prompts, speech, low/high physics, and holistic alignment.
- Claim 3: verify the released benchmark-comparison artifact and broad metric inventory; do not independently re-evaluate prior benchmarks.
- Claim 4: recompute leaderboard-derived signs that models can have strong basic aesthetics and weaker fine-grained semantic dimensions, using the released README table as cached aggregate data.
- Claim 5: require expert-human judgment artifacts; if absent, mark inconclusive.
- Claim 6: require cached repeated-run/prompt-subset outputs; if absent, mark toy or inconclusive based on source-code availability only.

## Risks

- The HF dataset is about 20 GB, so evidence fetches must avoid downloading videos and only fetch prompt JSON, README, and repository metadata.
- Current HF Space creation quota is exhausted, so deployment may remain blocked after validation.
- Main checkout Git index is read-only in this environment; controller validation should use a clean `/tmp` checkout and a manifest, as in earlier attempts.
