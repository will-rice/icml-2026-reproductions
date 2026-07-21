# NAPE ICML 2026 Reproduction

> **Archive notice:** This parent copy is an immutable convenience snapshot and
> has no nested Git metadata. The setup, submodule, and test commands below
> apply only to the [canonical NAPE repository](https://github.com/will-rice/icml-2026-repro).
> See the [parent workspace README](../../README.md) and
> [upstream provenance](UPSTREAM.md) for this archive's boundary and revision.

This repository audits the first three claims presented by the ICML 2026 Agent
Reproducibility Challenge for [A Benchmark and Framework for Evaluating Next
Action Predictions in Spreadsheets](https://arxiv.org/abs/2606.13802). It
packages portable evidence from the pinned NAPE release and official evaluator
without paid model inference.

## Results

- **Claim 1: reproduced.** The release contains 52 trajectories and 11,907
  operations, with sequence lengths of 35-821, a paper-rounded mean of 229,
  and a median of 164. The audit also verifies the corresponding raw artifacts
  and released construction-pipeline source.
- **Claim 2: reproduced from released outputs.** Auditing the 52 released oracle
  outputs gives 68.04% weighted property coverage. The original paid
  frontier-model oracle calls were not rerun.
- **Claim 3: reproduced.** Across one deterministic adaptation case per each
  of the 52 released trajectories, the official
  evaluator removes satisfied future operations in 50/52 cases, inserts inverse
  operations in 52/52 cases, and preserves the target in 52/52 cases. One
  residual fixture separately exercises residual correction and preserves its
  target. This is not a full per-action model rollout.

Claims 4-6 were not replicated because named model outputs and paid API budget
were not available. These results do not benchmark model quality or reproduce
model-quality tables. The run is CPU-only and uses no paid API calls; paid API
cost is `$0.00`.

## Prerequisites

- Python 3.13 or newer
- [`uv`](https://docs.astral.sh/uv/)
- The repository checkout, including the pinned `external/NAPE` submodule

Install the locked environment with:

```bash
uv sync
```

## Reproduce

Generate the six-artifact evidence bundle with:

```bash
uv run nape-repro
```

The command writes Claim 1 benchmark evidence, Claim 2 predictability evidence,
Claim 3 future-adaptation evidence, explicit Claims 4-6 status, an environment
manifest, and an artifact guide under `repro_bundle/`. The bundle contains no
credentials, access tokens, or machine-specific absolute paths.

Run the focused evidence and CLI tests with:

```bash
uv run pytest tests/test_cli.py -q
uv run pytest tests/test_benchmark_evidence.py tests/test_predictability_evidence.py tests/test_future_adaptation_evidence.py -q
```

Run the full root suite and official upstream tests with:

```bash
uv run pytest -q
uv run pytest external/NAPE/tests -q
```

The upstream suite is expected to report 244 passing tests. Formatting and type
checks are:

```bash
uv run pre-commit run -a
uv run ty check src tests
git diff --check
```

## OpenResearch Provenance

Both experiment nodes use the same fixed command:

```bash
git submodule update --init --recursive && uv run nape-repro && cp repro_bundle/README.md EVAL.md && uv run pytest -q
```

| Experiment | Git branch | Commit | Local run | Result |
| --- | --- | --- | --- | --- |
| Previously judged baseline | [`orx/public-scored-submission-baseline`](https://github.com/will-rice/icml-2026-repro/tree/orx/public-scored-submission-baseline) | `e302a794359812e060b52673517db5f79103191c` | `c9e2273a-1c61-4c58-8cef-af044239950b` | 37 passed |
| Judge-aligned Claims 1-3 evidence | [`orx/judge-aligned-claims-1-3-evidence`](https://github.com/will-rice/icml-2026-repro/tree/orx/judge-aligned-claims-1-3-evidence) | `20864f5e844e63cccb80608f26113aec4bd8e0e1` | `f4b19198-1047-4cc9-a551-eddb7e33e1be` | 119 passed |

The local OpenResearch project id is
`d039cdbd-9330-4578-a879-2bd753a4dfd2`; the child experiment is
`4242cbfc-ab63-46fe-933a-f18484160e6c`.

## Challenge Validation

For the ICML 2026 Agent Reproducibility Challenge, download the current
`scripts/validate_icml_logbook.py` from the target Space and run:

```bash
uv run python scripts/validate_icml_logbook.py
```

The target Space is [`wrice/repro-a-benchmark-and-framework-for-evaluating-next-action-predictions-in-spreadsheets`](https://huggingface.co/spaces/wrice/repro-a-benchmark-and-framework-for-evaluating-next-action-predictions-in-spreadsheets).

## Pinned Sources

- [NAPE GitHub revision `ac0d10e4`](https://github.com/Tej-55/NAPE/tree/ac0d10e4dc345f982a5665a2c4bdb6b752d663f2)
- [NAPEval dataset revision `c7e28fe9`](https://huggingface.co/datasets/Tej-a55/napeval/tree/c7e28fe9b08ee2c0bfc429519cf100197b7e018c)
- [Paper: arXiv `2606.13802`](https://arxiv.org/abs/2606.13802)
