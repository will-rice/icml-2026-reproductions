---
title: AGoQ Evidence Audit
emoji: 🔎
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 4.44.1
python_version: "3.12"
app_file: app.py
pinned: false
tags:
  - paper-ymHDVBwmta
  - icml2026-repro
---

# AGoQ pinned evidence audit

This project audits AGoQ from immutable primary inputs. It uses paper
`arXiv:2605.00539v2` and official repository
`Eutenacity/AGoQ@006fa0f6318228d1fcd6727f0578c0e548e5cbff`. Ten selected upstream
files are checked byte-for-byte by SHA-256 and Git blob ID before any source
observation is accepted.

## What was reproduced

`paper_context` contains transcribed paper statements. Only
`reproduced_observations` contains code-computed results:

- Exact Table 1 component sums are BF16 `28U`, COAT `33/2U` (`16.5U`), and
  AGoQ `31/4U` (`7.75U`), where `U = B*S*H*2 bytes`.
- The printed four-stage equation yields stored-batch counts `(11, 9, 7, 5)`
  in device order and exact bit widths `(4, 44/9, 44/7, 44/5)`.
- The paper-reported integer widths `(4, 5, 6, 8)` produce storage products
  `(44, 45, 42, 40)`. The second stage overshoots the nominal 44-unit target
  by one; the release does not specify the integer rounding policy.
- Pinned source establishes activation quantization integration, local
  dequantize/accumulate/requantize, and the
  All-to-All → local reduce → AllGather path.
- Transformer Engine changes contain quantize/dequantize call sites next to
  GEMM calls, but the selected source does not contain the fused GPU kernel
  implementation body or reproduce its overhead reduction.

## Six live claims

Claims 1–4 are `partial`. They are supported respectively by released-source
mechanism tracing, exact Table 1 arithmetic, pipeline-equation auditing, and
call-site tracing with an explicit missing-kernel limitation.

Claims 5 and 6 are `unavailable`. Table 2 requires 64 GPUs, and Table 3 requires
16 NVIDIA Blackwell GPUs. No distributed training, throughput, convergence, or
memory benchmark was run. Paper-reported values are not presented as reproduced
measurements.

## Reproduce offline

```bash
uv run pytest -q
uv run python generate_evidence.py --output evidence.json
uv run python app.py
```

The evidence generator performs no network or Git calls. To reacquire the
selected upstream bytes separately:

```bash
uv run python scripts/acquire_upstream.py \
  --repository-url https://github.com/Eutenacity/AGoQ.git \
  --revision 006fa0f6318228d1fcd6727f0578c0e548e5cbff \
  --manifest evidence/inputs/upstream_manifest.json \
  --output evidence/inputs/upstream
```

This repository is a worker evidence proposal, not a controller validation,
deployment, challenge submission, or official verdict.
