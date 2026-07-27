---
title: Graph Dataset Pruning Formal Evidence
sdk: gradio
sdk_version: 6.20.0
app_file: app.py
tags:
  - icml2026-repro
  - paper-a3GdvuPItd
---

# Formal evidence for graph-based dataset pruning

This independent reproduction audits two target claims from “Selecting
Samples on Graphs: A Unified Dataset Pruning Framework for Lossless Training
Acceleration”: the graph/maximum-weight-clique formulation and the claimed
submodular greedy guarantee. It recomputes formal and bounded finite evidence
from pinned textual transcriptions; it does not report the paper's experimental
tables as reproduced results.

The accepted result has an important literal/repaired boundary. The
sample-wise objective as written double-counts symmetric interactions relative
to the maximum-weight-clique objective, and the Appendix E inline shift yields
a cardinality-quadratic set function whose marginal can increase. A separate
`modular_shift_candidate` is audited as a repair candidate, not attributed to
the paper. The rendered [report](report.md) and [poster](poster.html) lead with
the claim-level results and give an RFC 6901 evidence pointer beside every
selected displayed value.

## Reproduce and validate

From this directory, with `uv` installed:

```bash
uv sync --frozen
uv run --frozen python -m pytest -q
EVIDENCE_SOURCE_REVISION="$(
  uv run --frozen python -c \
    'import json; from pathlib import Path; print(json.loads(Path("evidence/evidence.json").read_text())["source_revision"])'
)"
uv run --frozen python -m graph_pruning_repro.cli recompute \
  evidence --source-revision "$EVIDENCE_SOURCE_REVISION"
uv run --frozen python -m graph_pruning_repro.cli validate \
  evidence/evidence.json
uv run --frozen python -m graph_pruning_repro.cli render \
  evidence/evidence.json .
cp poster.html poster_embed.html
```

The canonical source is
<https://arxiv.org/pdf/2606.12913v2>, pinned as `arxiv:2606.12913v2`;
the PDF digest and the reproduction source revision are recorded in
`evidence/evidence.json`. The evidence generator uses exact rational
arithmetic. This boundary matters: bounded enumeration can refute a universal
claim by producing a counterexample, but bounded enumeration cannot prove an
arbitrary-real universal claim.

## Unavailable empirical claims

The paper-reported CIFAR-10/100, ImageNet-1k, detection, segmentation,
accuracy, training-time, and acceleration results are unavailable as
reproduced measurements. No released implementation was used to resolve the
edge-counting or shift ambiguities. See `/unavailable_claims` in
`evidence/evidence.json`; unavailable items remain visibly separated from the
formal evidence in every rendered artifact.

## Provenance and licenses

`NOTICE.md` preserves attribution to all seven paper authors and identifies
the pinned source. Original executable code, tests, and schema are distributed
under `LICENSE` (MIT). Paper transcriptions, evidence, report, and poster text
are distributed under
`LICENSES/CC-BY-NC-SA-4.0.txt`. The original PDF, its figures and tables, and
unreleased implementation code are not redistributed.
