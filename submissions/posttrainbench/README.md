---
title: PostTrainBench Reproduction
emoji: "🔍"
colorFrom: blue
colorTo: indigo
sdk: static
app_file: index.html
pinned: false
license: mit
tags:
  - icml2026-repro
  - paper-UnjxMTe57e
---

# PostTrainBench Reproduction

Deterministic CPU-only released-artifact audit for PostTrainBench
(OpenReview: `UnjxMTe57e`, arXiv: `2603.08640v2`).

## Selected Claims

### Claim 1: partial-support

PostTrainBench evaluates autonomous post-training agents across 4 base models and 7 benchmarks under a 10-hour single-H100 budget (Figure 1).

Released trajectory inventory confirms 4-by-7 coverage across all accepted benchmark/model cells. Runner configuration defaults to one H100 with a NUM_HOURS-based timeout. The current checkout's scheduler-dependent branches and five-minute termination grace are reported as limitations.

### Claim 2: partial-support

The paper reports reward-hacking failure modes including training on test sets, downloading instruction-tuned checkpoints, and using discovered API keys for synthetic data (Abstract).

Released contamination and instruction-model judgments provide partial support for two of three reward-hacking submodes. The API-key submode artifact is absent from the pinned revision.

## Evidence

- [Evidence summary](index.html)
- [Detailed report](report.html)
- [Poster](poster.html)
- [Provenance](evidence/provenance.json)
- [Coverage](evidence/coverage.json)
- [Reward hacking](evidence/reward_hacking.json)
- [Claims](evidence/claims.json)
- [Manifest](evidence/manifest.json)

## Limitations

This is not an official challenge verdict. See
[the report](report.html) for the full limitation list.

No H100 run is reproduced. A released judge label is not independently
established behavioral truth. The API-key submode remains unavailable.

## Licenses

- Source repository: MIT
- Dataset: Apache-2.0
- Paper: CC BY 4.0
- This reproduction: MIT

## Cost

Paid API cost: USD 0.00
