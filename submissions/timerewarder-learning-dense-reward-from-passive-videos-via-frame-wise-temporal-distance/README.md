---
title: TimeRewarder Reproduction Evidence
emoji: ⏱️
colorFrom: blue
colorTo: yellow
sdk: gradio
sdk_version: 6.0.1
python_version: "3.12"
app_file: app.py
tags:
  - paper-XztRm216YS
  - icml2026-repro
---

# TimeRewarder reproduction evidence

This project recomputes independently executable evidence from pinned released
artifacts. Paper-reported values are not presented as reproduced measurements.

Measurement SHA-256:
`c1a47c914a17101941a81d7de4422cd5b88ab3c74cc5822bcc9bce041ec2f305`.

## Claim outcomes

- Claim 1 — `verified`: 10/10 released-checkpoint task strata passed the fixed five-video-per-task temporal-distance protocol. Limitation: This verifies released-checkpoint behavior and the pinned action-free label path; it does not reproduce reward-model training.
- Claim 2 — `verified`: All 33 pinned source spans and 106 temporal-distance plus three transition formula cases passed. Limitation: The source/formula audit verifies the implementation path, not paper-scale downstream RL outcomes.
- Claim 3 — `verified`: All finite Bellman recurrences and the gamma-one temporal-distance identity passed under the enumerated assumptions. Limitation: The derivation assumes full observability, deterministic transitions, an optimal trajectory, a terminal goal, and unaliased observations.
- Claim 4 — `partial`: All 50 released-model videos produced finite VOC; the fixed five-video-per-task mean was 0.998. Limitation: The comparative highest-VOC component is unavailable: baseline predictions/checkpoints and the paper's full Figure 3 protocol were not released. This is a five-video-per-task released-model protocol.
- Claim 5 — `unavailable`: No matched successful/failed rollout comparison was computed. Limitation: Released successful/failed rollout videos and matched VIP, Rank2Reward, and PROGRESSOR predictions are unavailable.
- Claim 6 — `unavailable`: No Meta-World or DrQ-v2 training was run. Limitation: The CPU scope performs no Meta-World/DrQ-v2 training, multi-seed evaluation, or 200,000-interaction budget.

The Figure 3 result is explicitly a five-video-per-task released-model
protocol, not the full comparative paper protocol. The small learned fixture is
diagnostic-only.

## Reproduce

Acquire and verify pinned inputs, convert one legacy file, and independently
review it:

```bash
uv run timerewarder-repro acquire --manifest artifacts/manifest.json --cache-dir .cache
uv run timerewarder-repro convert --task TASK --registry artifacts/checkpoints.json --cache-dir .cache --output-dir .cache/converted --converter ID
uv run timerewarder-repro review-conversion --task TASK --registry artifacts/checkpoints.json --receipt PATH --output PATH --reviewer ID --approval PATH
```

Run the released-model protocol and rebuild evidence:

```bash
uv run timerewarder-repro representative --registry artifacts/checkpoints.json --dataset-manifest artifacts/dataset-manifest.json --schema artifacts/model-schema.json --cache-dir .cache --output artifacts/representative.json
uv run timerewarder-repro build-evidence --manifest artifacts/manifest.json --acquisition artifacts/acquisition.json --registry artifacts/checkpoints.json --source-root artifacts/source --representative artifacts/representative.json --output artifacts/evidence.json
uv run timerewarder-repro fixture
uv run pytest -q
```

No deployment or official verdict is included in this proposal.
