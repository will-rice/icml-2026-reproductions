# video-SALMONN S Reproduction Design

Attempt: `90bf5a14-ca7f-49d8-9085-a633e800b5ca`
Paper: `tJP3FxzSPs`
Title: video-SALMONN S: Memory-Enhanced Streaming Audio-Visual LLM
Owner: `agy-paper-owner-09`
Snapshot: `da6a4e094daf8c09c6b94948f6ffe8b2cfa3384f1ccaf4502b3d573704a7172a`

## Pinned Primary Artifacts

- Paper source: `arxiv:2510.11129`
- OpenReview: `https://openreview.net/forum?id=tJP3FxzSPs`
- Official repository / codebase: `https://github.com/bytedance/video-SALMONN-S`

## Evidence Plan

Implement a CPU-only evidence bundle for `video-SALMONN S` that:

1. Audits the core streaming memory layer (Test-Time Training / TTT layer) formulation and fast-weight update mechanisms for long audio-visual sequence streaming under fixed memory bounds.
2. Validates the two-stage training scheme logic and freezing of TTT parameters during scale-up.
3. Implements a synthetic test suite verifying TTT streaming memory update invariants, memory token reduction vs similarity merging, and long-range dependency prediction objectives.
4. Audits benchmark datasets (ELViM, LVBench) and reports verifiable vs toy/unavailable claims based on CPU-only execution limits.

## Claim Mapping

- Claim 1: toy/partial. Verify fixed memory budget streaming logic and 1 FPS frame sampling formulation for 3-hour video streaming.
- Claim 2: toy. Implement TTT layer fast-weight update logic and long-span prediction loss computation on synthetic sequence tensors.
- Claim 3: partial. Validate freezing TTT parameters during scale-up training stage 2.
- Claim 4: unavailable/toy. Full Qwen3-VL long-video benchmark performance evaluation requires GPU model inference; evaluate streaming vs non-streaming memory trade-offs on synthetic benchmarks.
- Claim 5: unavailable/toy. Absolute accuracy gain on ELViM over non-streaming baselines requires full LLM weights; verify metric calculation and evaluation setup.
- Claim 6: toy. Verify that TTT memory representations achieve <25% token footprint compared to similarity-based token merging under equivalent representation capacity.

## Tests

Write tests for:
- TTT fast-weight update tensor shapes and gradient steps;
- Memory token compression ratio calculation (TTT vs similarity merging);
- Streaming video frame chunking and memory state propagation;
- Schema validation and evidence logbook generation.

## Costs And Safety

No paid API calls or autonomous GPU work. All operations run CPU-only on workstation resources.
