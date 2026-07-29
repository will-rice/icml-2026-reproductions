# PipeSD Reproduction Plan

## Overview

- **Paper Title**: PipeSD: An Efficient Cloud-Edge Collaborative Pipeline Inference Framework with Speculative Decoding
- **Paper ID**: `1ebAvNphi7`
- **Submission Directory**: `submissions/pipesd-an-efficient-cloud-edge-collaborative-pipeline-inference-framework-with-speculative-decoding`
- **Upstream Revision**: `arxiv:2502.16480v1+github:ecpiping/PipeSD@871e89d078202c7d9d18d0924bd76cf161cd6606`

## Target Claims

1. **Token-Batch Pipeline Scheduling (Section 3.2)**: PipeSD overlaps draft-token generation and communication using token-batch pipeline scheduling optimized by dynamic programming.
2. **Dual-Threshold NAV Triggering (Section 3.3)**: PipeSD uses a dual-threshold NAV triggering mechanism that jointly considers cumulative sequence confidence and single-token confidence.

## Implementation Architecture

- `pipesd/scheduler.py`: Dynamic programming solver for optimal token-batch pipeline scheduling, balancing draft generation latency against transmission overhead across edge-cloud stages.
- `pipesd/nav_trigger.py`: Dual-threshold Non-Acceptance Verification (NAV) triggering mechanism taking cumulative sequence confidence and single-token confidence thresholds.
- `pipesd/eval.py`: Benchmark suite running deterministic CPU verification of DP pipeline schedule optimality and dual-threshold NAV decision boundaries.
- `tests/test_pipesd.py`: Automated pytest test suite verifying core contracts and edge cases.

## Validation & Deliverables

- `validation-manifest.json` under `submissions/pipesd-an-efficient-cloud-edge-collaborative-pipeline-inference-framework-with-speculative-decoding`
- Self-contained python implementation with 0 external paid API dependencies
- Clean, reproducible unit tests passing `pytest`
