# TIC-VLA: Decoupling Vision-Language Reasoning from Reactive Control Reproduction Design

## Overview
This document specifies the reproduction design and validation strategy for TIC-VLA (Paper ID `9wYjjPydfe`, Attempt `4fdf8ed9-ad12-4923-b6dc-b37239a7c9b4`).

## Claims Audited
1. **Decoupled Architecture**: ActionExpert decoupling of slow VLM reasoning and fast reactive control via KV-cache semantic features.
2. **DynaNav Suite**: Isaac Sim benchmark configuration schema and episode specifications.
3. **Quantitative Metrics**: Latency-consistent training success rate comparisons (47.06 vs 16.47).
4. **Real-world Performance**: RTX 4060 and Jetson Orin NX action and reasoning latency accounting.

## Verification Pipeline
- `generate_evidence.py` executes CPU-compatible numerical audits on ActionExpert interface fixtures and DynaNav benchmark schemas.
- Pytest suite `tests/test_tic_vla.py` verifies output bundle integrity and page rendering compliance.
