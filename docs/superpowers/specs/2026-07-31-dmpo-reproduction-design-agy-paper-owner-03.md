# Enhancing Reasoning for Diffusion LLMs via Distribution Matching Policy Optimization Reproduction Design

## Authority, attempt, and phase

- Attempt: `449b17d7-04fc-49c1-8c64-7ff40151b7d9`
- Challenge paper: `09CSjVeDug`
- Author: `dmpo-paper-owner-author`
- Pinned paper: Yuchen Zhu et al., *Enhancing Reasoning for Diffusion LLMs via Distribution Matching Policy Optimization*, `arxiv:2510.08233v1`.
- License: Apache 2.0 / MIT compatible.
- Phase covered by this document: `design`.

## Target claims and verdict boundaries

The reproduction evaluates the following testable target claims:

1. `DMPO fine-tunes diffusion LLMs by matching the model policy distribution to an optimal reward-tilted distribution through cross-entropy optimization (Section 3).`
2. `The method introduces weight baseline subtraction to make small-batch DMPO training effective (Section 3.4).`
3. `DMPO is trained without supervised fine-tuning in an R1-Zero-like recipe for reasoning tasks (Section 4).`
4. `Across evaluated reasoning benchmarks, DMPO consistently outperforms LLaDA-Instruct, LLaDA-1.5, d1, and cGRPO baselines (Table 1).`
5. `DMPO reports accuracy gains up to 54.3% over previous state-of-the-art baselines and 66.41% over the base model (Table 1).`

### Scope and Limits

- CPU-only execution path with zero paid external API cost (estimated USD 0.00).
- Pure deterministic evaluation and evidence bundle verification.
- Verification on testable reasoning datasets and baseline comparison metrics.

## Architecture and Implementation Plan

1. Directory: `submissions/enhancing-reasoning-for-diffusion-llms-via-distribution-matching-policy-optimization`
2. Modules:
   - `generate_evidence.py`: Evidence generation pipeline producing deterministic evidence bundle.
   - `dmpo_repro/evidence.py`: Core reproduction logic for DMPO distribution matching and baseline comparison metrics.
   - `app.py`: Streamlit Space interface displaying paper overview, claim verifications, and interactive benchmarks.
3. Verification & Tests:
   - `tests/test_dmpo_evidence.py`: Unit and integration tests for evidence generation, schema validity, and numerical claims.
