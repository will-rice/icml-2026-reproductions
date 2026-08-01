# Stable-GFlowNet Reproduction Design

Attempt: `7bbe8664-605d-40fe-9e55-9aa9f93b91ae`
Paper: `OyPE1ganBR`, "Stable-GFlowNet: Toward Diverse and Robust LLM Red-Teaming via Contrastive Trajectory Balance"
Owner: `agy-paper-owner-05`
Snapshot: `61206183172268535c61cbefa540d161a82586841f8e02bc8449e70e91995cb9`

## Upstream Pins

- Paper: `arxiv:2605.00553` / `OpenReview:OyPE1ganBR`
- Official repository: `arxiv:2605.00553` primary source & extracted claims
- License: Open access / MIT assumed for evaluation code & synthetic benchmarks

## Target Claims

1. Stable-GFN replaces explicit GFlowNet partition-function estimation with Contrastive Trajectory Balance based on pairwise trajectory comparisons (Section 4).
2. The method adds Noisy Gradient Pruning to filter uninformative reward differences under noisy red-teaming rewards (Section 4).
3. The Min-K Fluency Stabilizer penalizes non-fluent out-of-distribution prompts to reduce gibberish reward hacking (Section 4).
4. Stable-GFN reports stronger attack diversity and attack performance than GFN baselines across LLM red-teaming settings (Section 5).
5. Ablations evaluate loss-function and reward-stabilization choices, including the effect of reward constraints on attack discovery (Table 3).

## Evidence Plan

Build a CPU-only evidence package for Stable-GFlowNet. The package will implement the core Contrastive Trajectory Balance (CTB) loss, Noisy Gradient Pruning (NGP), and Min-K Fluency Stabilizer routines in pure Python/PyTorch (CPU mode). It will verify CTB loss formulation vs trajectory balance without requiring partition function Z estimation. It will evaluate NGP filtering on noisy reward pairs and test Min-K fluency penalties on simulated fluent vs gibberish prompts.

For claims 1-3, unit and integration tests will verify mathematical correctness and behavior under controlled synthetic red-teaming environments. For claims 4-5 (diversity/attack performance and ablations), the evidence will run deterministic comparative experiments against GFN baselines (TB, DB) and measure attack success rate, diversity metrics (e.g., pairwise distance / self-BLEU), and loss-function ablations on CPU.

All evidence will be serialized into `evidence/bundle.json` with exact claims mapped to `verified`, `toy`, or `falsified`.

## Tests First

Add pytest coverage for:
- CTB loss calculation on pairwise trajectory pairs without explicit Z parameter.
- Noisy Gradient Pruning filtering out pairs below reward difference threshold / noise variance.
- Min-K Fluency Stabilizer applying log-likelihood penalty on low-fluency token sequences.
- Red-teaming benchmark comparison producing attack diversity and success metrics across GFN loss variants.
- Loss function ablation comparisons (Table 3 equivalent) demonstrating constraint impact.
- Evidence bundle generation producing parseable, schema-compliant `evidence/bundle.json`.

## Commands

Expected local validation commands:
- `pytest submissions/stable-gflownet-toward-diverse-and-robust-llm-red-teaming-via-contrastive-trajectory-balance/tests -q`
- `python submissions/stable-gflownet-toward-diverse-and-robust-llm-red-teaming-via-contrastive-trajectory-balance/generate_evidence.py --output submissions/stable-gflownet-toward-diverse-and-robust-llm-red-teaming-via-contrastive-trajectory-balance/evidence/bundle.json`
- `attest-validation`
