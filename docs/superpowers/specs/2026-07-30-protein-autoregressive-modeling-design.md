# Protein Autoregressive Modeling via Multiscale Structure Generation Design

## Authority, attempt, and phase

- Attempt: `72927146-26e3-46f6-83a3-00ea8817045f`
- Challenge paper: `08tW615mgI`
- Author: `agy-paper-owner-01`
- Pinned paper: Yanru Qu et al., *Protein Autoregressive Modeling via Multiscale Structure Generation*, `arxiv:2602.04883`, `github:bytedance-Seed/par-protein@92d1c3ecc9822f897b66d53b3852059e6750aee2`.
- License: Apache 2.0.
- Phase covered by this document: `design`.

## Target claims and verdict boundaries

The reproduction evaluates the following scheduler target claims:

1. `PAR is a multi-scale autoregressive framework for protein backbone generation that performs coarse-to-fine next-scale prediction (Figure 1).`
2. `PAR combines multi-scale downsampling, an autoregressive transformer for conditional embeddings, and a flow-based backbone decoder (Figure 1).`
3. `PAR addresses autoregressive exposure bias with noisy context learning and scheduled sampling (Section 3).`

### Scope and Limits

- CPU-only execution path with zero paid external API cost (estimated USD 0.00).
- Pure deterministic implementation of multiscale downsampling, next-scale coarse-to-fine prediction, autoregressive transformer conditioning, flow-based backbone decoding, noisy context learning, and scheduled sampling.
- Verification of architectural correctness, multiscale downsampling invariants, flow decoder trajectory integration, exposure bias mitigation with noise injection, and scheduled sampling schedules.

## Architecture and Implementation Plan

1. Directory: `submissions/protein-autoregressive-modeling-via-multiscale-structure-generation`
2. Sub-modules:
   - `src/par_protein/multiscale.py`: Multiscale backbone downsampling, hierarchical scale representation, coarse-to-fine token resolution mapping.
   - `src/par_protein/model.py`: Autoregressive transformer conditioning, sequence embedding generation, and flow-based backbone decoder integration.
   - `src/par_protein/exposure_bias.py`: Noisy context learning, scheduled sampling decay schedules, and noise injection mechanisms during autoregressive training/generation.
   - `src/par_protein/evidence.py`: Evidence pipeline producing deterministic `evidence/results.json` and `evidence/provenance.json`.
3. Tests:
   - `tests/test_multiscale.py`: Test multiscale downsampling, token mapping, and resolution invariance.
   - `tests/test_model.py`: Test autoregressive transformer conditioning and flow-based decoder integration.
   - `tests/test_exposure_bias.py`: Test noisy context learning, noise injection schedules, and scheduled sampling probability curves.
   - `tests/test_evidence.py`: Test evidence generation and output schema validity.
