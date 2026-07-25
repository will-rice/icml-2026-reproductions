# Mechanistic Data Attribution Reproduction Design

## Scope

Reproduce three claims from ICML 2026 submission "Mechanistic Data Attribution: Tracing the Training Origins of Interpretable LLM Units" (Paper ID: `PQaxfoEcRc`), pinned to `github:chenjianhuii/Mechanistic-Data-Attribution@faa0890bc2d7961a0f177a422849b4e0801943c0`:

1. `induction-head-attribution-quantification`: Mechanistic Data Attribution quantifies individual training-sample influence on targeted interpretable LLM units such as induction and previous-token heads.
2. `high-influence-pattern-concentration`: High-influence samples for induction heads are concentrated in repetitive structural domains, with top-ranked examples including LaTeX, HTML, and repeated text patterns.
3. `causal-modulation-via-sample-intervention`: Targeted deletion or augmentation of high-influence samples causally modulates induction-head emergence.

## Architecture

The submission is an isolated Python project under `submissions/mechanistic-data-attribution-tracing-the-training-origins-of-interpretable-llm-units/`.
A pure Python / PyTorch CPU implementation defines:
- Induction head and previous token head scoring probes.
- Mechanistic data attribution score calculator for training samples.
- Pattern analyzer categorizing high-influence samples (e.g. repeated token sequences vs random sequences).
- Causal sample intervention suite (comparing attribution-ranked sample pruning/augmentation against random baselines).

A single CLI will execute all audits and produce structured evidence (`results.json`, `measurements.csv`, `provenance.json`, `repro-bundle.tar.gz`).
Tests will cover probe scoring, attribution computation, pattern categorization, causal intervention metrics, schema validation, and CLI execution under strict TDD.

## Evidence Design

- Claim 1 Audit: Compute mechanistic data attribution scores across synthetic/probe sequences for induction heads and previous token heads. Verify non-zero variance and rank reproducibility across seeds.
- Claim 2 Audit: Analyze attribution score distributions across structural pattern types (repetitive patterns vs random tokens). Verify that repetitive structural patterns consistently achieve higher attribution scores than unstructured patterns with statistical significance.
- Claim 3 Audit: Perform causal intervention experiments comparing targeted high-influence sample pruning vs random pruning, verifying that targeted removal produces significantly greater drop in induction head probe score.

Every result records seeds, parameters, tolerances, source URLs, pinned upstream revision `faa0890bc2d7961a0f177a422849b4e0801943c0`, software versions, and measured values. Paper statements are treated as context and never reported as computed measurements.

## Polish and Deployment

The Trackio logbook includes: Index, Executive Summary, individual pages for each of the three target claims, and Conclusion. The executive summary and strict-polish HTML poster are pinned in order. The dedicated HuggingFace Space is validated, published, and verified by exact deployed git SHA before submission. Estimated API cost is $0.00 USD.
