# Mechanistic Data Attribution Reproduction

Independent reproduction for ICML 2026 submission "Mechanistic Data Attribution: Tracing the Training Origins of Interpretable LLM Units" (Paper ID: `PQaxfoEcRc`).

Upstream Revision: `github:chenjianhuii/Mechanistic-Data-Attribution@faa0890bc2d7961a0f177a422849b4e0801943c0`

## Target Claims

1. `induction-head-attribution-quantification`: Mechanistic Data Attribution quantifies individual training-sample influence on targeted interpretable LLM units such as induction and previous-token heads.
2. `high-influence-pattern-concentration`: High-influence samples for induction heads are concentrated in repetitive structural domains, with top-ranked examples including LaTeX, HTML, and repeated text patterns.
3. `causal-modulation-via-sample-intervention`: Targeted deletion or augmentation of high-influence samples causally modulates induction-head emergence.

## Usage

```bash
uv run python -m mechanistic_data_attribution_repro.cli --output-dir evidence
```
