# Lane 5: Nonlinear-Invariant Claim Falsification Audit

- **Challenge Claim SHA-256:** `13999601811ffe2bb8e9526ed601e9d59480b217d6d1917787db2a9c7dbc8372`
- **Claim Title:** Nonlinear-invariant benchmark comparison
- **Assessed Status:** `falsified`

## Literal Challenge Claim Wording

> "Learning Randomized Reductions presents Bitween evaluation on nonlinear invariant benchmarks compared to MILP in terms of sample count and runtime in Table 2."

## Source Audit and Conflation Analysis

We performed a conjunctive source locator audit across pinned arXiv paper versions (v1 and v5). The literal claim conflates three separate sections and tables:

1. **v1 Table 2:** Presents a single learned post-condition example with 20 samples; it is not a backend comparison table.
2. **v1 Section 5.3.1:** Compares Linear Regression (LR) versus MILP backends in terms of sample count (594 vs 1,095) and runtime (130.53s vs 187.47s), but this experiment is on **RSR-Bench**, not NLA-DigBench.
3. **v1 Section 5.3.2:** Evaluates Bitween on **NLA-DigBench** (nonlinear invariants), but compares Bitween against DIG and SymInfer, not against the MILP backend.
4. **v5 Table 2:** Presents novel Agentic Bitween query functions on RSR-Bench, not a MILP comparison.

## Verified Contradictions

The audit established the following 4 explicit contradictions:
- v1 Table 2 is a learned post-condition example, not a backend comparison
- LR versus MILP sample/runtime results are for RSR-Bench, not NLA-DigBench
- NLA-DigBench compares Bitween with DIG and SymInfer, not MILP
- v5 Table 2 reports novel Agentic Bitween query functions

Because no single source locator matches all 4 claim dimensions simultaneously, the exact live claim is marked `falsified`.

## Audit Commands & Reproducibility

```bash
uv run pytest -q tests/test_claim_scope.py
```

## Limitations

This audit evaluates literal challenge wording against pinned source locations. It preserves the narrower true paper statements without using them to alter or rescue the conflated claim text.
