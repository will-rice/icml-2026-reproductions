# Lane 2: 80-Function RSR-Bench Census Audit

- **Challenge Claim SHA-256:** `79d94d106cfded95104c54624068a07dc9ae16dca681a6ad5370bbb648e8c7de`
- **Claim Title:** 80-function RSR-Bench census
- **Assessed Status:** `verified`

## Census Strategy and Upstream Inspection

RSR-Bench is defined across two Python evaluation scripts in the upstream repository:
- `src/bitween/evaluation/evaluation_rsr_bench_paper.py` (Base script, IDs 1..40)
- `src/bitween/evaluation/evaluation_rsr_bench_paper_extended.py` (Extended script, IDs 41..80)

We used Python AST parsing to walk all function calls to `evaluate(...)` and extract literal keyword registrations of the form `test_id="<ID>_<name>"`. We reconciled these 80 extracted registrations with column 0 of the released primary CSV spreadsheet `results/Bitween-Results(Sheet1-ICML).csv`.

## Key Reproduced Findings

1. **Exact 1..80 Range:** Exactly 40 registrations were found in the base script (IDs 1 through 40) and exactly 40 registrations in the extended script (IDs 41 through 80).
2. **Primary CSV Rows:** Column 0 of the raw CSV contains exactly 80 primary benchmark rows numbered 1 through 80. Extra continuation rows contain multi-line equations and do not inflate the benchmark count.
3. **One-to-One Mapping:** Every parsed registration name maps uniquely to a primary CSV row. Benchmark 33 corresponds to `sigmoid`.

## Audit Commands & Reproducibility

```bash
uv run pytest -q tests/test_benchmark.py
```

## Limitations

This is a syntax-only AST extraction of registration calls; it verifies the 80 benchmark definitions without re-executing the full benchmark evaluation suite.
