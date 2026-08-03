# Lane 4: Agentic Bitween Coverage & Novel Query Audit

- **Challenge Claim SHA-256:** `9b35061b3b4e2873f1b7a4fffc6fa22d659f281c096d990706ebd805303c4c00`
- **Claim Title:** Agentic Bitween coverage and novel queries
- **Assessed Status:** `verified`

## Raw CSV Aggregation Results

We recomputed the Claude-Opus-4.1 Agentic Bitween backend statistics across all 80 primary benchmark rows in `Bitween-Results(Sheet1-ICML).csv` (Columns 53 and 56):

- **Covered Benchmarks:** 64 out of 80 benchmarks achieved at least 1 verified RSR (\( 64/80 = 80.0\% \)).
- **Total RSRs Discovered:** 793 RSRs.
- **Full Row Representation:** All 80 benchmark rows are represented in the raw data, including zero-coverage rows.

## Novel Query Function Extraction

Vanilla Bitween uses a fixed prior set of query candidates: \( \{x+r, x-r, x \cdot r, x, r\} \). Agentic Bitween prompts an LLM to propose dynamic candidate query functions. We scanned all released property continuation rows for calls to \( f(\cdot) \), extracted outer arguments using balanced-parenthesis parsing, and canonicalized whitespace and operator order.

### Key Finding

The released agentic outputs contain novel query functions outside the fixed prior set, including:
- `x+log(k)`

This confirms that Agentic Bitween generated expressive, non-fixed query candidates during the released experiment run.

## Audit Commands & Reproducibility

```bash
uv run pytest -q tests/test_results.py
```

## Limitations

This lane recomputes results from released raw CSV outputs; it did not rerun live remote Claude LLM inference or paid API calls.
