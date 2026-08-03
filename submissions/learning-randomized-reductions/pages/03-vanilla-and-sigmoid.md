# Lane 3: Vanilla Bitween and Sigmoid Reduction Audit

- **Challenge Claim SHA-256:** `4b8bfdf084cb0038acc0a589837dc4379ba1fb079f30f4be8edf839a21d23a51`
- **Claim Title:** Vanilla Bitween coverage and sigmoid reduction
- **Assessed Status:** `partial`

## Raw CSV Aggregation Results

We recomputed the linear-regression backend (mreg / Vanilla Bitween) statistics across all 80 primary benchmark rows in `Bitween-Results(Sheet1-ICML).csv` (Columns 18 and 21):

- **Covered Benchmarks:** 43 out of 80 benchmarks achieved at least 1 verified RSR (\( 43/80 = 53.75\% \)).
- **Total RSRs Discovered:** 87 RSRs.
- **Runtime Statistics:**
  - Minimum runtime: 0.13 seconds
  - Mean runtime: 4.791 seconds
  - Maximum runtime: 19.12 seconds

## Symbolic Algebra Verification of Sigmoid Identity

Row 33 of the CSV reports a 3-RSR reduction for the sigmoid function \( \sigma(x) = \frac{1}{1 + e^{-x}} \). We parsed the released identity:
\[
\sigma(x) = \frac{\sigma(x+r)(\sigma(r) - 1)}{2\sigma(x+r)\sigma(r) - \sigma(x+r) - \sigma(r)}
\]
Using SymPy, we simplified \( \sigma(x) - \text{RHS} \) symbolically over real variables \( x, r \) and verified that the difference simplifies identically to 0.

## Audit Commands & Reproducibility

```bash
uv run pytest -q tests/test_results.py
```

## Limitations & Scope Note

While the 43/80 coverage and the symbolic sigmoid reduction identity are fully verified, an exhaustive historical literature search was not performed. Therefore, historical priority ("first known" reduction) remains unreplicated.
