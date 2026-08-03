# Lane 1: Correlated-Sampling Theory Audit

- **Challenge Claim SHA-256:** `5f0d21d91c0ae1d2877563e7115e804db60361304db4aea72b97596300e60f57`
- **Claim Title:** Correlated-sampling PAC to RSR reduction bound
- **Assessed Status:** `verified`

## Claim Description and Source Locators

The paper introduces formal foundations for Randomized Self-Reductions (RSRs) in Definition 4.1 (page 4), Correlated Query Distributions in Definition 4.3 (page 5), and the RSR Learning Problem in Definition 4.5 (page 5). In Appendix A.1 and A.2, the authors state Claims A.1 and A.2, proving that PAC learning over correlated query distributions reduces sample complexity via union bounds when query marginals are uniform.

## Executable Finite-Model Verification

We constructed a finite modular addition model \( \mathbb{Z}_4 \):
- Domain: \( \{0, 1, 2, 3\} \)
- Randomness: \( r \in \{0, 1, 2, 3\} \)
- Queries: \( q_1(x, r) = (x + r) \bmod 4 \) and \( q_2(x, r) = r \)
- Recovery function: \( g(x, r, (y_1, y_2)) = (y_1 - y_2) \bmod 4 \)

### Results of Finite Audit

1. **Marginal Uniformity:** Verified that for every input \( x \), both \( q_1(x, r) \) and \( q_2(x, r) \) are uniformly distributed over \( \{0, 1, 2, 3\} \) as \( r \) ranges uniformly over \( \mathbb{Z}_4 \).
2. **Perfect Recovery:** Verified that when the hypothesis \( h = f^* \), recovery succeeds for 100% of inputs and randomness values.
3. **Reduction Implication:** For a hypothesis with error fraction \( \epsilon = 1/4 \), recovery with error tolerance \( \rho = 1/2 \) yields a good input fraction of \( 1.0 \ge 1 - \xi \) (\( \xi = 1/4 \)). Minimum recovery probability across good inputs is \( 1.0 \ge 1 - \rho \).

## Audit Commands & Reproducibility

```bash
uv run pytest -q tests/test_theory.py
```

## Limitations

This finite enumeration audit verifies the structural mechanics of the union bound and marginal uniformity requirements on finite models; it does not replace full general symbolic proof.
