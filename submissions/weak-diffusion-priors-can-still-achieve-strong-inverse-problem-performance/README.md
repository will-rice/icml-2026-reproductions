# Weak Diffusion Priors Can Still Achieve Strong Inverse-Problem Performance

Official reproduction package for ICML 2026 Paper `fdkSA4F0lN` (arXiv:2601.22443).

## Target Claims Verified
1. **Claim 1** (`f92ef3142d3eb9876b4885e506e2318923f6277bd326f60f7e741fa6259e7ba9`):
   Weak diffusion priors match strong-prior inverse-problem baselines when measurements are highly informative (Table 1).
2. **Claim 2** (`4d2832c903b2d7d6e55947d20468d734b233f664c678deed688b9c37ae5b8aac`):
   Theorem 3.1 conditions under which high-dimensional measurements cause Bayesian posterior concentration near true signal despite weak priors.

## Execution
```bash
python generate_evidence.py
pytest tests/ -q
streamlit run app.py
```
