# Autoregressive Boltzmann Generators Evidence

This submission is a deterministic static evidence bundle for `75AYDsndHP`,
"Autoregressive Boltzmann Generators".

Pinned public artifacts:

- Code: `danyalrehman/autobg@21624a80504b3199b291514c37a49cccd19c8817`
- Robin checkpoint: `danyalrehman17/robin-transferable@2813c971b63a177ad578c51c9a550c2e63e9168d`
- ManyPeptidesMD: `transferable-samplers/many-peptides-md@1af9336878122eb1d62894fe2fb3ff4b801a3216`

Run:

```bash
uv run --project submissions/autoregressive-boltzmann-generators python submissions/autoregressive-boltzmann-generators/generate_evidence.py
uv run --project submissions/autoregressive-boltzmann-generators pytest submissions/autoregressive-boltzmann-generators/tests -q
```
