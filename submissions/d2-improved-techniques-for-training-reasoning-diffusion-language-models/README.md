# d2 Reproduction Evidence

This submission audits the pinned official d2 source and recomputes CPU-only finite-state checks for masked-DLM trajectory likelihood claims.

Generate evidence:

```bash
uv run --project submissions/d2-improved-techniques-for-training-reasoning-diffusion-language-models python submissions/d2-improved-techniques-for-training-reasoning-diffusion-language-models/generate_evidence.py
```

Run tests:

```bash
uv run --project submissions/d2-improved-techniques-for-training-reasoning-diffusion-language-models python -m pytest submissions/d2-improved-techniques-for-training-reasoning-diffusion-language-models/tests -q
```

The evidence intentionally does not claim benchmark speed or accuracy numbers unless they are recomputed or released as raw artifacts.
