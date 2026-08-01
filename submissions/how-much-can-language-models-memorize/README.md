# How much can language models memorize?

ICML 2026 reproduction attempt `75efbd6f-ce4a-467f-a094-5e62efc328b8` for
paper `bA6BgSbaUi`.

This submission runs an independent CPU-only toy reimplementation of the
uniform-random-data memorization protocol from arXiv `2505.24832`. It does not
use paper-reported values as measurements and does not reproduce the full
500K-to-1.5B parameter sweep.

Generate evidence:

```bash
python generate_evidence.py
```

Run tests:

```bash
PYTHONPATH=. pytest -q
```
