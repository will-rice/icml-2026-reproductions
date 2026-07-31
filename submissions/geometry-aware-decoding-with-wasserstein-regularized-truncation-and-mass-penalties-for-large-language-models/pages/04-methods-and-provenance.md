# Methods and provenance

## Pinned sources

| Source | Pin |
| --- | --- |
| Paper | arXiv:2602.10346v2 |
| Official code | https://github.com/arashgholami/top-w-decoding@5949bfae5e6a81bc279c65923f1adc1c9f2e2059 |

Vendored byte-exact upstream files (MIT license retained):

| File | SHA-256 |
| --- | --- |
| `logit_processor_w1.py` | `a13d6e416ede9fd9788ca016e4e9d23f8e0c1bb57e58cd14dc3552081247d54c` |
| `LICENSE` | `dc7fb9e02ec7b836ab71eb2904a9c4eaa573ff830e3e3ddf28d570118b45c74b` |

## Environment

Python 3.10.12, torch
2.13.0+cu130, cpu
(Linux-5.15.0-174-generic-x86_64-with-glibc2.35). Paid API cost: USD
0.00.

## Reproduce these numbers

```bash
uv run --project . python generate_evidence.py
uv run --project . python -m pytest tests -q
```

All audits use fixed torch seeds recorded in
`src/top_w_repro/evidence.py`; `evidence/bundle.json` is the exact
machine-readable output of the last run.
