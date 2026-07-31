# Claim 2: Exact prefix-form subset update inside the candidate-pool loop

**Claim.** The method instantiates a practical alternating decoder with an exact subset-update step inside a candidate-pool loop (Section 4.2).

**Self-assessed status: verified**

## Brute-force enumeration control (Theorem 3.4a, beta >= lam)

For 10-token candidate pools, every one of the
1023 nonempty subsets was enumerated
and scored with the fixed-potential objective; the prefix-form linear
scan returned the exact optimum in
**120/120** trials
(max objective gap 0.00e+00), across four
hyperparameter configurations satisfying the theorem's beta >= lam
hypothesis: (lam=2.2, beta=2.8, geom_scale=0.6); (lam=1.5, beta=2.4, geom_scale=0.3); (lam=3.0, beta=3.5, geom_scale=1.0); (lam=2.2, beta=2.2, geom_scale=0.6).

| Method | Mean time per instance |
| --- | --- |
| Prefix scan (exact S-step) | 0.104 ms |
| Brute-force enumeration | 43.8 ms |

## Relaxation control (beta < lam)

Theorem 3.4(a) assumes `beta - lam >= 0`. Rerunning the identical
brute-force comparison with
(lam=2.2, beta=1.0)
produced **7/120**
instances where the pure prefix scan was strictly suboptimal (worst
objective gap 0.299) — the collapse
regime the paper describes, showing the hypothesis is load-bearing. The
official implementation's defaults (lam=2.2, beta=2.8) satisfy the
hypothesis.

## Cross-check against the official implementation

The reimplemented alternating decoder (candidate pool top_m=64, 9
alternating iterations, lam=2.2, beta=2.8, geom_scale=0.6 — the official
defaults) kept token sets **identical to the vendored official
`logit_processor_w1.py`** in
25/25 random
400-token instances (mean kept size 1.5).
The official file is byte-exact at
`evidence/inputs/upstream/logit_processor_w1.py` (SHA-256 in
`evidence/inputs/upstream_manifest.json`).

## Limitations

Brute force is feasible only for 10-token pools; larger pools rely on the theorem, not enumeration.
