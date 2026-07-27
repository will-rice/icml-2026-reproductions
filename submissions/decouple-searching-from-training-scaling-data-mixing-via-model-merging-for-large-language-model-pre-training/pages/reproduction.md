# DeMix released-artifact audit

This is a CPU-only audit of “Decouple Searching from Training: Scaling Data
Mixing via Model Merging for Large Language Model Pre-training”
(`uyRIOjFgOn`). It uses arXiv `2602.00747v3`,
`Lucius-lsr/DeMix@d0c945ca84d5632c6ed1bfe469337cf880757422`, and
`lucius1022/DeMix_Corpora@82a2effc58eb79bec691280a4e4fc50be0968b1e`.

## Claim evidence

The weighted-linear-merging claim receives partial support from the released
mixture manifest and merge configuration. Independent decimal arithmetic found
17 mixture entries over seven domains; eight raw mixture vectors require
normalization because they do not sum exactly to one. No checkpoint was merged
and no model behavior was measured.

The Table 2 Spearman proxy-accuracy claim is unavailable. The release contains
no per-mixture OpenCompass results, and the pinned `proxy_eval.py` uses path
placeholders and `random.random()`. The Table 3 mixture-optimization and final
benchmark claims are also unavailable because their outputs were not released
and full evaluation was not run. Paper values retained in the evidence bundle
are explicitly context with `reproduced: false`.

The primary manifest SHA-256 is
`2be00152f98c44a740bc2f8e2098be3740ea2f1cd31b7158ade9d54c8e852dc2`.
The 14 released component-checkpoint shards total 48,176,346,736 bytes before
merge outputs, evaluation datasets, and caches. The machine-readable evidence
is in `evidence/bundle.json` and `evidence/provenance.json`.
