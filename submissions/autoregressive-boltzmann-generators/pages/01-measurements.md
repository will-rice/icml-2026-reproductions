# ArBG Measurements and Pins

This page lists concrete measurements and immutable artifact pins generated
from `evidence/autobg_results.json`. Headline benchmark values are not treated
as reproduced measurements.

## Attempt

- Attempt ID: `1ff17cfb-669a-4192-90f0-c014220f7f12`
- Paper ID: `75AYDsndHP`
- Snapshot ID: `dc14d49cb209316c2a8f5cd9ff0e2ff27eacc29f5d2c4c4a2bacccca9ab6b4cc`
- OpenReview paper: `2606.27361`
- Estimated API cost: `0.0` USD

## Upstream Pins

- AutoBG code commit: `21624a80504b3199b291514c37a49cccd19c8817`
- Robin model revision: `2813c971b63a177ad578c51c9a550c2e63e9168d`
- ManyPeptidesMD revision: `1af9336878122eb1d62894fe2fb3ff4b801a3216`
- Robin checkpoint LFS size: `1062461935` bytes
- Robin checkpoint SHA-256: `f7510d82312c6aab9546288e7714e123088d67837f1750bd37e55c479204103f`
- ManyPeptidesMD file count: `26945`

## Source Tree Audit

- Python file count in pinned source tree: `82`
- Config file count in pinned source tree: `116`
- Script file count in pinned source tree: `3`
- Important audited file count: `11`
- License status: `mit-plus-notice`
- Git tree truncated: `false`
- Autoregressive module detected: `true`
- Exact log-likelihood path detected: `true`
- SNIS/reweighting path detected: `true`
- Sequential intervention controls detected: `true`
- Chignolin config detected: `true`
- Robin eval script detected: `true`

## Claim Outcomes

- Claim 1 status: `artifact_verified`
- Claim 2 status: `artifact_verified`
- Claim 3 status: `released_claim_values_unrecomputed`
- Claim 4 status: `metadata_verified`
- Claim 5 status: `released_claim_values_unrecomputed`

## Important File Hashes

- `src/models/autoregressive_module.py`: `1e63a069480557652642a0122e7ffe7647686e11c6d29dd1cfa6b43079bff3ef`
- `src/models/transferable_boltzmann_generator_module.py`: `746ab691f3c5ca0416db66988260a6f4c1bbd1e949b21b564f4fea47a5c18446`
- `src/models/neural_networks/autoregressive/causal_transformer.py`: `1734485cc292f795ec3be40c840fe2a7ded78a0328f9bdd81943caa902d73e5e`
- `configs/model/autoregressive.yaml`: `efd554b8eca34438dbb88c7087a068355ce106c4519b1494edf78bf24c980eea`
- `scripts/eval_transferable.sh`: `7a41c550d5521a158a37964c90aa4f8a050f53468b46990a575a052ba983d9aa`
