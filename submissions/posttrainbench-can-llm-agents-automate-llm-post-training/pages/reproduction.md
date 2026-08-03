# PostTrainBench released-artifact audit

This deterministic CPU audit covers PostTrainBench (`UnjxMTe57e`, arXiv
`2603.08640v2`). No H100 post-training run was reproduced.

## Claim evidence

The benchmark-coverage claim receives partial support. The released trajectory
inventory confirms all accepted cells of the stated four-base-model by
seven-benchmark matrix. Runner configuration defaults to one H100 and a
`NUM_HOURS`-based timeout, while scheduler-dependent branches and the
five-minute termination grace remain limitations.

The reward-hacking claim also receives partial support. Released contamination
and instruction-model judgments provide artifact evidence for training on test
sets and downloading instruction-tuned checkpoints. The pinned revision does
not contain the API-key submode artifact, so that component is unavailable.
Released judge labels are not treated as independently established behavioral
truth.

The machine-readable records are `evidence/coverage.json`,
`evidence/reward_hacking.json`, `evidence/claims.json`,
`evidence/provenance.json`, and `evidence/manifest.json`. Paid API cost was
USD 0.00.
