# Autoregressive Boltzmann Generators Static Evidence Report

Attempt: `1ff17cfb-669a-4192-90f0-c014220f7f12`
Paper: `75AYDsndHP`
Snapshot: `dc14d49cb209316c2a8f5cd9ff0e2ff27eacc29f5d2c4c4a2bacccca9ab6b4cc`

This report summarizes deterministic metadata and source observations from
pinned public artifacts. No paper-reported value is treated as a reproduced
measurement.

## Results

- ArBG likelihood and importance correction: `artifact_verified`. The pinned
  source includes `AutoregressiveLitModule`, exact log-likelihood computation,
  and SNIS/reweighting evaluation paths.
- Topology and intervention claim: `artifact_verified`. The pinned code and
  README expose a diffeomorphism-free autoregressive model with sequential
  temperature/top-k/top-p controls.
- Benchmark improvements and Chignolin: `released_claim_values_unrecomputed`.
  The pinned source includes relevant benchmark/evaluation configuration, but
  no full GPU benchmark rerun or pinned result table is used here.
- Robin model: `metadata_verified`. The pinned HF model repository exposes
  `robin.ckpt` as a 1.06 GB LFS object with a recorded SHA-256.
- Robin energy reduction: `released_claim_values_unrecomputed`. The checkpoint
  and dataset are pinned, but the headline E-W2 percentage reduction is not
  recomputed in this CPU/static bundle.

## Limits

The full Robin checkpoint and ManyPeptidesMD trajectories are not downloaded
during validation. Headline benchmark values require full evaluation runs or
released result artifacts.
