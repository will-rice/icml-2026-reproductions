# RBench released-artifact reproduction

This CPU-only audit covers paper `p5QSlnwume`. It recomputes facts from pinned
prompt manifests and leaderboard artifacts; it does not rerun video generation
or human-correlation experiments.

## Claim evidence

The nine prompt manifests contain 650 records spanning five task and four
embodiment categories. The paper-era leaderboard has 25 valid unique records.
The later cohort has 28 and prepends three models while preserving the
paper-era entries. A displayed-field consistency check finds one material
discrepancy: `LingBot-Video` reports mean `0.620`, while its nine displayed
fields average to `0.614`. The aggregation rule itself was not source-traced,
so this is an internal consistency observation rather than a reconstructed
official score.

The exact phrases “structural distortion,” “floating components,” and
“key-action omission” were not found in the pinned allowlisted artifacts.
Consequently the named failure-mode claim remains inconclusive rather than
being inferred from absence. Machine-readable provenance and results are in
`evidence/input-manifest.json`, `evidence/results.json`,
`evidence/commands.json`, and `evidence/validation.json`.
