# Numeric Evidence Surface

This page renders the computed values from `evidence/bundle.json` so the served Space exposes concrete evidence without relying on paper-reported table values as reproduced measurements.

## Recomputed Dataset Counts

| Artifact | Recomputed value | Evidence status |
| --- | ---: | --- |
| MEnvBench tasks | 1,000 tasks | verified |
| MEnvBench repositories | 200 repositories | verified |
| MEnvBench languages | 10 languages | verified |
| MEnvBench Python rows | 100 rows | verified |
| MEnvData-SWE task instances | 3,005 task instances | verified |
| MEnvData-SWE repositories | 942 repositories | verified |
| MEnvData-SWE languages | 10 languages | verified |
| MEnvData-SWE-Trajectory viewer rows | 3,918 rows | verified |
| Claimed solution trajectories | 3,872 solution trajectories | checked |
| Trajectory count match | false | falsified boundary |
| Challenge claim bindings | 6 claims | recorded |
| Claims with unavailable experiment evidence | 3 claims | explicit boundary |

## Source Release Checks

| Released-source check | Observed value |
| --- | --- |
| Planning-Execution-Verification terms present | true |
| Environment reuse terms present | true |
| EnvPatchAgent present | true |
| Curation scripts present | true |
| Public core-code note | "The core code is currently being organized for public release." |

## Claim Boundary

Claims about MEnvAgent runtime wins, component ablations, and fine-tuned model resolved rates remain unavailable in this reproduction because the served evidence does not contain raw executable logs for those experiments. The dataset-count claims are recomputed from pinned public artifacts, and the 3,872 trajectory count is explicitly not matched by the 3,918-row Dataset Viewer observation.
