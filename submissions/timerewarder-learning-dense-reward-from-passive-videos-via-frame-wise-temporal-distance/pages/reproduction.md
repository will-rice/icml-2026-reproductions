# TimeRewarder reproduction evidence

This project recomputes evidence from pinned released artifacts. Its measurement
SHA-256 is
`c1a47c914a17101941a81d7de4422cd5b88ab3c74cc5822bcc9bce041ec2f305`.
Paper-reported values are not treated as reproduced measurements.

## Claim evidence

Ten of ten released-checkpoint task strata pass a fixed
five-video-per-task temporal-distance protocol. Thirty-three pinned source
spans, 106 temporal-distance cases, and three transition-formula cases pass.
All enumerated finite Bellman recurrences and the gamma-one temporal-distance
identity pass under the stated assumptions: full observability, deterministic
transitions, an optimal trajectory, a terminal goal, and unaliased
observations. These three claims are supported by released-checkpoint,
source/formula, and finite-state evidence respectively; they do not reproduce
reward-model training or paper-scale downstream reinforcement learning.

All 50 released-model videos produce finite VOC; the fixed five-video-per-task
mean is `0.998`. This is partial because baseline predictions/checkpoints and
the complete Figure 3 protocol were not released. Matched successful/failed
rollout comparisons and Meta-World/DrQ-v2 training are unavailable.

Exact acquisition, checkpoint, representative-run, and claim records are in
`artifacts/acquisition.json`, `artifacts/checkpoints.json`,
`artifacts/representative.json`, and `artifacts/evidence.json`.
