# tau2-Bench Reproduction Evidence

This reproduction targets paper `OC2z7iSQKa`, "$\tau^2$-Bench: Evaluating
Conversational Agents in a Dual-Control Environment". It uses pinned upstream
source `github:sierra-research/tau2-bench@1d244f5dca42944b67a379b44bfeb9f5748f189d`
and CPU-only source/artifact checks. No paid LLM agent evaluations were run.

## Claim 1

`0199b3b43b308ce8469189f64e2310b12cb869a8c6255975c3c4cb7e9093f78a`

The pinned `TelecomEnvironment` wires both `TelecomTools` and
`TelecomUserTools`, backed by separate `TelecomDB` and `TelecomUserDB`
objects. Its `sync_tools` method propagates line activity, roaming allowance,
data-limit state, and payment updates through shared surroundings state. This
verifies the selected dual-control shared-state structure claim.

## Claim 2

`36d94de446993da42bb35022e284615dd122232225cf76fb0d3ccf26116e2788`

The evidence counts 13 `@is_tool` registrations in `telecom/tools.py` and
30 `@is_tool` registrations in `telecom/user_tools.py`. It also loads the
telecom split and task artifacts, observing 114 `base` tasks and 2,285 full
tasks. These values are computed from the pinned source files, not copied from
the paper.

## Claim 3

`9e672b1894ac3fb6b00f8fa2a33a5d31355aa5663481e563c0594d337f0356b5`

The evidence audits `telecom/tasks/create_tasks.py` and confirms composition
from mobile-data, service, and MMS task managers. The generator groups tasks
by intent, subtask count, and persona, samples from those bins, and serializes
the generated task objects to JSON. This supports the selected compositional
task-generator claim.
