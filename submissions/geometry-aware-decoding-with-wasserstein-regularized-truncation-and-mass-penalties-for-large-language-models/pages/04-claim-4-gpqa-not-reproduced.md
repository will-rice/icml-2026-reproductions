# Claim 4: GPQA comparison against Min-p, Top-p, and Top-H

**Claim.** Top-W is evaluated against the same decoding baselines on GPQA across multiple temperatures and models (Table 2).

**Self-assessed status: unreplicated**

## What this reproduction did NOT do

Not reproduced. The Table 2 comparison requires GPQA decoding runs across the same instruction-tuned models and temperatures as Table 1. No language model was executed in this reproduction; no GPQA accuracy numbers exist here. The official repository pinned in the upstream manifest ships run_gpqa.sh as the entry point for an independent GPU reproduction. The decoding mechanism GPQA would exercise is the same audited mechanism as claims 1-2: the objective, exact S-step, and official-code cross-check numbers on the claim 1 and claim 2 pages are the only mechanism-level evidence this attempt provides.

## What exists for an independent benchmark rerun

The official repository (pinned at
`github.com/arashgholami/top-w-decoding@5949bfae5e6a81bc279c65923f1adc1c9f2e2059`) ships `run_gpqa.sh`
as the Table 2 entry point. Reproducing Table 2 requires GPU decoding
of GPQA with the paper's instruction-tuned models across its
temperature grid — outside this CPU-only attempt's budget.

## Relation to the audited mechanism

GPQA exercises the same Top-W decoder mechanism that this attempt
audits numerically: the claim 1 page shows the objective and
convergence numbers, and the claim 2 page shows exact-S-step and
official-code cross-check numbers. No benchmark accuracy is claimed
from that mechanism evidence.
