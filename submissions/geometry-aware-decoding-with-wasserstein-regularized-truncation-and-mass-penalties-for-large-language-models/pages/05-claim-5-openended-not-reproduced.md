# Claim 5: AlpacaEval and MT-Bench judge-based evaluations

**Claim.** Judge-based open-ended evaluations report Top-W wins on more AlpacaEval and MT-Bench temperature-model tuples than the compared decoding methods (Figure 1, Figure 2).

**Self-assessed status: unreplicated**

## What this reproduction did NOT do

Not reproduced. The AlpacaEval and MT-Bench win-rate comparisons (Figure 1, Figure 2) require open-ended generation with multiple models and temperatures plus a judge model. No language model or judge was executed in this reproduction, and the challenge budget excludes paid judge APIs (recorded cost USD 0.00). The official repository pinned in the upstream manifest ships alpaca_generate_w.py for generation; judge-side evaluation would additionally require the AlpacaEval and MT-Bench harnesses.

## What exists for an independent rerun

The official repository (pinned at
`github.com/arashgholami/top-w-decoding@5949bfae5e6a81bc279c65923f1adc1c9f2e2059`) ships
`alpaca_generate_w.py` for the generation side. Win-rate judging
additionally requires the AlpacaEval and MT-Bench harnesses and a
judge model; this attempt records USD 0.00 paid API cost and ran no
judge.
