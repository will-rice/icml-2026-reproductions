# Learning Randomized Reductions Reproduction Summary

This directory contains the independent, CPU-only reproduction bundle for paper `hCAEcqig2C` (**Learning Randomized Reductions**, arXiv:2412.18134v5) under attempt `eb10c79b-fc26-47c4-88c1-6f45cb592833`.

## Overview of Challenge Claims and Outcomes

| Lane / Claim Title | Challenge Claim SHA-256 | Status | Independent Reproduction Key Observation |
|---|---|---|---|
| Correlated-sampling theory | `5f0d21d91c0ae1d2877563e7115e804db60361304db4aea72b97596300e60f57` | Verified | Definitions 4.1, 4.3, 4.5 and Appendix A.1-A.2 bounds verified on finite modular addition model with uniform query marginals; implication holds with good input fraction 1.0 >= 0.75. |
| RSR-Bench census | `79d94d106cfded95104c54624068a07dc9ae16dca681a6ad5370bbb648e8c7de` | Verified | Reconciled 40 base script IDs + 40 extended script IDs to exactly 80 primary CSV benchmark rows (1..80). Benchmark 33 is sigmoid. |
| Vanilla Bitween & Sigmoid | `4b8bfdf084cb0038acc0a589837dc4379ba1fb079f30f4be8edf839a21d23a51` | Partial | Recomputed linear regression coverage of 43/80 (87 RSRs, mean runtime 4.791s). Symbolically verified sigmoid identity diff=0. Historical priority ("first known") is unreplicated. |
| Agentic Bitween | `9b35061b3b4e2873f1b7a4fffc6fa22d659f281c096d990706ebd805303c4c00` | Verified | Recomputed Claude-Opus-4.1 coverage of 64/80 (793 RSRs). Extracted novel query functions including `x+log(k)` from released property outputs. |
| Nonlinear-invariant comparison | `13999601811ffe2bb8e9526ed601e9d59480b217d6d1917787db2a9c7dbc8372` | Falsified | Falsified exact wording. Source locators establish that v1 Table 2 is a post-condition example (20 samples), RSR-Bench compares LR vs MILP, NLA-DigBench compares against DIG/SymInfer, and v5 Table 2 reports novel Agentic query functions. |

## Provenance and Pins

- Upstream Repository: `github:ferhaterata/learning-randomized-reductions@e13d4b59f6d23051c73e07cfc447336da84e7bd2`
- Paper Version 1: arXiv:2412.18134v1 (SHA-256: `abaac08eabec2e77c8af7ae3ca028691b9cd862e21bfa779452b9fd729e3222f`)
- Paper Version 5: arXiv:2412.18134v5 (SHA-256: `93cab4aa8cec06434b704e639bab87dd15ea95ac46a335961138a94fc1bae2b8`)
- Raw CSV Results: `results/Bitween-Results(Sheet1-ICML).csv` (SHA-256: `7198413f93830f7903bf3b670b718f2ccfbab1a41496a1fc3fe085850af0df0b`)
