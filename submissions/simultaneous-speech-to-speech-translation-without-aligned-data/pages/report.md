# Reproduction Report: Simultaneous Speech-to-Speech Translation Without Aligned Data (Hibiki-Zero)

**Paper ID:** `76XSBLdBdg`

## What this reproduction does and does not claim

This is a CPU-only, toy-scale mechanism reproduction. Three of the paper's
mechanisms are exercised with real, seeded computations whose numbers appear
below. The paper's benchmark tables (Table 1), human MOS study (Table 2), and
850h Italian adaptation (Table 3) are **unreplicated**: they require the 3B
model, benchmark harnesses, and human raters. Earlier revisions of this
package fabricated those tables; the fabricated values have been removed and
no benchmark or MOS numbers are reported here.

## Claim 1 - training from sentence-aligned data (Section 3): partially reproduced

A streaming wait-k emitter is trained from whole-sentence pairs only (no
word-level timing labels) on a toy translation task whose mapping needs one
token of lookahead. Measured on held-out sentences:

| wait-k | final train CE | held-out token accuracy |
| --- | --- | --- |
| 0 | 1.1521 | 0.3125 |
| 1 | 0.6841 | 1.0 |
| 2 | 0.7518 | 1.0 |

Accuracy jumps once one token of lookahead is allowed, showing the
sentence-level objective alone suffices to learn a simultaneous translator at
this scale.

## Claim 2 - GRPO latency-quality optimization (Section 3.3): partially reproduced

Real group-relative policy optimization (group size 8, 200 iterations) on a
tabular wait/emit policy; reward = emission quality - beta x average lag.

| beta | initial quality | initial avg lag | final quality | final avg lag |
| --- | --- | --- | --- | --- |
| 0.0 | 0.75 | 3.1315 | 1.0 | 4.1094 |
| 0.05 | 0.75 | 3.1315 | 0.9974 | 2.0052 |
| 0.2 | 0.75 | 3.1315 | 1.0 | 2.0143 |

The latency weight beta measurably moves the trained operating point along
the latency-quality trade-off, which is the mechanism the paper's Section 3.3
relies on.

## Claim 3 - RQ-Transformer multistream decoder (Section 4.1): partially reproduced

Real residual vector quantization (4 levels x 32 codes, Lloyd iterations on
seeded synthetic embeddings): reconstruction MSE by depth = 0.3089, 0.1927, 0.1213, 0.0763 - strictly
decreasing, as residual quantization requires. Multistream factorization:
flat joint vocabulary 1,048,576 entries versus
factorized 128 - the arithmetic that makes
multi-codebook audio decoding tractable. A depth-conditional predictor
achieves cross-entropy 2.8958, 2.8735, 2.8632 against unigram baselines 3.33, 3.3198, 3.3004,
demonstrating exploitable inter-codebook structure. The 3B-parameter model
itself is **not** instantiated.

## Claims 4-6 - benchmarks, human MOS, Italian adaptation: unreplicated

No numbers are reported for these claims. Table 1 needs the released model
and benchmark harnesses; Table 2 is a human study that cannot be reproduced
computationally; Table 3 needs 850 hours of speech and fine-tuning
infrastructure. Reporting invented values for these would be fabrication.

## Reproducibility

`uv run python generate_evidence.py` regenerates `evidence_summary.json` and
this page byte-identically (pinned seeds, no timestamps).
