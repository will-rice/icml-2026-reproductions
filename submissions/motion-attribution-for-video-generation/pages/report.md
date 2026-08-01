# Reproduction Report: Motion Attribution for Video Generation (Motive)

**Paper ID:** `zAl9heLw4q`

## Scope

CPU-only, toy-scale mechanism reproduction on deterministic synthetic videos
with known moving regions. The VBench fine-tuning comparisons (Table 1) and
the 74.1% human-preference study (Table 2) are **unreplicated** - they need
trained video models and human raters. An earlier revision fabricated those
numbers; they have been removed and nothing here restates them.

## Motion-weighted attribution localizes true motion (Sections 3.4, Figure 1): partially reproduced

On a video with one moving square over a static textured background, the
motion mask concentrates on truly-moving patches:

| measurement | value |
| --- | --- |
| mean mask weight on moving patches | 0.5488 |
| mean mask weight on static patches | 0.0 |
| fraction of total mask weight on true motion | 1.0 |

With uniform gradients, motion weighting shrinks the attribution norm from
313.5347 to
101.4323, i.e. static-appearance gradients
are suppressed and dynamic regions dominate the score.

## Frame-length bias fix (Section 3.3, Figure 4): partially reproduced

Measured attribution norms grow with clip length on statistically identical
clips (lengths [8, 16, 32]): raw scores
[101.5599, 141.8283, 205.0041] (ratio
2.02x longest/shortest);
after S_raw / sqrt(T/T_ref) normalization:
[np.float64(155.1353), np.float64(153.1921), np.float64(156.5745)] (ratio
1.01x).

Ranking with known ground truth: a strongly-moving 8-frame clip versus a
weakly-moving 32-frame clip scores [121.3788, 186.0043] raw
(ranking ['weak_long', 'strong_short']) and
[np.float64(191.9168), np.float64(147.0493)] normalized (ranking
['strong_short', 'weak_long']): normalization recovers the truly more
influential short clip.

## Influence tracks dynamics, not magnitude (Figure 6): partially reproduced

An incoherent jitter clip has higher raw motion energy
(0.1667) than a coherent translating clip
(0.0258), yet the coherent clip's
motion-masked field is far more similar to the target dynamic (cosine
0.6433 versus
0.1649). High motion magnitude alone does not
make a clip influential for a target dynamic, matching the paper's Figure 6
mechanism.

## Unreplicated claims

Table 1 (VBench improvements) and Table 2 (74.1% human preference) are
reported without numbers: producing them requires fine-tuning video
generation models and running a human study. Fabricating them would be
misconduct.

## Reproducibility

`uv run python generate_evidence.py` regenerates `evidence_summary.json` and
this page byte-identically (pinned seeds, no timestamps).
