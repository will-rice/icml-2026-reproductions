# Memorization-capacity reproduction

This submission runs a CPU-only toy reimplementation of the paper's
uniform-random-data memorization protocol. It does not reproduce the
full 500K-to-1.5B parameter sweep or treat paper-reported values as
measurements.

## Claim status

- `toy`: GPT-style transformers trained on uniform random data show an empirical memorization-capacity plateau of about 3.6 bits per parameter (Figure 1)
- `toy`: Capacity estimates across model widths and depths support a roughly linear bits-per-parameter scaling law, with bfloat16 to float32 increasing capacity only modestly (Table 1)

## Measurements

| model | dataset | params | memorized bits | bits/param |
| --- | ---: | ---: | ---: | ---: |
| tiny-8 | 8 | 1224 | 24.579597 | 0.020081 |
| tiny-8 | 16 | 1224 | 38.594349 | 0.031531 |
| tiny-16 | 8 | 3968 | 65.521833 | 0.016513 |
| tiny-16 | 16 | 3968 | 76.243511 | 0.019215 |
