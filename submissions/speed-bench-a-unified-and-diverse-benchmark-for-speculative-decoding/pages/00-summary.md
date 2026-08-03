# SPEED-Bench Evidence Summary

This Space recomputes selected evidence for ICML 2026 challenge paper
`Rl2uQlCoQX` from pinned public artifacts. The audit uses the Hugging Face
dataset `nvidia/SPEED-Bench` at revision
`487aa718444e816458d1a0a52bfce7a454285cf4` and the Model-Optimizer repository
at commit `a23390dbb6e52b0c028f3e9455a74da824c88735`.

The released dataset card verifies the qualitative split, fixed throughput
configs from 1K through 32K, 880 qualitative samples, and 1,536 examples for
each throughput bucket. The same card reports semantic-similarity comparisons
against SpecBench and random selection; the audit checks those values directly.
Speculative-decoding speedup and acceptance-length tables are not recomputed
because they require GPU inference and no machine-readable Table 1 result file
was found in the pinned repository.
