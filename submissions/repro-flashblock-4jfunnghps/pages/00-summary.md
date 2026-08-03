# FlashBlock Reproduction Summary

This Space provides CPU-only evidence for paper `4jfuNNghPS`, attempt
`ee4b5986-ff11-4f99-9a93-cd8fc43eb04d`, pinned to `arxiv:2602.05305v3` and
snapshot `c68adfe585882f99e8f3dd3ed496aedc650f5b64684955045d04513816cbe106`.

The reproduced evidence supports the FlashBlock mechanism rather than the full
Trado-8B hardware benchmark. The bundle verifies that log-space composition of
block-external and block-internal attention matches full dense attention within
`1e-5`, that cache reuse follows the update-threshold rule, and that analytic
FLOP and memory traffic counts decrease when block-external attention is reused.

The reported token-throughput and generation-quality claims remain model- and
hardware-scale claims. This submission marks them as mechanism-supported, not as
directly reproduced Trado-8B measurements.
