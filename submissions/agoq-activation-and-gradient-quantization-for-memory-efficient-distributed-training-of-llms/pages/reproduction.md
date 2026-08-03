# AGoQ pinned evidence audit

This audit uses arXiv `2605.00539v2` and
`Eutenacity/AGoQ@006fa0f6318228d1fcd6727f0578c0e548e5cbff`. Ten selected
upstream files are checked by SHA-256 and Git blob ID before use.

## Claim evidence

Exact arithmetic reproduces the Table 1 component sums: BF16 `28U`, COAT
`33/2U` (`16.5U`), and AGoQ `31/4U` (`7.75U`), where
`U = B*S*H*2 bytes`. The printed four-stage pipeline equation yields stored
batch counts `(11, 9, 7, 5)` and exact widths `(4, 44/9, 44/7, 44/5)`.
The paper’s integer widths `(4, 5, 6, 8)` yield storage products
`(44, 45, 42, 40)`, so stage two overshoots the nominal 44-unit target by one;
the release does not specify its rounding policy.

Pinned source supports the activation-quantization integration,
dequantize/accumulate/requantize sequence, and
All-to-All → local reduce → AllGather path. It contains quantize/dequantize
call sites adjacent to GEMMs, but not the fused GPU-kernel body or a reproduced
overhead measurement. These four mechanism/arithmetic claims therefore have
partial evidence.

The 64-GPU Table 2 and 16-Blackwell-GPU Table 3 claims are unavailable because
no distributed training, throughput, convergence, or memory benchmark was
run. Machine-readable computed results are in `evidence.json`; paper statements
are context only.
