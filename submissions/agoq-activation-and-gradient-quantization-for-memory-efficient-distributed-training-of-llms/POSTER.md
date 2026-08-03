# AGoQ: pinned arithmetic and source evidence

Paper: `arXiv:2605.00539v2`
Official source: `Eutenacity/AGoQ@006fa0f6318228d1fcd6727f0578c0e548e5cbff`

## Exact activation-memory audit

| Method | QKV | Attention | Linear 1 | RMSNorm | FFN 1 | Activation | FFN 2 | Total |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| BF16 | 1 | 5 | 1 | 4 | 1 | 12 | 4 | 28 |
| COAT | 1 | 5 | 1 | 1 | 1/2 | 6 | 2 | 33/2 |
| AGoQ | 0 | 5 | 1/4 | 1/2 | 0 | 2 | 0 | 31/4 |

These totals are exact arithmetic over paper-transcribed components, not
runtime memory measurements.

## Pipeline equation audit

- Stored batches in device order: `(11, 9, 7, 5)`
- Exact widths from the printed equation: `(4, 44/9, 44/7, 44/5)`
- Paper-reported integer widths: `(4, 5, 6, 8)`
- Resulting storage products: `(44, 45, 42, 40)`

The reported allocation has a one-unit overshoot at device 2. No rounding rule
is available in the pinned paper/source, so none is inferred.

## Pinned source trace

The source audit verifies activation quantization integration, BF16 local
gradient accumulation around quantized storage, and the quantized
All-to-All/local reduction/AllGather route. Quantize/dequantize call sites are
adjacent to GEMM calls in selected Transformer Engine changes. A fused GPU
kernel body is absent from those selected sources, so overhead reduction is not
claimed as reproduced.

## Claim disposition

| Claims | Status | Scope |
|---|---|---|
| 1–4 | partial | Deterministic arithmetic and released-source mechanism evidence |
| 5 | unavailable | Table 2 requires 64 GPUs |
| 6 | unavailable | Table 3 requires 16 NVIDIA Blackwell GPUs |

No distributed training, throughput, convergence, or paper-scale memory
benchmark was run. Paper statements remain context, separate from reproduced
arithmetic and source observations.
