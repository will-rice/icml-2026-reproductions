# AGoQ evidence design

The implemented evidence flow is:

```text
pinned paper transcription + hash-verified selected source
                         |
         exact rational and semantic audits
                         |
             canonical evidence.json
                         |
               README / poster / Space
```

The immutable inputs bind paper `arXiv:2605.00539v2` and official repository
commit `006fa0f6318228d1fcd6727f0578c0e548e5cbff`. Source verification fails on
repository, revision, path, byte count, SHA-256, or Git blob drift.

The arithmetic layer parses Table 1 values as exact rational numbers. It does
not encode expected totals in production code. The pipeline layer preserves
the printed equation, device ordering, exact fractional widths, and the
paper-reported discrete allocation as separate facts.

The semantic layer parses verified Python sources and checks activation
quantization, local gradient accumulation, quantized collective, and pipeline
schedule relationships. Quantize/dequantize and GEMM call-site adjacency is
partial evidence only: the selected source lacks the fused GPU kernel body.

The canonical bundle binds all six live claims and the controller attempt.
Claims 1–4 remain partial. Table 2 and Table 3 training claims remain
unavailable because this CPU audit does not have the required 64-GPU or
16-NVIDIA-Blackwell environments.

Presentation code rebuilds the canonical evidence at import and rejects
committed-byte drift. It displays evidence but does not recompute training
results or invent measurements.
