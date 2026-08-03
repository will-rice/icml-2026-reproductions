# Numeric Evidence

Paper: PoRigyDOcC
Attempt: 18872478-4b49-464f-b63c-0ee39d354284

This page surfaces the CPU measurements from `evidence_summary.json` for the
official scoring pass.

## Gradient Consistency

- TBPTT segments: 5
- Segment width: 4
- State width: 3
- Parameter count: 21
- Maximum absolute gradient error: 0.000e+00

## Retrieval Isolation

- Retrieval seed: 19
- Retrieved prefix gradient norm: 0.000000
- Carried state gradient norm: 0.628693
- Retrieved prefix changed the forward output: true

## Memory Scaling Proxy

- 4096 context full attention units: 131072.0
- 4096 context segmented execution units: 307515.1
- 4096 context full/segmented ratio: 0.426
- 8192 context full/segmented ratio: 0.819
- 16384 context full/segmented ratio: 1.518
- 32768 context full/segmented ratio: 2.650
- 65536 context full/segmented ratio: 4.223
- 131072 context full attention units: 4194304.0
- 131072 context segmented execution units: 698210.5
- 131072 context full/segmented ratio: 6.007

## Claim Outcomes

- Bound claims: 6
- Verified claims: 1
- Toy claims: 4
- Inconclusive claims: 1
