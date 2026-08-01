# Training-Inference Consistent Segmented Execution

Paper: PoRigyDOcC
Attempt: 18872478-4b49-464f-b63c-0ee39d354284

This reproduction provides CPU-only evidence for the formal and architectural
claims in arXiv:2605.11744v1.

## Evidence Summary

- TBPTT gradient equality is checked against an explicitly truncated objective
  in a deterministic PyTorch surrogate.
- Retrieved prefixes are verified to affect the forward output while remaining
  detached from gradient flow.
- A peak-memory proxy validates the direction and approximate 128K scaling
  ratio claimed for segmented execution.
- Full LongBench-E and FlashAttention GPU measurements are marked
  inconclusive or toy when direct artifacts are unavailable.

Run:

```bash
uv run python submissions/training-inference-consistent-segmented-execution-for-long-context-llms/generate_evidence.py
```
