# Training-Inference Consistent Segmented Execution Reproduction

This submission contains CPU-only evidence for ICML 2026 paper `PoRigyDOcC`.

It does not claim to reproduce full LLaMA2 LongBench-E training or GPU
FlashAttention measurements. It verifies the formal and architectural claims
with deterministic surrogate checks and marks benchmark-scale claims as
inconclusive or toy when direct artifacts are unavailable.

Run:

```bash
uv run pytest -q submissions/training-inference-consistent-segmented-execution-for-long-context-llms/tests/test_segmented_execution.py
uv run python submissions/training-inference-consistent-segmented-execution-for-long-context-llms/generate_evidence.py
```

The generated evidence bundle is `evidence_summary.json`.
