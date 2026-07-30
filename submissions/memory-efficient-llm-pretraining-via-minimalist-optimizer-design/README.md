# SCALE Reproduction

This submission provides CPU-only evidence for two SCALE optimizer mechanism claims from `prvGhNz39e`.

It recomputes column-normalization behavior, last-layer-only momentum state allocation, and optimizer-state memory accounting using independent toy inputs. It records the pinned upstream repository commit and optimizer file hash, but does not vendor upstream source because the repository has no explicit license.

Large-scale LLaMA/C4 perplexity, memory, and 100B-token stability claims are recorded as limitations rather than reproduced measurements.

## Commands

```bash
uv run pytest -q submissions/memory-efficient-llm-pretraining-via-minimalist-optimizer-design/tests
uv run python submissions/memory-efficient-llm-pretraining-via-minimalist-optimizer-design/generate_evidence.py
```
