# Hive Reproduction Evidence

This submission audits released artifacts for paper `vCc2NAe0OS`, focusing on
the Hive dataset construction pipeline and released Hive-trained AudioSep and
FlowSep artifacts.

The evidence is CPU-only. It does not download multi-GB audio archives or model
weights, and it does not report paper benchmark tables as recomputed
measurements.

## Commands

```bash
uv run --project submissions/a-semantically-consistent-dataset-for-data-efficient-query-based-universal-sound-separation python submissions/a-semantically-consistent-dataset-for-data-efficient-query-based-universal-sound-separation/generate_evidence.py
uv run --project submissions/a-semantically-consistent-dataset-for-data-efficient-query-based-universal-sound-separation python -m pytest submissions/a-semantically-consistent-dataset-for-data-efficient-query-based-universal-sound-separation/tests -q
```
