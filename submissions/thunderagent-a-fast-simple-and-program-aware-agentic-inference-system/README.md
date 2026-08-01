# ThunderAgent Reproduction

CPU-only source audit for `kR4iOTaAOJ`, using the pinned upstream revision
`ThunderAgent-org/ThunderAgent@7ddc8610270e56d3b109eed8796b3a4360fc67c9`.

Run:

```bash
uv run --project submissions/thunderagent-a-fast-simple-and-program-aware-agentic-inference-system python -m thunderagent_repro.run_evidence
uv run --project submissions/thunderagent-a-fast-simple-and-program-aware-agentic-inference-system python -m pytest submissions/thunderagent-a-fast-simple-and-program-aware-agentic-inference-system/tests -q
```
