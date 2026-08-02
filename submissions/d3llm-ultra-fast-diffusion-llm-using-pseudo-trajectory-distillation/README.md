# d3LLM Reproduction Evidence

This submission provides a CPU-only, deterministic audit for ICML 2026 paper
`rzBAQT2Fkg`, "d3LLM: Ultra-Fast Diffusion LLM using Pseudo-Trajectory
Distillation".

It recomputes AUP rankings from the released leaderboard data, tests toy
instances of the pseudo-trajectory and entropy multi-block decoding mechanics,
and records released throughput-table consistency. It does not claim fresh
large-model inference or fresh GPU throughput measurements.

Run:

```bash
uv run python generate_evidence.py --output-dir evidence
uv run python -m pytest tests -q
```
