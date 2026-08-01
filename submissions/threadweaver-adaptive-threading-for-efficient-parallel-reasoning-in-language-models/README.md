# ThreadWeaver Reproduction Evidence

This submission is a CPU-only evidence package for paper `Efq2VvYk1o`,
`ThreadWeaver: Adaptive Threading for Efficient Parallel Reasoning in Language
Models`.

It recomputes only static artifact checks and deterministic toy mechanisms:

- released artifact audit at upstream revision
  `b944f0139209258caa34fa7dea6a58c2502912fa`;
- `<Parallel>/<Outlines>/<Thread>` parser consistency;
- token-prefix-trie ancestor attention masking;
- critical-path token latency accounting;
- P-GRPO mean-centered advantage normalization.

It does not treat the paper's reported accuracy or latency values as
reproduced measurements. The public artifact states that SFT and RL require a
single node with `8x80G A100 or H100` GPUs, so Qwen3-8B benchmark and speedup
claims are marked `unreplicated`.

Run:

```bash
THREADWEAVER_UPSTREAM=/path/to/threadweaver \
python generate_evidence.py
pytest -q
```
