---
title: RelayCaching Repro
emoji: ⚡
colorFrom: blue
colorTo: indigo
sdk: streamlit
sdk_version: 1.32.0
app_file: app.py
pinned: false
license: mit
short_description: Reproduction of RelayCaching Accelerating LLM Collaboration via Decoding KV Cache Reuse
tags:
- icml-2026
- paper-1tbhBSXcyX
- repro-challenge
---

# ICML 2026 Reproduction: RelayCaching: Accelerating LLM Collaboration via Decoding KV Cache Reuse

**Paper Title:** RelayCaching: Accelerating LLM Collaboration via Decoding KV Cache Reuse
**Paper ID:** `1tbhBSXcyX`
**Authors:** Yingsheng Geng, Yuchong Gao, Weihong Wu, Guyue Liu, Jiang Liu
**Upstream Revision:** `main`
**License:** MIT License
**Space:** `wrice/repro-relaycaching-accelerating-llm-collaboration-via-decoding-kv-cache-reuse`

---

## Reproducibility Claims & Computed Evidence

1. **Decode-to-Prefill Cache Reuse (Figure 5):** Direct reuse of upstream agents' decoding-phase KV caches during downstream prefilling with localized layer/token position rectification.
2. **Macro KV Alignment (Figure 2):** Decoding KV caches remain highly aligned with full-prefill KV caches at macro level (similarity > 0.95).
3. **Accuracy and Cache Reuse (Table 1; Figure 6):** Maintains accuracy comparable to full prefilling while achieving over 80% KV cache reuse across GSM8K, MMLU, and HumanEval multi-agent workflows.
4. **Latency Reduction (Table 2):** Reduces per-agent TTFT by up to 4.7x in latency breakdown experiments.
5. **Cumulative Context Speedup (Figure 8):** Achieves 9.2x average per-agent TTFT speedup over full prefill and 2.5x over KVCOMM as cumulative context grows.
6. **Ablation Studies (Table 3):** Confirms critical-layer rectification, deviation-based token selection, and influence-aware token selection are essential.

---

## Quickstart

Generate evidence:
```bash
python generate_evidence.py
```

Run test suite:
```bash
pytest tests/ -q
```
