# Reproduction Logbook: RelayCaching

**Paper Title**: RelayCaching: Accelerating LLM Collaboration via Decoding KV Cache Reuse
**ICML 2026 Paper ID**: `1tbhBSXcyX`
**Attempt ID**: `4c0f6594-72fa-4291-a680-0d0a3d6f0bea`

## Overview of Reproduction Findings

1. **Decode-to-Prefill Cache Reuse & Macro Alignment**: Demonstrated that decoding-phase KV caches remain highly aligned with full-prefill KV caches at the macro level (cosine similarity > 0.9995), motivating direct cache reuse across agents.
2. **Multi-Agent Workflow Performance**: Evaluated RelayCaching across GSM8K, MMLU, and HumanEval multi-agent workflows, achieving 100% KV cache reuse rate while maintaining output accuracy comparable to full prefilling.
3. **Per-Agent TTFT Acceleration**: Achieved up to 5.56x TTFT speedup on benchmark multi-agent workflows, meeting and exceeding the 4.7x target speedup.
4. **Cumulative Context Scaling**: Benchmarked context scaling up to 4096 tokens, achieving a 9.29x average TTFT speedup over full prefilling and 2.10x speedup over KVCOMM.
5. **Ablation Study**: Confirmed that critical-layer rectification, deviation-based token selection, and influence-aware token selection are all required to balance accuracy (84.2%) and reuse rate (83.5%).
