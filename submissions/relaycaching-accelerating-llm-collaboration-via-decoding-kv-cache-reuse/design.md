# RelayCaching Reproduction Design

**Paper ID**: `1tbhBSXcyX`  
**Title**: RelayCaching: Accelerating LLM Collaboration via Decoding KV Cache Reuse  
**Slug**: `relaycaching-accelerating-llm-collaboration-via-decoding-kv-cache-reuse`  
**Upstream Revision**: `main`  

## 1. Overview & Key Mechanism

RelayCaching is a training-free inference acceleration method designed for multi-agent LLM collaboration pipelines. In multi-agent systems, downstream agents frequently prefill contexts that contain outputs generated during upstream decoding phases. RelayCaching reuses decoding-phase Key-Value (KV) caches during downstream prefilling, identifying and rectifying localized cache deviations only at critical layer and token positions.

Key Architectural Components:
1. **Decode-to-Prefill Cache Reuse**: Direct reuse of decoding-phase KV caches during downstream prefilling.
2. **Layer-Range Profiler**: Identifies critical layer ranges based on inter-layer correlation and macro alignment profile across Transformer layers.
3. **Token Selector**: Combines deviation-based and influence-based token selection to identify precise KV cache positions requiring rectification.

## 2. Target Claims

1. **Claim 1**: RelayCaching directly reuses upstream agents' decoding-phase KV caches during downstream prefilling and rectifies only selected layer/token positions (Figure 5).
2. **Claim 2**: Decoding KV caches remain highly aligned with full-prefill KV caches at macro level, motivating decode-to-prefill cache reuse (Figure 2).
3. **Claim 3**: RelayCaching maintains accuracy comparable to full prefilling while achieving over 80% KV cache reuse on GSM8K, MMLU, and HumanEval multi-agent workflows (Table 1; Figure 6).
4. **Claim 4**: RelayCaching reduces per-agent TTFT by up to 4.7x in the latency breakdown experiment (Table 2).
5. **Claim 5**: RelayCaching achieves a 9.2x average per-agent TTFT speedup over full prefill and 2.5x over KVCOMM as cumulative context grows (Figure 8).
6. **Claim 6**: Ablations show that combining critical-layer rectification, deviation-based token selection, and influence-aware token selection is needed to balance accuracy and reuse rate (Table 3).

## 3. Implementation Plan & Verification Strategy

- Implement deterministic modular Python components under `submissions/relaycaching-accelerating-llm-collaboration-via-decoding-kv-cache-reuse/`:
  - `relaycaching/cache_reuse.py`: Core KV cache alignment, decode-to-prefill reuse logic, and layer/token rectification algorithms.
  - `relaycaching/profiler.py`: Layer-range profiling and deviation/influence token selection.
  - `generate_evidence.py`: Executes evaluation benchmarks, measures KV cache reuse rates, TTFT speedups, and ablation dynamics, outputting `evidence.json`.
  - `tests/test_relaycaching.py`: Pytest test suite asserting contract correctness, claims verification, and evidence structure.
- Hugging Face Space app (`app.py`, `README.md`) rendering interactive evaluation results and verified paper claims.
