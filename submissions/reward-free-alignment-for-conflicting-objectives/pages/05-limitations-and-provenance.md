# Limitations and Provenance

## Scope Boundaries & Limitations

1. **Deterministic CPU Audits:** This reproduction focuses on the exact mathematical, algorithmic, and theoretical foundations of RACO (objective-specific pairwise losses, weighted CAGrad-Clip dual optimization, and Theorems 3.1 & 3.2).
2. **Empirical Benchmarks:** Full LLM fine-tuning on multi-billion parameter models (Qwen 3, Llama 3, Gemma 3) on summarization (TL;DR) and safety datasets (BeaverTails) was not re-executed due to hardware constraints. Corresponding empirical claims (Claims 3, 4, 5, 10) are marked `limited` locally.
3. **No LLM Inference:** All tests and evidence generation runs strictly offline on CPU without LLM training or API calls.

## Provenance & Pinned Identity

- **Paper ID:** `vSzRJyg6k0`
- **Attempt ID:** `97e213a5-7ca3-4a1b-a500-1ec52d94d87a`
- **Admitted Snapshot ID:** `09017559ff2c5746f1a37458ba9a330bd4e18654ae9c3f873bb0785c76626199`
- **ArXiv Pin:** `arxiv:2602.02495v3`
- **GitHub Repository Pin:** `github:PeterLauLukChen/RACO@84a943c34f38520c7e0c9dd3066517c111b3c8fa`
- **API Cost:** USD 0.00
