# Reproduction Report: Efficient Parallel Samplers for Recurrent-Depth Models

- **Paper Title**: Efficient Parallel Samplers for Recurrent-Depth Models and Their Connection to Diffusion Language Models
- **arXiv ID**: `2510.14961v1`
- **OpenReview ID**: `h7WBYYJF1Q`
- **Attempt ID**: `534db42c-5b16-4f00-9a7d-a47056fc9dd4`
- **Provenance Token**: `arxiv:2510.14961v1+pdf-sha256:74e7985abe41ee2a75914a65e3778a15353fb0c0964d6ea34e7bfeb1f18312c8+source-sha256:60a795d123a2d2d642971834b6e0cba6dda80b5dfcd539f78d01639582d9c41d+github:seal-rg/recurrent-pretraining@1ea7220ec7eb42d13e89db0663df254d0bcdc28e+git-blob:recpre/raven_modeling_minimal.py@0e83a0766644df9113a8923f43350c6a1b5a182c`

---

## Executive Summary

| Claim | Text | Status | Primary Finding |
|---|---|---|---|
| **Claim 1** | The sampler decodes new tokens every forward pass while refining latent states for those tokens in parallel through recurrent depth (Section 3.1). | **`partial`** | Verified released sampler control flow AST and wavefront schedule, but decoding occurs after `inner_recurrence` loop rather than after every single inner step. |
| **Claim 2** | The paper proves the sampler is strictly more expressive than baseline autoregressive generation under the same time budget on modern hardware (Theorem 4.2). | **`unavailable`** | Citation mismatch (Theorem 4.2 is prefilling; decoding is Theorem 4.4) and missing independent proof reproduction. |

---

## 1. Claim 1 Audit & Wavefront Mechanism (`partial`)

### AST Analysis of Released Code
- **File**: `vendor/recurrent-pretraining/recpre/raven_modeling_minimal.py`
- **Git Blob**: `0e83a0766644df9113a8923f43350c6a1b5a182c`
- **Dispatcher**: `generate()` dispatches to `generate_diffusion_style()`.
- **Defaults**: `headway=1`, `inner_recurrence=4`, `freeze_strategy='latent-diff'`, `max_wavefront=128`.

### Invariants Verified
1. **Parallel Refinement**: Each outer iteration runs `inner_recurrence` steps across all active positions in the wavefront.
2. **Token Append**: Exactly `headway` (default 1) new candidate position is appended per outer step.
3. **Wavefront Limit**: Active state width is constrained by `max_wavefront`.

---

## 2. Claim 2 Audit & Expressiveness Theorem (`unavailable`)

### Citation Audit Findings
- **Theorem 4.2**: Prefilling depth scaling theorem.
- **Theorem 4.4**: Decoding same-runtime expressiveness theorem (conditional on $r > 1$, KV sharing, $W \le L_*$).
- **Remark 4.5**: Hardware and memory bandwidth interpretation.

The claim remains **`unavailable`** because the released v1 source does not include a complete, independently checkable proof.

---

## 3. Provenance & Digest Verification

- `arxiv_submission.tex` SHA-256: `cdc058830d1e51f631e4fb8d1f2de0b79de91670fd4111646fe624f8c258d3b8`
- `raven_modeling_minimal.py` SHA-256: `18fcacd53fb5696a76c0d3bda44480f2f3900aa9659c137a08962c593a9a9e42`
- `raven_modeling_minimal.py` Git Blob: `0e83a0766644df9113a8923f43350c6a1b5a182c`
- `LICENSE` SHA-256: `bc6c264d8ba4450599cf95c4699c6b82142f32ca1ecd91011c17b50a5a36a2f5`
- `ATTRIBUTION.md` SHA-256: `79775b50c72988b90eae75ef87e9d4df9dbd0bfceefaed60b398656a88d8a735`
- Claim 1 SHA-256: `d0da87ee16f7485d3dff369e7465f66299c55ac003a54e1cf8c00b3a0ad8b265`
- Claim 2 SHA-256: `2e15221c8b5516b0ab705e29a3d7c5d924ed5f0187c970a0caf60a1402757804`

---

## Limitations

1. **No 3.5B Checkpoint Execution**: No Huginn-0125 model weights were loaded or evaluated.
2. **No Hardware Timing**: No A100 GPU speedup or wall-clock benchmarking was conducted.
3. **No Official Verdict Claim**: Evidence statuses (`partial`, `unavailable`) reflect code/paper audit results and do not replace official challenge verdicts.
