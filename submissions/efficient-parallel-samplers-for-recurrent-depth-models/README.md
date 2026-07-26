# Reproduction: Efficient Parallel Samplers for Recurrent-Depth Models

This directory contains the reproduction project for:
**"Efficient Parallel Samplers for Recurrent-Depth Models and Their Connection to Diffusion Language Models"** (OpenReview ID: `h7WBYYJF1Q`, arXiv: `2510.14961v1`).

Attempt ID: `534db42c-5b16-4f00-9a7d-a47056fc9dd4`

## Target Claims & Evidence Status

1. **Claim 1 (Wavefront Sampler Mechanism)**:
   - *Text*: "The sampler decodes new tokens every forward pass while refining latent states for those tokens in parallel through recurrent depth (Section 3.1)."
   - *Status*: **`partial`**
   - *Findings*: Source AST inspection of `recpre/raven_modeling_minimal.py` confirms `generate()` dispatches to `generate_diffusion_style()`. The control flow executes `inner_recurrence` steps per outer iteration across active latent states, decodes logits for the active wavefront, appends new positions, and bounds active width via `max_wavefront`. However, decoding occurs once after the inner-recurrence loop per outer step rather than after each single inner step.

2. **Claim 2 (Expressiveness Theorem)**:
   - *Text*: "The paper proves the sampler is strictly more expressive than baseline autoregressive generation under the same time budget on modern hardware (Theorem 4.2)."
   - *Status*: **`unavailable`**
   - *Findings*: Citation audit of `arxiv_submission.tex` reveals a citation mismatch: Theorem 4.2 addresses prefilling depth vs width scaling, while the same-runtime decoding result is Theorem 4.4 (conditional on $r > 1$, KV sharing, $W \le L_*$). The released source does not contain an independently checkable proof for strict hardware-dependent expressiveness.

## Running Evidence Generation & Tests

Generate evidence artifacts:
```bash
uv run --project submissions/efficient-parallel-samplers-for-recurrent-depth-models \
  python -m recurrent_sampler_repro.evidence \
  --project-root submissions/efficient-parallel-samplers-for-recurrent-depth-models
```

Run unit tests:
```bash
uv run --project submissions/efficient-parallel-samplers-for-recurrent-depth-models \
  python -m pytest \
  submissions/efficient-parallel-samplers-for-recurrent-depth-models/tests
```

## Structure

- `src/recurrent_sampler_repro/`: Core verification logic, AST parser, schedule simulator, theorem auditor, and evidence generator.
- `tests/`: Complete test suite covering claim bindings, provenance, source audit, schedule invariants, theorem audit, output determinism, and static Space assets.
- `vendor/`: Pinned immutable upstream source files.
- `evidence/`: Generated deterministic evidence JSON bundles and report.
- `space/`: Static Hugging Face Space application assets.
