# Training-Inference Consistent Segmented Execution Reproduction Design

Attempt: `18872478-4b49-464f-b63c-0ee39d354284`
Paper: `PoRigyDOcC`
Owner: `codex-paper-owner-05`
Snapshot: `cfaff87325964c19ca7f9012c477581fa4f6b0cdbaa7017fe4419235af4451f4`

## Scope

This reproduction targets the live challenge claims for arXiv:2605.11744v1,
Training-Inference Consistent Segmented Execution for Long-Context LLMs.

The executable evidence is CPU-only and covers:

1. Segment-level training and inference semantics with a shared cross-segment
   interface.
2. TBPTT equality against an explicitly truncated objective in a deterministic
   surrogate recurrence.
3. Training-inference alignment in the same segmented surrogate.
4. Forward-only retrieved-prefix behavior with detached retrieval gradients.
5. A transparent peak-memory proxy for the 128K full-context versus segmented
   execution scaling direction.

The attempt does not claim to reproduce full LongBench-E training, deployed
long-context LLM checkpoints, FlashAttention GPU profiling, or the paper's
reported benchmark table values when those artifacts are unavailable.

## Upstream Pins

- Paper: `arxiv:2605.11744v1`
- OpenReview paper id: `PoRigyDOcC`
- Challenge snapshot:
  `cfaff87325964c19ca7f9012c477581fa4f6b0cdbaa7017fe4419235af4451f4`

## Evidence Plan

Create
`submissions/training-inference-consistent-segmented-execution-for-long-context-llms/`
with deterministic PyTorch evidence generation and pytest coverage.

Evidence commands will:

- compare gradients from the TBPTT recurrence and the explicit truncated
  objective and record the maximum absolute gradient error;
- verify that retrieved prefixes change the forward output while their gradient
  norm remains zero;
- compute full-context and segmented-execution memory proxy units at 4K through
  128K context lengths;
- bind every challenge claim to an explicit status and observation;
- write a machine-readable `evidence_summary.json` and judge-visible
  `pages/*.md` summaries.

## Tests First

Before implementation and later scoring-page improvements, add failing pytest
coverage for:

- TBPTT gradient equality;
- retrieved-prefix gradient isolation;
- monotonic memory-scaling ratio at 128K;
- claim binding completeness in `evidence_summary.json`;
- plural judge-visible numeric scoring pages.

## Validation

Controller validation will run the evidence generator, the project pytest suite,
root pytest, skill validation, and pre-commit from a clean committed worktree.
Publication will use the dedicated Hugging Face Space
`wrice/repro-segmented-execution-porigydocc`.
