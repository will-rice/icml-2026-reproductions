# Reproduction design: To Grok Grokking

Attempt: `4b8e5145-c432-4786-ace1-6270e8a2e192`
Paper: `5nNNVY8NW4`
Snapshot: `81062818c98b62c0ef8f7571d12365a765424ca5f17d865481b9ba1af45f0a0b`
Upstream pin: `arxiv:2601.19791v4`

## Target claims

1. The paper proves end-to-end grokking for zero-teacher ridge regression,
   including early training overfitting, delayed poor generalization, and
   eventual low generalization error (Theorem 4.1).
2. Separate theorem statements decompose the trajectory into training-loss
   convergence, poor generalization during overfitting, and eventual
   generalization (Theorems 4.4-4.6).
3. Decreasing weight decay and sample size can amplify grokking time in
   ridge-regression simulations, matching the paper's quantitative
   hyperparameter predictions (Figure 2).

## Evidence plan

The submission will not treat paper-reported values as reproduced
measurements. It will build a CPU-only evidence bundle with two layers:

- A theorem-audit table extracted from the arXiv v4 claim structure. The audit
  records assumptions, theorem dependencies, and whether each target claim is a
  proof-structure claim, an empirical claim, or unsupported by executable
  evidence.
- A deterministic ridge-regression simulator using synthetic Gaussian
  teacher-student data, gradient descent, and L2 weight decay. The simulator
  measures training loss, test loss, an overfit time, a grokking time, and a
  delay interval across small hyperparameter sweeps.

The simulation will target qualitative evidence only: small CPU settings should
show that lower weight decay or fewer samples can increase the delay between
training fit and test-loss improvement. Full theorem proof checking, arbitrary
teacher functions, and two-layer ReLU experiments are out of scope and will be
marked unreplicated.

## Validation

The implementation will include a failing test first for the grokking-metric
detector and evidence bundle metadata. Validation will run:

- `uv run python submissions/to-grok-grokking-provable-grokking-in-ridge-regression/generate_evidence.py`
- focused pytest for this submission
- root `uv run pytest -q`
- skill quick validation
- `uv run pre-commit run -a`

The Space source will include `app.py`, a machine-readable evidence bundle, and
`pages/00-summary.md` for the scoring surface.
