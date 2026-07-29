# Steer Like the LLM Paper-Owner Design

**Paper:** Steer Like the LLM: Activation Steering that Mimics Prompting

**Paper ID:** `06Nk3dJDMq`

**Attempt ID:** `4e292a9a-9ba5-44ac-a309-f3fe685a0643`

**Owner:** `codex-paper-owner-01`

**Fencing token:** `1`

**Admitted snapshot:** `d84e8174f494825bf70b4d9201754ddbb3e065bbea0b46d1f90da85949bb882b`

**Upstream pin:** `arxiv:2605.03907+github:Nokia-Bell-Labs/steer-like-the-llm@3d916c618d146c5d657f055e432a432b0fa493c6`

## Scope

Build a CPU-executable evidence bundle under
`submissions/steer-like-the-llm-activation-steering-that-mimics-prompting`
for the three bound methodological claims:

1. Prompt-steering intervention vectors are activation differences between
   prompted and unprompted forward passes, and PSR models approximate
   successful prompt interventions.
2. Prompt steering uses token-dependent intervention strengths, motivating
   token-specific coefficients rather than a constant steering coefficient.
3. PSR models estimate token-specific steering coefficients from activations
   and can be trained with MSE or log-likelihood objectives.

The quantitative Persona Vectors, AxBench, and accumulated-RMSE claims remain
out of scope for full-score reproduction because they require model-scale
benchmarks. The evidence may include toy or sanity checks for those paths, but
must not present them as reproduced full benchmark results.

## Evidence Plan

Use the released repository as an upstream artifact reference, but recompute
all reported evidence locally. Implement deterministic small-array and
small-model checks that:

- verify exact activation subtraction and per-token intervention construction;
- compare token-specific coefficient recovery against a constant-coefficient
  baseline on controlled synthetic activations;
- train simple linear/MLP PSR estimators with MSE and Gaussian
  log-likelihood losses and record convergence plus provenance.

## Validation

Write failing pytest tests before implementation. The evidence bundle must
record the upstream commit, commands, dependency versions, random seeds, output
JSON, and per-claim statuses. Local validation requires the submission pytest
suite and repository pre-commit, excluding the archival NAPE snapshot as
required by workspace policy.
