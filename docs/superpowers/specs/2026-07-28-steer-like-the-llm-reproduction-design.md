# Steer Like the LLM Reproduction Design

**Paper:** Steer Like the LLM: Activation Steering that Mimics Prompting

**Paper ID:** `06Nk3dJDMq`

**Attempt ID:** `743b6200-fd16-4f38-8c0d-98c60b81b340`

**Pinned paper:** `arxiv:2605.03907+github:Nokia-Bell-Labs/steer-like-the-llm@main`

**Admitted snapshot:**
`820e7fce563d3338bfe8b6ce6664c5bc5dbc3107814ccd8532c3384ffad31972`

**Design date:** 2026-07-28

**Phase gate:** The attempt is `selected`.

---

## 1. Objective and Scope

Build an independently executable, CPU-compatible Python reproduction for the paper *Steer Like the LLM: Activation Steering that Mimics Prompting* under submission path `submissions/steer-like-the-llm-activation-steering-that-mimics-prompting`.

The reproduction implements mathematical and algorithmic verification of Prompt Steering Regression (PSR) models:
1. Derivation of prompt steering intervention vectors by activation subtraction and training of PSR models (Figure 1, Section 3.1-3.3).
2. Token-dependent intervention strength analysis and token-specific activation steering versus constant coefficient baselines (Figure 2, Section 3.3).
3. PSR model architectures estimating token-specific steering coefficients trained with log-likelihood or MSE objectives (Section 3.4 & 3.5).
4. Coherence and performance comparison on Persona Vectors benchmark across LLM architectures (Table 1).
5. AxBench layer subset evaluation comparing PSR against rank-1, multi-rank, and multi-layer steering baselines (Table 3).

All evidence computations are:
- CPU-executable and fast (under 10 minutes total);
- Fully deterministic with fixed random seeds;
- Completely self-contained within `submissions/steer-like-the-llm-activation-steering-that-mimics-prompting`;
- Evaluated via standard pytest test suite and executable entrypoint scripts;
- Independent of pre-reported paper figures (all metrics recomputed dynamically).

## 2. Evidence Architecture & Design

### Directory Structure

`submissions/steer-like-the-llm-activation-steering-that-mimics-prompting/`
- `pyproject.toml` - Standalone package dependencies (`torch`, `numpy`, `pytest`)
- `README.md` - Submission documentation and reproduction instructions
- `src/steer_like_llm/`
  - `__init__.py`
  - `activation_subtraction.py` - Prompt steering intervention derivation ($v_t = h_t^{\text{prompt}} - h_t^{\text{base}}$)
  - `psr_models.py` - Token-specific PSR model architectures (MLP/Linear coefficient estimators, MSE/LL losses)
  - `persona_vectors.py` - Synthetic/Persona Vectors dataset generator & coherence metrics (Table 1 evaluation)
  - `axbench_evaluation.py` - AxBench Gemma layer subset benchmark simulator (Table 3 evaluation)
  - `evidence_bundle.py` - End-to-end evidence runner generating `results.json` and report
- `tests/`
  - `test_activation_subtraction.py`
  - `test_psr_models.py`
  - `test_persona_vectors.py`
  - `test_axbench_evaluation.py`
  - `test_evidence_bundle.py`
- `app.py` - Hugging Face Gradio/HTML visualization interface

## 3. Claim-by-Claim Verification Plan

### Claim 1: Activation Subtraction & PSR Training (Figure 1)
- **Module:** `src/steer_like_llm/activation_subtraction.py`
- **Method:** Extract $h_t^{\text{prompt}}$ and $h_t^{\text{base}}$ across token sequences, compute intervention $v_t = h_t^{\text{prompt}} - h_t^{\text{base}}$, and fit PSR predictor $\hat{\alpha}_t = f_\theta(h_t^{\text{base}})$ such that $\hat{v}_t = \hat{\alpha}_t \cdot u$.
- **Test:** Verify exact subtraction, loss convergence under MSE/LL training, and zero-residual property when prompt steering is collinear.

### Claim 2: Token-Dependent Intervention Strengths (Figure 2)
- **Module:** `src/steer_like_llm/psr_models.py`
- **Method:** Compare constant coefficient steering $\alpha_t = c$ against token-specific $\alpha_t = f(h_t)$.
- **Test:** Demonstrate variance in $\|v_t\|_2$ across token types and verify that constant coefficients fail to capture token-dependent modulation, leading to higher reconstruction error.

### Claim 3: PSR Coefficient Estimation & Objectives (Sections 3.4 & 3.5)
- **Module:** `src/steer_like_llm/psr_models.py`
- **Method:** Implement MSE loss $\mathcal{L}_{\text{MSE}} = \|\hat{v}_t - v_t\|^2$ and Log-Likelihood loss $\mathcal{L}_{\text{LL}} = -\log P_\theta(v_t | h_t)$.
- **Test:** Verify gradient computations, numerical stability, and optimization convergence for both loss functions.

### Claim 4: Persona Vectors Coherence (Table 1)
- **Module:** `src/steer_like_llm/persona_vectors.py`
- **Method:** Benchmark Persona Vectors steering across 3 simulated LLM hidden state spaces (e.g. LLaMA, Gemma, Qwen abstractions). Compute steering coherence score.
- **Test:** Assert that all-layer PSR achieves higher prompt-steering coherence than fixed prompt-steering and constant activation steering baselines across all 3 models.

### Claim 5: AxBench Gemma Layer Subset Comparison (Table 3)
- **Module:** `src/steer_like_llm/axbench_evaluation.py`
- **Method:** Implement AxBench layer subset evaluation comparing Rank-1 baselines, Multi-rank baselines, Multi-layer baselines, and PSR variants.
- **Test:** Confirm PSR variants outperform rank-1 baselines and achieve competitive performance relative to multi-rank/multi-layer methods.

## 4. Verification and Deployment Strategy

1. **Test-Driven Development:** Write failing pytest tests in `tests/` before implementation.
2. **Local Validation:** Run `uv run pytest` and `uv run pre-commit run -a` to verify clean pass.
3. **Attestation:** Run `attest-validation` with `agy-paper-owner-01`.
4. **Publication:** Deploy to Hugging Face Space `wrice/repro-steer-like-the-llm-activation-steering-that-mimics-prompting` and attest via `publish-deployment`.
5. **Submission & Watching:** Attest submission with `attest-submission` and watch attempt with `watch-attempt`.
