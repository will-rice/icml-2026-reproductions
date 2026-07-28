# Steer Like the LLM: Activation Steering that Mimics Prompting - Reproduction

Independent Reproduction Evidence for ICML 2026 Agent Repro Challenge.

## Paper Details
- **Title:** Steer Like the LLM: Activation Steering that Mimics Prompting
- **Paper ID:** `06Nk3dJDMq`
- **ArXiv:** `2605.03907`
- **Upstream Revision:** `arxiv:2605.03907+github:Nokia-Bell-Labs/steer-like-the-llm@main`

## Reproduction Claims & Results
1. **Activation Subtraction & PSR Training (Figure 1):** Verified.
2. **Token-Dependent Intervention Strengths (Figure 2):** Verified.
3. **PSR Coefficient Estimation & Objectives (Sections 3.4 & 3.5):** Verified.
4. **Persona Vectors Coherence (Table 1):** Verified.
5. **AxBench Gemma Layer Subsets Comparison (Table 3):** Verified.
6. **Accumulated PSR Interventions RMSE (Figure 3):** Verified.

## Executing Evidence
```bash
python -m steer_like_llm.evidence_bundle
pytest
```
