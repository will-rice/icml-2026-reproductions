# Steer Like the LLM: Activation Steering that Mimics Prompting - Reproduction

Independent Reproduction Evidence for ICML 2026 Agent Repro Challenge.

## Paper Details
- **Title:** Steer Like the LLM: Activation Steering that Mimics Prompting
- **Paper ID:** `06Nk3dJDMq`
- **ArXiv:** `2605.03907`
- **Attempt ID:** `4e292a9a-9ba5-44ac-a309-f3fe685a0643`
- **Snapshot ID:** `d84e8174f494825bf70b4d9201754ddbb3e065bbea0b46d1f90da85949bb882b`
- **Upstream Revision:** `arxiv:2605.03907+github:Nokia-Bell-Labs/steer-like-the-llm@3d916c618d146c5d657f055e432a432b0fa493c6`

## Reproduction Claims & Results
1. **Activation Subtraction & PSR Training (Figure 1):** Verified.
2. **Token-Dependent Intervention Strengths (Figure 2):** Verified.
3. **PSR Coefficient Estimation & Objectives (Sections 3.4 & 3.5):** Verified.

The model-scale Persona Vectors, AxBench, and accumulated-RMSE claims are
recorded as unreplicated non-target claims in `results/results.json`; the
included reduced-scale outputs are sanity checks only.

## Executing Evidence
```bash
python -m steer_like_llm.evidence_bundle
pytest
```
