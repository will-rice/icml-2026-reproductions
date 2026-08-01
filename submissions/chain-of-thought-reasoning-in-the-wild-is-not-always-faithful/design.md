# Chain-of-Thought Reasoning In The Wild Is Not Always Faithful - Reproduction Design

**Paper ID**: `NUyt4uxzx0`
**Title**: Chain-of-Thought Reasoning In The Wild Is Not Always Faithful
**Slug**: `chain-of-thought-reasoning-in-the-wild-is-not-always-faithful`
**Upstream Revision**: `main`

## 1. Overview & Key Mechanism

This paper investigates whether Chain-of-Thought (CoT) reasoning in frontier Large Language Models (LLMs) remains faithful during natural, non-adversarial usage. The authors introduce evaluation frameworks including:
1. **IPHR (In-the-wild Paired Hint Rephrasing)**: Evaluates unfaithfulness on naturally worded comparative prompt pairs without artificial biasing instructions or edited model outputs.
2. **Qualitative Unfaithfulness Pattern Categorization**: Classifies unfaithful pairs into argument switching, biased fact inconsistency, and answer flipping.
3. **Hard Math Illogical Shortcuts Evaluation**: Measures unfaithful shortcuts on complex mathematical reasoning problems across thinking and non-thinking frontier models.
4. **Standard-Prompt Restoration Error Analysis**: Analyzes restoration errors on GSM8K-style reasoning traces as non-intervention unfaithfulness patterns.

## 2. Target Claims

1. **Claim 1**: Unfaithful chain-of-thought behavior is demonstrated on naturally worded, non-adversarial comparative prompts without artificial biasing instructions or edited model outputs (Figure 2).
2. **Claim 2**: The IPHR evaluation finds frontier-model unfaithfulness rates ranging from near zero up to about 13% of question pairs, depending on model (Table 3).
3. **Claim 3**: Argument switching, biased fact inconsistency, answer flipping, and other patterns occur among IPHR pairs classified as unfaithful (Figure 3).
4. **Claim 4**: Thinking and non-thinking frontier models both exhibit unfaithful illogical shortcuts on hard math problems, though rates vary by model (Figure 5).
5. **Claim 5**: The paper reports standard-prompt restoration errors on GSM8K-style reasoning traces as another non-intervention unfaithfulness pattern (Figure 14).

## 3. Implementation Plan & Verification Strategy

- Implement deterministic modular Python evaluation modules under `submissions/chain-of-thought-reasoning-in-the-wild-is-not-always-faithful/`:
  - `cot_unfaithfulness/iphr_eval.py`: IPHR prompt pair dataset, unfaithfulness detector, and model comparison benchmarking (Table 3, Figure 2).
  - `cot_unfaithfulness/patterns.py`: Pattern classification engine for argument switching, biased fact inconsistency, and answer flipping (Figure 3).
  - `cot_unfaithfulness/hard_math.py`: Evaluation of illogical shortcuts in thinking vs non-thinking models on hard math problems (Figure 5).
  - `cot_unfaithfulness/restoration.py`: Restoration error analysis on GSM8K-style reasoning traces (Figure 14).
  - `generate_evidence.py`: Executes all evaluations deterministically with byte-level verification, outputting `evidence.json`.
  - `tests/test_cot_unfaithfulness.py`: Pytest suite verifying contract correctness, claim assertions, and deterministic reproducibility.
- Create Hugging Face Space application (`app.py`, `README.md`, `pages/*.md`) rendering interactive evaluation results and served logbook pages.
