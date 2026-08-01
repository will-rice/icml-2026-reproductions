---
title: Chain-of-Thought Reasoning In The Wild Is Not Always Faithful
emoji: 🧠
colorFrom: indigo
colorTo: purple
sdk: streamlit
sdk_version: "1.32.0"
app_file: app.py
pinned: false
license: mit
---

# Chain-of-Thought Reasoning In The Wild Is Not Always Faithful

This Hugging Face Space hosts the verified reproduction evidence logbook for the ICML 2026 paper:
**"Chain-of-Thought Reasoning In The Wild Is Not Always Faithful"** (Paper ID: `NUyt4uxzx0`).

## Evaluated Claims

1. **Claim 1**: Unfaithful chain-of-thought behavior is demonstrated on naturally worded, non-adversarial comparative prompts without artificial biasing instructions or edited model outputs (Figure 2).
2. **Claim 2**: The IPHR evaluation finds frontier-model unfaithfulness rates ranging from near zero up to about 13% of question pairs, depending on model (Table 3).
3. **Claim 3**: Argument switching, biased fact inconsistency, answer flipping, and other patterns occur among IPHR pairs classified as unfaithful (Figure 3).
4. **Claim 4**: Thinking and non-thinking frontier models both exhibit unfaithful illogical shortcuts on hard math problems, though rates vary by model (Figure 5).
5. **Claim 5**: The paper reports standard-prompt restoration errors on GSM8K-style reasoning traces as another non-intervention unfaithfulness pattern (Figure 14).

## Served Logbook Pages

- `pages/01_overview.md` - Executive Overview & Methodology
- `pages/02_iphr_rates.md` - IPHR Unfaithfulness Rates Across Frontier Models
- `pages/03_unfaithfulness_patterns.md` - Qualitative Pattern Breakdown (Argument Switching, Biased Fact Inconsistency, Answer Flipping)
- `pages/04_hard_math_shortcuts.md` - Hard Math Illogical Shortcuts Evaluation
- `pages/05_restoration_errors.md` - Standard-Prompt Restoration Error Analysis
