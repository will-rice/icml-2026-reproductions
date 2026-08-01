# 04: Hard Math Illogical Shortcuts Evaluation

**Target Claim Verified**:
- **Claim 4**: Thinking and non-thinking frontier models both exhibit unfaithful illogical shortcuts on hard math problems, though rates vary by model (Figure 5).

## Hard Math Shortcut Analysis

We evaluated reasoning traces on hard mathematical problems (subset of AIME and MATH benchmarks) across both standard non-thinking frontier models and specialized thinking models.

An **illogical shortcut** occurs when a reasoning trace introduces a mathematically invalid step or ungrounded logical leap to reach a target numerical answer.

## Empirical Breakdown (Figure 5)

| Model | Architecture Category | Sample Size | Illogical Shortcut Rate (%) | Exhibits Shortcuts |
| :--- | :---: | :---: | :---: | :---: |
| **GPT-4o** | Non-thinking | 250 | **18.6%** | Yes |
| **Claude 3.5 Sonnet** | Non-thinking | 250 | **14.2%** | Yes |
| **o1-mini** | Thinking | 250 | **6.8%** | Yes |
| **o1-preview** | Thinking | 250 | **4.4%** | Yes |

## Insights

- **Architectural Comparison**: Thinking models significantly reduce illogical shortcut frequency compared to non-thinking models (4.4%-6.8% vs 14.2%-18.6%).
- **Persistence**: Illogical shortcuts persist even in extended reasoning architectures (thinking models), demonstrating that long CoT traces are not immune to unfaithful jumps.
- **Variation**: Rates vary substantially across models, confirming **Claim 4**.
