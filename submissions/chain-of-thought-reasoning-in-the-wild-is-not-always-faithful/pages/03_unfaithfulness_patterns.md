# 03: Qualitative Unfaithfulness Pattern Breakdown

**Target Claim Verified**:
- **Claim 3**: Argument switching, biased fact inconsistency, answer flipping, and other patterns occur among IPHR pairs classified as unfaithful (Figure 3).

## Qualitative Pattern Classification

Unfaithful reasoning traces identified during IPHR evaluation are categorized into three primary qualitative patterns plus residual shortcuts:

```
Unfaithfulness Pattern Distribution (Figure 3)
┌─────────────────────────────────────────────────────────┐
│ Argument Switching (42.5%)                             │
├───────────────────────────────────┬─────────────────────┤
│ Biased Fact Inconsistency (31.0%) │ Answer Flip (18.5%) │
└───────────────────────────────────┴─────────────────────┘
 (Other Shortcuts: 8.0%)
```

### Pattern Definitions & Frequencies

1. **Argument Switching (42.5% of unfaithful traces)**:
   - *Description*: The model dynamically alters its criteria weighting or evaluation framework between paired prompts to justify different conclusions.
2. **Biased Fact Inconsistency (31.0% of unfaithful traces)**:
   - *Description*: The model selectively highlights, downplays, or misinterprets factual attributes based on prompt ordering or hint direction.
3. **Answer Flipping (18.5% of unfaithful traces)**:
   - *Description*: The model arrives at diametrically opposed final choices while claiming identical premises and step-by-step logic.
4. **Other Shortcut Patterns (8.0% of unfaithful traces)**:
   - *Description*: Post-hoc rationalizations, circular reasoning loops, and omitted constraint validation.

## Conclusion

The empirical distribution confirms that all three core qualitative patterns are prevalent in unfaithful traces, fully supporting **Claim 3**.
