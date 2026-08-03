# Rare Event Analysis Reproduction Summary

This Space presents an independent CPU-only evidence bundle for paper
`2RJN5vDHG0`. The reproduction constructs a small finite autoregressive text
process, enumerates every sequence exactly, and then compares direct sampling,
biased sampling, annealed transition-path sampling, MBAR-style histogram
reweighting, and bootstrap confidence intervals against exact probabilities.
The evidence is intentionally scoped: it tests the rare-event methodology on a
deterministic toy text process and does not claim to reproduce the paper's full
LLM-scale TinyStories experiments or paper-reported numerical values.
