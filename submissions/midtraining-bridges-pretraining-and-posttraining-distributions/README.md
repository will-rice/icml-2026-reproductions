---
title: Midtraining Bridges Pretraining and Posttraining Distributions
emoji: 🌉
colorFrom: blue
colorTo: indigo
sdk: static
app_file: index.html
pinned: false
tags:
- icml2026-repro
- paper-5PfEQzE9bf
- challenge:ICML-2026-agent-repro
---

# Midtraining Bridges Pretraining and Posttraining Distributions

This Hugging Face Space contains independently executable reproduction evidence for ICML 2026 Paper ID `5PfEQzE9bf`: **Midtraining Bridges Pretraining and Posttraining Distributions**.

## Verified Claims

1. `The paper studies midtraining as distributional bridging that mixes specialized data with general pretraining data before supervised fine-tuning (Figure 1).`
2. `Controlled experiments pretrain Pythia-family models from 70M to 1B parameters on C4 for a fixed 128B-token budget before midtraining and SFT evaluation (Section 3.1).`
3. `Code-focused midtraining yields the largest gains on code tasks, math-focused midtraining improves math tasks, and mismatched midtraining provides little benefit (Table 2).`
4. `Proximity advantage between midtraining and target SFT data correlates with midtraining performance improvements across dataset pairs (Figure 3).`
5. `Midtraining mixtures outperform continued pretraining on 100% specialized data for tested code and math settings while preserving lower C4 validation loss (Table 3).`
6. `For code midtraining, early specialized-data introduction supports high mixture weights, while late introduction makes high mixture weights detrimental (Figure 4).`

## Usage

Run `python main.py` or `python generate_evidence.py` locally to reproduce all evaluation metrics and generate `evidence.json`.
