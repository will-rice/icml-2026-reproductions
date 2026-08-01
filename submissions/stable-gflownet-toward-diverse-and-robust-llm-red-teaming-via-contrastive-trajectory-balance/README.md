# Stable-GFlowNet Reproduction

Reproduction repository for ICML 2026 Paper **OyPE1ganBR**: *Stable-GFlowNet: Toward Diverse and Robust LLM Red-Teaming via Contrastive Trajectory Balance*.

## Features

- **Contrastive Trajectory Balance (CTB)**: Replaces explicit partition-function Z estimation with pairwise trajectory comparisons.
- **Noisy Gradient Pruning (NGP)**: Filters out uninformative reward differences under noisy safety classifier evaluations.
- **Min-K Fluency Stabilizer**: Penalizes out-of-distribution / gibberish prompts to prevent reward hacking.
- **Red-Teaming Benchmark & Ablations**: Evaluates attack diversity, attack success rate, and component contributions.

## Execution

Run unit tests:
```bash
PYTHONPATH=src python -m pytest tests/ -q
```

Generate evidence bundle:
```bash
python generate_evidence.py --output evidence/bundle.json
```
