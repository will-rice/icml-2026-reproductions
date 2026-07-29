# QuArch: A Benchmark for Evaluating LLM Reasoning in Computer Architecture

This repository contains the reproduction suite and verified evaluation evidence for the paper *QuArch: A Benchmark for Evaluating LLM Reasoning in Computer Architecture* (arXiv:2510.22087).

## Target Claims

1. **Claim 1**: QuArch contains 2,671 expert-validated computer-architecture question-answer pairs built from synthetic generation, crowdsourcing, and academic exams.
2. **Claim 2**: The benchmark evaluates four skills: Recall, Analyze, Design, and Implement, with relevant context and figures when appropriate.

## Reproducing Results

Run the evidence generation script:
```bash
python generate_evidence.py
```

Run tests:
```bash
pytest tests/
```
