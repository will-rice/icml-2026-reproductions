---
dataset_info:
- config_name: qualitative
  splits:
  - name: test
    num_examples: 880
- config_name: throughput_1k
  splits:
  - name: test
    num_examples: 1536
- config_name: throughput_2k
  splits:
  - name: test
    num_examples: 1536
- config_name: throughput_8k
  splits:
  - name: test
    num_examples: 1536
- config_name: throughput_16k
  splits:
  - name: test
    num_examples: 1536
- config_name: throughput_32k
  splits:
  - name: test
    num_examples: 1536
---

This dataset combines a qualitative split and a throughput split.

| Category | SpecBench | SPEED (random selection) | SPEED (greedy algorithm) |
|---|---:|---:|---:|
| Math | 0.24 | 0.21 (-12.5%) | 0.15 (-37.5%) |
| Coding | 0.33 | 0.48 (+45%) | 0.16 (-51%) |
| RAG | 0.15 | 0.17 (+13%) | 0.13 (-13%) |
| QA | 0.10 | unchanged samples | unchanged samples |

| Metric | **SPEED-Bench** | **SpecBench** |
| :--- | :--- | :--- |
| Avg. Pairwise Similarity | 0.14 | 0.22 |
