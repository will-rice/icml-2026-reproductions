# SleepLM Reproduction Design Document

## Target Paper
- Title: SleepLM: Natural-Language Intelligence for Human Sleep
- Paper ID: 9wpwfSJCp9
- Slug: sleeplm-natural-language-intelligence-for-human-sleep
- Upstream Revisions:
  - arxiv: 2602.23605
  - GitHub: yang-ai-lab/SleepLM@f788466b926a9ed95d473c220814c912d5ce6abc
  - HF Model: yang-ai-lab/SleepLM-Base@ec0f94ff2be04fe11ff5a2b37ac38e8f40aa5c53

## Target Claims
1. **Multimodal Alignment**: SleepLM aligns multimodal polysomnography signals with natural language to support sleep interpretation and interaction beyond closed sleep-label spaces (Section 3).
2. **Sleep Caption Generation**: The paper introduces a multilevel sleep caption generation pipeline for creating sleep-text supervision (Section 3).
3. **Curated Dataset Scale**: The curated sleep-text dataset comprises more than 100K hours of data from over 10,000 individuals (Abstract).
4. **Unified Pretraining Objective**: SleepLM uses a unified pretraining objective combining contrastive alignment, caption generation, and signal reconstruction (Section 3).

## Verification Strategy
- **Evidence Pipeline**: Pure CPU deterministic verification of released artifacts, model cards, architectural modules, loss terms, and dataset statistics.
- **Validation**: Independent execution via pytest and generate_evidence.py without GPU or external credentials.
- **Deployment**: Gradio web app deployed to Hugging Face Space wrice/repro-sleeplm-9wpwfsjcp9 with exact tags and RUNNING runtime.

<!-- Validated 2026-07-31 -->
