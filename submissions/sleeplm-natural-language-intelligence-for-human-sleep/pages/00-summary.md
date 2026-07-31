# SleepLM Reproduction Summary

Paper: `9wpwfSJCp9`

Attempt: `ca01c0a8-f6cc-4d80-bf3a-c569ba7b4896`

Pinned upstream revision:
`arxiv:2602.23605+github:yang-ai-lab/SleepLM@f788466b926a9ed95d473c220814c912d5ce6abc+hf-model:yang-ai-lab/SleepLM-Base@ec0f94ff2be04fe11ff5a2b37ac38e8f40aa5c53`

## Claim Results

- `0a9ab1e42662e1b1e40ded1370179413b554c5a8663d8f3bd293c56ea6f694f8`: verified from pinned GitHub/HF artifacts for PSG-to-language alignment, shared signal-text embedding, retrieval support, and open-vocabulary sleep understanding.
- `a98161b9f57420109f2d31de27ac2b2d45960406af43b39dc86e0f0b17463d01`: verified as a documented multilevel sleep-caption supervision pipeline with released targeted caption-generation inference support.
- `c0760f7182d3b658fb12eb9409432891beb5e7634a49f10187cea9afcb595666`: inconclusive. The 100K+ hours and 10,000+ individuals statement is present in primary artifacts, but raw NSRR training cohorts are credentialed and not redistributed.
- `142c585a4ac9c87506014c60d24333769cec610fc3b02fa9e464082325b984af`: verified from project-page objective text plus released model config/source showing contrastive alignment, caption generation, and signal reconstruction support.

## Limitations

Dataset-scale evidence is primary-artifact documentation, not a raw-data recount.
No NSRR raw training cohorts were downloaded or inspected. No GPU training,
benchmark reproduction, or model-checkpoint inference was run.
