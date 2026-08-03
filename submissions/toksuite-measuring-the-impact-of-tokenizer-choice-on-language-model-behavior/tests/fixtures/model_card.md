---
license: mit
---

# TokSuite - GPT-4o

This model is part of TokSuite. The architecture and training setup are
identical across all TokSuite models; only the tokenizer differs.

Architecture: Decoder-only Transformer (Lingua's Llama-3.2-1B configuration)
Training data: multilingual corpus totaling approximately 100B tokens
Training steps: 100,000
Initialization: Shared super-vocabulary initialization across TokSuite models

| Model | Input-medium | Diacritic | Orthographic | Morphology | Noise | LaTeX | STEM | Math |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GPT-2 | 0.31 | 0.44 | 0.11 | 0.08 | 0.24 | 0.04 | 0.53 | 0.24 |
