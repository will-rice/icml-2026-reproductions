---
title: "Mind Omni Reproduction"
colorFrom: "green"
colorTo: "blue"
sdk: "static"
pinned: false
tags:
  - icml2026-repro
  - paper-3gCdh3u2GK
---

# Mind-Omni Reproduction

CPU-only evidence audit for `3gCdh3u2GK`, using the pinned official source commit `818dcd160c130334bd36a7a8a7f7e7f00772084d`. The bundle separates source-supported architecture and artifact-release checks from unavailable state-of-the-art and synergy claims that require raw evaluation outputs.

Run:

```bash
uv run --project submissions/mind-omni-a-unified-multi-task-framework-for-brain-vision-language-modeling-via-discrete-diffusion python submissions/mind-omni-a-unified-multi-task-framework-for-brain-vision-language-modeling-via-discrete-diffusion/generate_evidence.py --output submissions/mind-omni-a-unified-multi-task-framework-for-brain-vision-language-modeling-via-discrete-diffusion/evidence/bundle.json
uv run --project submissions/mind-omni-a-unified-multi-task-framework-for-brain-vision-language-modeling-via-discrete-diffusion python -m pytest submissions/mind-omni-a-unified-multi-task-framework-for-brain-vision-language-modeling-via-discrete-diffusion/tests -q
```
