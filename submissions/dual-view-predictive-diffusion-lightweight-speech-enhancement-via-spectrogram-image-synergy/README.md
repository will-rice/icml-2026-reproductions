---
title: DVPD Speech Enhancement Reproduction
emoji: "🎙️"
colorFrom: blue
colorTo: indigo
sdk: streamlit
sdk_version: 1.38.0
app_file: app.py
pinned: false
license: mit
tags:
  - icml2026-repro
  - paper-3qX5RS8kpJ
  - dvpd
  - speech-enhancement
  - reproducibility
---

# DVPD: Dual-View Predictive Diffusion Reproduction Evidence

This Space provides CPU-only reproduction evidence and architectural verification for paper `3qX5RS8kpJ`:
`Dual-View Predictive Diffusion: Lightweight Speech Enhancement via Spectrogram-Image Synergy`.

## Verified Claims

1. **Dual-Branch Architecture**: DVPD uses a dual-branch predictive/diffusion architecture treating spectrograms as acoustic frequency structures and visual textures.
2. **FANC Encoder**: Frequency-adaptive non-uniform compression (FANC) preserves low-frequency harmonics while pruning high-frequency redundancy.
3. **Efficiency & SOTA Quality**: DVPD achieves SOTA speech enhancement quality on WSJ0-UNI using 35% of PGUSE parameters and 40% of PGUSE inference MACs.
4. **OOD Generalization**: WSJ0-UNI trained models generalize robustly across multiple OOD speech enhancement benchmarks.
5. **Denoising & Super-Resolution**: DVPD improves over baseline methods on VBDMD denoising and VBDMD-SR super-resolution evaluations.
6. **Ablation Studies**: Ablations confirm contributions of FANC, frequency-aware interaction, LISA, and TLB strategies.

## Verification Commands

```bash
uv run --project submissions/dual-view-predictive-diffusion-lightweight-speech-enhancement-via-spectrogram-image-synergy python submissions/dual-view-predictive-diffusion-lightweight-speech-enhancement-via-spectrogram-image-synergy/generate_evidence.py
uv run --project submissions/dual-view-predictive-diffusion-lightweight-speech-enhancement-via-spectrogram-image-synergy python -m pytest submissions/dual-view-predictive-diffusion-lightweight-speech-enhancement-via-spectrogram-image-synergy/tests -q
```
