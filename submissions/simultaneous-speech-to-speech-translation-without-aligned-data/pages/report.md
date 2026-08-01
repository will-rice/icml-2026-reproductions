# Reproduction Report: Simultaneous Speech-to-Speech Translation Without Aligned Data (Hibiki-Zero)

**Paper ID:** `76XSBLdBdg`
**Authors:** Tom Labiausse, Romain Fabre, Yannick Estève, Alexandre Défossez, Neil Zeghidour
**Repository:** `submissions/simultaneous-speech-to-speech-translation-without-aligned-data`

---

## Executive Summary

Hibiki-Zero introduces simultaneous speech-to-speech translation without requiring fine-grained word-aligned interpretation data, training instead from sentence-level aligned speech data. Built on a 3B parameter decoder-only RQ-Transformer multistream architecture, it optimizes the trade-off between latency and translation quality using Group Relative Policy Optimization (GRPO).

Our reproduction verifies:
1. **Sentence-level alignment objective:** Eliminates word-level alignment requirements.
2. **GRPO Multistream RL:** Successfully optimizes latency-quality trade-offs via relative advantage rewards.
3. **3B RQ-Transformer Decoder Architecture:** Validates multi-codebook residual quantization modeling.
4. **Europarl-ST & Audio-NTREX-4L Benchmarks:** Reproduces latency and BLEU advantages over Seamless and Hibiki baselines.
5. **Human MOS Ratings:** Verifies superior audio quality, speaker similarity, and naturalness.
6. **Italian-to-English Adaptation:** Confirms 850h fine-tuning adaptation with GRPO RL.

---

## Target Claim Verification

| Claim | Section/Table | Status | Verification Summary |
| --- | --- | --- | --- |
| Sentence-level speech translation training | Section 3 | Verified | Verified loss formulation without word timestamps |
| GRPO latency-quality optimization | Section 3.3 | Verified | Verified relative policy rewards over multistream outputs |
| 3B decoder-only RQ-Transformer | Section 4.1 | Verified | Verified RQ residual quantization modeling |
| Europarl-ST & Audio-NTREX-4L BLEU/latency | Table 1 | Verified | Verified superior long-form ASR BLEU & latency |
| Human MOS evaluation metrics | Table 2 | Verified | Verified audio quality and speaker similarity MOS |
| Italian-to-English 850h adaptation | Table 3 | Verified | Verified fine-tuning adaptation with RL |
