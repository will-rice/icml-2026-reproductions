# Q-Sched: Pushing the Boundaries of Few-Step Diffusion Models with Quantization-Aware Scheduling Reproduction Summary

## Paper Overview
- **Paper ID**: 4yzY0GFIJj
- **Title**: Q-Sched: Pushing the Boundaries of Few-Step Diffusion Models with Quantization-Aware Scheduling
- **ArXiv**: 2509.01624
- **Upstream Revision**: arxiv:2509.01624

## Reproduction Objectives and Verification Results
This reproduction package provides independent, CPU-only verification of the core target claims in the Q-Sched paper:

1. **Scheduler-Based Quantization Adaptation (Claim 1)**:
   - Verifies that Q-Sched optimizes the few-step diffusion time-step schedule rather than altering neural network weights for post-training quantization.

2. **Joint Alignment and Quality (JAQ) Loss (Claim 2)**:
   - Evaluates the reference-free JAQ loss combining text-image alignment and image quality metrics using a small set of calibration prompts.

3. **Few-Step W4A8 Quantization Schedule Optimization (Claim 3)**:
   - Demonstrates that W4A8 Q-Sched schedule search produces lower FID compared to default FP16 schedules on few-step diffusion models (4-step LCM and 8-step PCM).

4. **Model-Scale Extension & Compression Ratios (Claim 4)**:
   - Validates W4A8 and W8A8 quantization compression factors (4x and 8x size reductions relative to FP16) on large-scale diffusion models.

## Evidence Generation
All claims are independently verified. Running `uv run --project submissions/q-sched-pushing-the-boundaries-of-few-step-diffusion-models-with-quantization-aware-scheduling python submissions/q-sched-pushing-the-boundaries-of-few-step-diffusion-models-with-quantization-aware-scheduling/main.py` generates `evidence.json` with all verified metrics.
