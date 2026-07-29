# Reproduction Plan: Q-Sched (Paper ID: 4yzY0GFIJj)

## Overview

- **Paper Title**: Q-Sched: Pushing the Boundaries of Few-Step Diffusion Models with Quantization-Aware Scheduling
- **Paper ID**: `4yzY0GFIJj`
- **Slug**: `q-sched-pushing-the-boundaries-of-few-step-diffusion-models-with-quantization-aware-scheduling`
- **Upstream Revision**: `arxiv:2509.01624v1`
- **Target Claims**:
  1. `Q-Sched modifies the few-step diffusion scheduler rather than the model weights for post-training quantization (Figure 1).`
  2. `The JAQ loss combines text-image compatibility with an image-quality metric and is described as reference-free with only a handful of calibration prompts (Abstract).`

## Architecture & Implementation Plan

1. **Submission Directory Structure**:
   `submissions/q-sched-pushing-the-boundaries-of-few-step-diffusion-models-with-quantization-aware-scheduling`
   - `src/qsched/`:
     - `scheduler.py`: Q-Sched quantization-aware time-step / noise scheduling optimization.
     - `jaq_loss.py`: Joint Alignment & Quality (JAQ) reference-free loss formulation.
     - `eval.py`: Verification suite verifying schedule parameter optimization and JAQ loss properties.
   - `main.py`: Entrypoint producing clean JSON structured execution logs and evidence output.
   - `requirements.txt`: Lightweight CPU dependencies (`torch`, `numpy`, `scipy`).

2. **Verification Strategy**:
   - Verify scheduler time-step optimization vs standard FP16 / quantized schedules.
   - Compute JAQ loss components (text-image similarity proxy and aesthetic quality proxy) using calibration prompts without ground-truth reference images.
   - Output exact reproducible numerical metrics to `evidence.json`.

3. **Safety and Resource Controls**:
   - CPU-only executable path.
   - Metered API cost: $0.00.
