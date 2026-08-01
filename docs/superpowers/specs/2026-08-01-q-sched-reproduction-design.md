# Reproduction Design: Q-Sched (Quantization-Aware Scheduling for Few-Step Diffusion Models)

**Paper ID:** 4yzY0GFIJj  
**Slug:** q-sched-pushing-the-boundaries-of-few-step-diffusion-models-with-quantization-aware-scheduling  
**ArXiv:** 2509.01624  
**Upstream Revision:** arxiv:2509.01624  

## 1. Overview and Core Claims

Q-Sched optimizes the diffusion sampling schedule rather than fine-tuning or modifying quantized model weights for post-training quantization (PTQ). By optimizing step boundaries according to a Joint Alignment and Quality (JAQ) loss on a small set of calibration prompts, Q-Sched improves generation quality under aggressive quantization (e.g. W4A8 and W8A8).

### Target Claims:
1. **Scheduler Optimization:** Q-Sched modifies the few-step diffusion scheduler rather than the model weights for post-training quantization (Figure 1).
2. **JAQ Loss Formulation:** The JAQ loss combines text-image compatibility with an image-quality metric and is described as reference-free with only a handful of calibration prompts (Abstract).
3. **SD v1-5 Performance:** On Stable Diffusion v1-5 few-step models, W4A8 Q-Sched reports lower FID than the listed FP16 original schedules for 4-step LCM and 8-step PCM settings (Table 1).
4. **SDXL Compression Scale:** For SDXL-scale models, Q-Sched is evaluated under W4A8 and W8A8 compression, described as 4x and 8x model-size reductions compared with FP16 (Table 2).
5. **FID Improvements:** The paper reports 15.5% FID improvement over FP16 4-step LCM and 16.6% improvement over FP16 8-step PCM (Abstract).

## 2. Reproduction Strategy and Test Harness Design

- **Code Structure:** The project is contained within `submissions/q-sched-pushing-the-boundaries-of-few-step-diffusion-models-with-quantization-aware-scheduling`.
- **Validation Pipeline:**
  1. `app.py`: Provides Gradio / FastAPI UI demonstrating Q-Sched scheduler optimization and comparison against default schedules.
  2. `evidence.json`: Records verified claim outputs, numerical metrics, prompt evaluations, and execution provenance.
  3. `tests/`: Automated unit and integration tests verifying scheduler step adjustment, JAQ loss computation, and metric generation.

## 3. Provenance and Environment
- Upstream paper: arXiv:2509.01624
- Target execution environment: CPU-only / lightweight workstation evaluation with deterministic synthetic/lightweight model harnesses.
