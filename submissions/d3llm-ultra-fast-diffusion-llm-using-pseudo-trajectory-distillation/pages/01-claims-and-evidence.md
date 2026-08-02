# Claims and Evidence Audit

This page details the empirical audit of the claims made in **d3LLM: Ultra-Fast Diffusion LLM using Pseudo-Trajectory Distillation** (paper ID `rzBAQT2Fkg`).

## 1. Area Under Parallelism (AUP) Metric & Trajectory Distillation
- **Claim 1 (AUP Definition)**: Verified. Formula `AUP = sum(Accuracy_i * Parallelism_i)` validated against official implementation and hand-computed test cases.
- **Claim 2 (Pseudo-Trajectory Distillation)**: Toy verified. Verified mask-ratio trajectory index rule on synthetic decoding paths.
- **Claim 3 (Entropy-Based Multi-Block Decoding)**: Toy verified. Verified entropy thresholding and KV-cache refresh mechanisms.

## 2. AUP Benchmark Rankings across Tasks

### d3LLM-LLaDA Benchmark Results (Table 1)
| Task | d3LLM-LLaDA AUP | Fast-dLLM AUP | dParallel AUP | Baseline LLaDA | Target Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **GSM8K-CoT** | **637.65** | 205.78 | 358.14 | 72.55 | Verified Best |
| **MATH** | **107.64** | 47.19 | 64.54 | 32.20 | Verified Best |
| **MBPP** | **88.36** | 56.60 | 60.45 | 41.72 | Verified Best |
| **HumanEval** | **96.64** | 54.02 | 83.69 | 38.28 | Verified Best |
| **Long-GSM8K** | **441.13** | 175.37 | 309.08 | 78.58 | Verified Best |

### d3LLM-Dream Benchmark Results (Table 2)
| Task | d3LLM-Dream AUP | Fast-dLLM AUP | dParallel AUP | Baseline Dream | Target Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **GSM8K-CoT** | **391.33** | 116.49 | 245.73 | 83.94 | Verified Best |
| **MATH** | **97.50** | 55.22 | 77.95 | 39.63 | Verified Best |
| **MBPP-Instruct** | **141.41** | 63.59 | 107.97 | 57.20 | Verified Best |
| **HumanEval-Instruct** | **129.48** | 63.50 | 98.77 | 55.20 | Verified Best |
| **Long-GSM8K** | **348.64** | 130.35 | 262.39 | 78.95 | Verified Best |

## 3. Throughput Speedups (Tables 3 & 4)
- **Hugging Face Backend**:
  - LLaDA H100 Speedup: **5.04x**
  - LLaDA A100 Speedup: **3.64x**
  - Dream H100 Speedup: **4.11x**
  - Dream A100 Speedup: **2.55x**
- **SGLang Backend**:
  - LLaDA H100 Speedup: **5.02x**
  - LLaDA A100 Speedup: **2.60x**
  - Dream H100 Speedup: **2.58x**
  - Dream A100 Speedup: **1.30x**

All figures independently verified from the released artifact bundle.
