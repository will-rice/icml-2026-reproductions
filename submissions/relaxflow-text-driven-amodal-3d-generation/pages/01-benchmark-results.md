# Detailed Benchmark Results: RelaxFlow

**Paper ID:** UamxHbDR3p  
**Title:** RelaxFlow: Text-Driven Amodal 3D Generation  

---

## 1. Table 1: ExtremeOcc-3D Benchmark Evaluation

The ExtremeOcc-3D benchmark measures 3D generation quality and view consistency under extreme occlusion.

| Model | CLIP-Text ($\uparrow$) | CLIP-Image ($\uparrow$) | FID ($\downarrow$) | LPIPS ($\downarrow$) | Point-FID ($\downarrow$) |
|---|---|---|---|---|---|
| TRELLIS | 0.2840 | 0.7210 | 34.20 | 0.2450 | 18.50 |
| SAM3D | 0.2910 | 0.7350 | 31.80 | 0.2280 | 16.20 |
| **RelaxFlow (Ours)** | **0.3250** | **0.7890** | **24.60** | **0.1740** | **11.80** |

---

## 2. Table 2: AmbiSem-3D Benchmark & Preference Evaluation

The AmbiSem-3D benchmark evaluates alignment and visual fidelity under semantic ambiguity.

| Model | CLIP Score ($\uparrow$) | User Alignment % ($\uparrow$) | 3D Fidelity % ($\uparrow$) | Overall Preference % ($\uparrow$) |
|---|---|---|---|---|
| TRELLIS | 0.2780 | 22.4% | 24.1% | 21.8% |
| SAM3D | 0.2860 | 26.5% | 27.2% | 25.9% |
| **RelaxFlow (Ours)** | **0.3340** | **51.1%** | **48.7%** | **52.3%** |

---

## 3. Theoretical & Architectural Ablations

- **Velocity Blending Weight ($\alpha$):** 0.65 provides optimal balance between observed constraint enforcement and semantic completion.
- **Low-Pass Filter Cutoff:** 0.25 frequency cutoff reduces high-frequency estimation noise by 39.46% (Proposition A.4).
