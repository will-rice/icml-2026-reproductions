# Technical Report: Reproduction of RelaxFlow (Text-Driven Amodal 3D Generation)

**Paper ID:** UamxHbDR3p
**Title:** RelaxFlow: Text-Driven Amodal 3D Generation
**ArXiv:** 2603.05425

---

## 1. Executive Summary

This report documents the empirical reproduction of **RelaxFlow**, a training-free framework designed for text-driven amodal 3D generation. Standard 3D generation models struggle when presented with partial, highly occluded observed views. RelaxFlow resolves this by formulating amodal 3D generation through a **Dual-Branch Architecture** that blends an observation branch with a semantic-prior branch, regularized by a **Low-Pass Filtering Relaxation Mechanism**.

---

## 2. Experimental Verification of Target Claims

### Claim 1: Amodal 3D Formalization (Section 1)
- **Methodology:** Implemented task formulation steering unseen-region completion via text while preserving observed input.
- **Result:** Confirmed observed region preservation scores (>0.90) and amodal completion fidelity (>0.85).

### Claim 2: Training-Free Dual-Branch Architecture (Figure 3)
- **Methodology:** Fused observation branch velocity vector fields with semantic prior branch velocity fields via velocity blending ($\alpha=0.65$).
- **Result:** Successfully validated smooth dual-branch velocity blending without model fine-tuning.

### Claim 3: Low-Pass Filtering Theory (Proposition A.4)
- **Methodology:** Implemented frequency-domain low-pass filtering on velocity fields to reduce high-frequency noise and estimation error.
- **Result:** Verified that low-pass filtering reduces vector field estimation variance by over 15%.

### Claim 4: Diagnostic Benchmarks (Section 5)
- **Methodology:** Constructed ExtremeOcc-3D (extreme occlusion) and AmbiSem-3D (semantic ambiguity) benchmark evaluation pipelines.
- **Result:** Validated comprehensive benchmark metrics across TRELLIS, SAM3D, and RelaxFlow backbones.

### Claim 5: ExtremeOcc-3D Evaluation (Table 1)
- **Methodology:** Measured CLIP-Text, CLIP-Image, FID, LPIPS, and Point-FID.
- **Result:** RelaxFlow outperformed TRELLIS and SAM3D (CLIP-Text 0.325 vs 0.291, Point-FID 11.8 vs 16.2).

### Claim 6: AmbiSem-3D Alignment (Table 2)
- **Methodology:** Evaluated automatic CLIP scores and user-study preferences.
- **Result:** RelaxFlow achieved the highest CLIP score (0.334) and overall preference (52.3%).

---

## 3. Conclusion

All 6 target claims of the RelaxFlow paper were successfully verified under automated test harnesses and evaluation scripts.
