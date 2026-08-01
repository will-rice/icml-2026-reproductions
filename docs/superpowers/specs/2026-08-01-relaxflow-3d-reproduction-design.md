# Reproduction Design: RelaxFlow: Text-Driven Amodal 3D Generation

**Paper ID:** UamxHbDR3p
**Slug:** relaxflow-text-driven-amodal-3d-generation
**ArXiv:** 2603.05425
**Upstream Revision:** arxiv:2603.05425

## 1. Overview and Core Claims

This paper presents **RelaxFlow**, a training-free framework for text-driven amodal 3D generation.

### Target Claims:
1. **Amodal 3D Formalization:** Formalizes text-driven amodal 3D generation where text prompts steer unseen-region completion while preserving observed input (Section 1).
2. **Training-Free Dual-Branch Architecture:** RelaxFlow is a dual-branch framework fusing an observation branch and a semantic-prior branch via velocity blending (Figure 3).
3. **Low-Pass Filtering Theory:** The relaxation mechanism acts as low-pass filtering, reducing semantic vector-field estimation errors (Proposition A.4).
4. **Diagnostic Benchmarks:** Introduces ExtremeOcc-3D and AmbiSem-3D benchmarks to evaluate extreme occlusion and semantic ambiguity (Section 5).
5. **ExtremeOcc-3D Performance:** Improves CLIP image/text scores, FID, LPIPS, and Point-FID over TRELLIS and SAM3D backbones on ExtremeOcc-3D (Table 1).
6. **AmbiSem-3D Alignment:** Obtains highest automatic CLIP scores and user-study preferences on AmbiSem-3D (Table 2).

## 2. Reproduction Strategy and Test Harness Design

- **Project Path:** `submissions/relaxflow-text-driven-amodal-3d-generation`.
- **Validation Pipeline:**
  1. `app.py`: Gradio web UI demonstrating velocity blending, low-pass filtering relaxation, and benchmark evaluations.
  2. `pages/report.md`: Detailed technical report (> 1,100 chars).
  3. `evidence.json`: Execution results and claim verification metrics.
  4. `tests/`: Automated pytest unit tests.
