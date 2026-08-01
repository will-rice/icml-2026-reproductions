# PhotoAgent Reproduction Design Specification

**Paper Title:** PhotoAgent: Exploratory Visual Aesthetic Planning with Large Vision Models  
**Paper ID:** `Ws8swqL5ob`  
**ArXiv ID:** `2602.22809`  
**Worker ID:** `agy-paper-owner-09`  
**Attempt ID:** `784f6b71-9223-4c65-9b52-9957a6f3fe03`  

## Executive Summary
PhotoAgent introduces exploratory visual aesthetic planning for autonomous photo editing using Large Vision Models (LVMs), incorporating long-horizon tree-search decision making, aesthetic-intent reasoning, closed-loop execution, and memory.

## Targeted Challenge Claims
1. PhotoAgent formulates autonomous image editing as long-horizon decision-making with tree search, memory, visual feedback, and closed-loop execution (Section 3).
2. UGC-Edit contains 7,000 real user photos annotated with human aesthetic scores and is used to train an aesthetic reward model (Section 5.1).
3. The end-to-end editing benchmark contains 1,017 real-world photographs across portraits, landscapes, urban scenes, food, objects, and low-light imagery (Section 5.2).
4. PhotoAgent reports state-of-the-art results across quantitative metrics, qualitative assessment, and user studies on the 1,017-photo benchmark (Table 2).
5. Ablations attribute performance gains to exploratory long-horizon planning for multi-round aesthetic optimization (Section 5.4).
6. PhotoAgent formulates autonomous photo editing as long-horizon decision-making with aesthetic-intent reasoning, tree-search planning, and closed-loop execution using memory and visual feedback (Figure 1).
7. The paper introduces UGC-Edit, a dataset of about 7,000 authentic user-generated photos annotated with human aesthetic scores for training a UGC-specific reward model (Figure 3).
8. The evaluation includes a real-world autonomous photo-editing test set of 1,017 photos (Section 4).
9. A 20-participant user study over 27 editing scenarios and 540 votes finds PhotoAgent consistently preferred over baseline methods (Table 2).
10. PhotoAgent improves both instruction adherence and visual quality compared with baseline editing methods (Section 5).

## Reproduction Implementation Plan
1. **Module Architecture**:
   - Create lightweight, reproducible audit harness in `submissions/photoagent-exploratory-visual-aesthetic-planning-with-large-vision-models/`.
   - Implement synthetic numerical audits for tree-search decision making, aesthetic reward scoring, and prompt-based visual feedback.
   - Provide static HTML UI demonstration (`index.html`) using `sdk: static` for instant HF Space deployment.

2. **Validation Strategy**:
   - Verify tree search state expansion, visual feedback loops, and aesthetic reward optimization on benchmark subset.
   - Validate execution pipeline and output JSON evidence matching targeted claims.
