# Motion Attribution for Video Generation Reproduction Design Document

## Target Paper
- Title: Motion Attribution for Video Generation
- Paper ID: zAl9heLw4q
- Slug: motion-attribution-for-video-generation
- Upstream Revision: main (arxiv:2601.08828)

## Target Claims
1. Motive computes motion-specific data attribution by applying motion-weighted loss masks so gradients emphasize dynamic regions rather than static appearance (Section 3.4).
2. The method includes a video-specific frame-length bias fix to reduce spurious attribution to longer clips (Section 3.3).
3. Fine-tuning on Motive-selected data improves VBench motion smoothness and dynamic degree over baselines while using only a fraction of the training data (Table 1).
4. Human evaluation reports a 74.1% preference win rate for Motive-selected fine-tuning compared with the pretrained base model (Table 2).
5. Motive computes motion-specific influence by detecting motion, forming motion-magnitude patches, and applying motion masks to gradient-based data attribution (Figure 1)
6. Motive-selected fine-tuning data improves VBench motion smoothness and dynamic degree compared with random and baseline data-selection methods (Table 1)
7. Human pairwise evaluation reports a 74.1% preference win rate for Motive-selected fine-tuning over the pretrained base model (Table 2)
8. Frame-length normalization prevents attribution rankings from being biased toward longer clips and yields more coherent top-ranked motion samples (Figure 4)
9. Motive's influence scores are not merely selecting high-motion clips; influential clips are those predicted to improve target motion dynamics (Figure 6)

## Verification Strategy
- **Implementation**: Pure CPU deterministic implementation of motion attribution algorithms, frame-length bias normalization, and evaluation metrics.
- **Evidence Pipeline**: CPU verification tests for motion masking, attribution calculation, and dataset selection.
- **Validation**: Independent execution via pytest and generate_evidence.py without external credentials.
- **Deployment**: Gradio application deployed to Hugging Face Space for interactive demonstration.
