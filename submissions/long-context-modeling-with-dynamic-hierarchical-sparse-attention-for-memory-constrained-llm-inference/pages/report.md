# Long-Context Modeling with Dynamic Hierarchical Sparse Attention for Memory-Constrained LLM Inference

| Claim | Status | Recomputed observation |
| --- | --- | --- |
| 1 | toy | Pinned model card names the boundary predictor as a lightweight transformer with Shared Encoder, Feature Fusion, and MLP Classifier stages. The model card supports dynamic boundary prediction, but it does not itself prove that a deployed LLM backbone stayed frozen. |
| 2 | toy | Synthetic sparse-routing check at equal 24-token budget: dynamic recall=0.458, block-sparse recall=0.250. This is an independently computed mechanism check, not the paper's Figure 2 attention-recall experiment. |
| 3 | inconclusive | No LongBench model inference was run for Llama-3.1-8B, Mistral-7B, or Qwen2.5-7B. The submission therefore does not reproduce the paper-reported Table 1 accuracy margins. |
| 4 | toy | Monotone density proxy: [0.312711, 0.527633, 0.77687] for densities [0.125, 0.25, 0.5]. The proxy verifies the expected direction of a larger attention budget, but not LongBench accuracy. |
| 5 | inconclusive | No 4-bit Llama-3.1-8B-Instruct latency benchmark or FlashAttention-2 comparison was executed. The local density work proxy is insufficient to claim Figure 6 latency reproduction. |
| 6 | toy | Length-bucket artifact contains 9383848 pretrain and 98557 fine-tune examples, including gt_128k examples. The artifact supports long-context data availability, but no ablation training was run. |

All numeric values above are produced by this submission. Paper-reported benchmark values are not treated as reproduced measurements.
