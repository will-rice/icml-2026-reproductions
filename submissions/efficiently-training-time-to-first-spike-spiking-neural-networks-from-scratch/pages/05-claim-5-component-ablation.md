# Claim 5 — Component ablation

> A Fashion-MNIST ablation improves from 89.61% baseline accuracy to 92.90% when ETTFS-init, average pooling, normalization, affine normalization, and TWD are all enabled (Table 4).

**Status: `partially_reproduced` (toy-scale synthetic 3-class task trained from scratch, not Fashion-MNIST).**

The paper's Table 4 ablation is on Fashion-MNIST. Here the same *structure*
of ablation is run at CPU scale: TTFS networks are trained from scratch with
surrogate gradients on a deterministic synthetic 3-class oriented-bar task
(seed 42, 30 epochs), toggling one component at a time. Accuracies below are
measured on a held-out split of that task.

| Configuration | Test accuracy |
| --- | --- |
| baseline kaiming maxpool nonorm | 91.67% |
| ettfs init only | 91.67% |
| ettfs init avgpool | 98.96% |
| full ettfs init avgpool norm | 100.0% |

- Gain from baseline to full configuration:
  **+8.33 points**
  (91.67% -> 100.0%).
- Most of the gain comes from replacing max pooling with average pooling,
  consistent with the pooling analysis on the previous page.

The *direction* of the paper's ablation reproduces (each component is
non-harmful and the full configuration is best), but these are toy-task
accuracies and are **not** comparable to the paper's 89.61% -> 92.90% on
Fashion-MNIST. The affine-normalization and TWD-in-training variants of the
paper's ablation are not separated at this scale.
