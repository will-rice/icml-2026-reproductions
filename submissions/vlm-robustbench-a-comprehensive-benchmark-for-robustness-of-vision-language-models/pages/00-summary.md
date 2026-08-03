# VLM-RobustBench Evidence Summary

This Space recomputes three selected ICML 2026 challenge claims for
`HwXyyvK7ZJ` from pinned public artifacts. The evidence script audits
`saxenarohit/vlm_robustbench` at commit
`8bc793d1649e574e000f91c59cb6ce7432c95073` and the project page snapshot with
SHA-256 `6eac82e58ad661354943a49457d81acca3736a127f7c95eeaa2d19744963e7ed`.

The augmentation taxonomy claims are verified by parsing the released
`aug.py` and README: 42 severity-based corruptions, 7 binary transforms, 49
total augmentations, and `42 * 3 + 7 = 133` corrupted settings per
model-dataset pair. The glass-blur severity-mismatch claim is marked `toy`
because the public project artifacts support the qualitative comparison but do
not expose the exact 8.1 percentage-point value as independently recomputed
machine-readable evidence.
