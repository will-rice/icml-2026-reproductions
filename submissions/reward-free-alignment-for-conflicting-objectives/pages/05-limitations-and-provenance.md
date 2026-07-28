# Limitations and Provenance

## Scope Boundaries & Limitations

1. **Deterministic CPU Audits:** This reproduction focuses on the exact mathematical, algorithmic, and theoretical foundations of RACO (objective-specific pairwise losses, weighted CAGrad-Clip dual optimization, and Theorems 3.1 & 3.2). All outcomes are derived from audit computations, never hard-coded.
2. **Empirical Benchmarks:** Full LLM fine-tuning on multi-billion parameter models (Qwen 3, Llama 3, Gemma 3) on summarization (TL;DR) and safety datasets (BeaverTails) was not re-executed due to hardware constraints. Corresponding empirical claims (Claims 3, 4, 5, 10) are marked `limited` locally. No paper-reported empirical values are entered as reproduced measurements.
3. **No LLM Inference:** All tests and evidence generation runs strictly offline on CPU without LLM training or API calls.
4. **Scale-Invariant Solver:** The solver uses scale-relative thresholds ($\max(|A|, |B|, |C|)$ for polynomial degeneracy, relative stationarity verification) so it finds correct interior solutions at all gradient magnitudes from $10^{-8}$ to $10^{8}$.
5. **Provenance Hardening:** Fully fail-closed: duplicate JSON keys rejected at every JSON load, extra manifest keys rejected, duplicate artifact IDs/paths rejected, empty artifact entries rejected, Git blob IDs recomputed and verified. Schema validation is mandatory. Evidence generation invokes `load_verified_artifacts` during generation.
6. **Claim 6 Consistency:** Uses a 3-parameter model to ensure non-colinear gradients. Audit persisted under `audits.claim6_pipeline`. Page and summary outcome derived from the exact audit.
7. **Theorem 3.2 Precondition Enforcement:** Every precondition (finite simplex weights, positive finite step size, admissible $c < 1$, finite gradients, interior coefficients, positive $\Gamma$ improvement) is checked before declaring support.

## Provenance & Pinned Identity

- **Paper ID:** `vSzRJyg6k0`
- **Attempt ID:** `97e213a5-7ca3-4a1b-a500-1ec52d94d87a`
- **Admitted Snapshot ID:** `09017559ff2c5746f1a37458ba9a330bd4e18654ae9c3f873bb0785c76626199`
- **ArXiv Pin:** `arxiv:2602.02495v3`
- **GitHub Repository Pin:** `github:PeterLauLukChen/RACO@84a943c34f38520c7e0c9dd3066517c111b3c8fa`
- **API Cost:** USD 0.00

## Verified Upstream Artifact Lineage

The following 4 packaged upstream files are verified at build time by `load_verified_artifacts()` and persisted in canonical evidence under `artifacts`:

| Artifact ID | Relative Path | SHA-256 | Git Blob | Size (Bytes) | Source URL | Acquisition Command | License |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `LICENSE` | `evidence/inputs/upstream/LICENSE` | `1d90ecbf7bb80ff27ef67b7785f621480b652b82e46c72d8a5730b0db65bf355` | `a98c551ed4f7ba78782da3c4f4c47ca6443592b9` | 11,355 | [LICENSE](https://raw.githubusercontent.com/PeterLauLukChen/RACO/84a943c34f38520c7e0c9dd3066517c111b3c8fa/LICENSE) | `git clone https://github.com/PeterLauLukChen/RACO.git && cd RACO && git checkout 84a943c34f38520c7e0c9dd3066517c111b3c8fa` | Apache-2.0 |
| `README.md` | `evidence/inputs/upstream/README.md` | `57c695233e2229194db9e0513c7523e306a1e4d6cd4ca85e56bd8e40f6fb7feb` | `f4f1f923e87a95c091e68e3f6f19ba53f4a0ea5d` | 10,047 | [README.md](https://raw.githubusercontent.com/PeterLauLukChen/RACO/84a943c34f38520c7e0c9dd3066517c111b3c8fa/README.md) | `git clone https://github.com/PeterLauLukChen/RACO.git && cd RACO && git checkout 84a943c34f38520c7e0c9dd3066517c111b3c8fa` | Apache-2.0 |
| `m=3-RACO-CAGrad-Algo.md` | `evidence/inputs/upstream/m=3-RACO-CAGrad-Algo.md` | `c8c8f210eb2d53f922cf1aacde68f4840355fa682a3055202c26a9590d20211b` | `a54009de54971f1893d2d47c19a38bb2ad19fc2b` | 10,923 | [m=3-RACO-CAGrad-Algo.md](https://raw.githubusercontent.com/PeterLauLukChen/RACO/84a943c34f38520c7e0c9dd3066517c111b3c8fa/m%3D3-RACO-CAGrad-Algo.md) | `git clone https://github.com/PeterLauLukChen/RACO.git && cd RACO && git checkout 84a943c34f38520c7e0c9dd3066517c111b3c8fa` | Apache-2.0 |
| `train_raco.py` | `evidence/inputs/upstream/train_raco.py` | `f47041b468784161bfebabfd6732478a0d0b7be5b57eb8a2321eea2de21469f4` | `7f78c09b840d614d86032062350d43df2e7463bf` | 19,536 | [train_raco.py](https://raw.githubusercontent.com/PeterLauLukChen/RACO/84a943c34f38520c7e0c9dd3066517c111b3c8fa/train_raco.py) | `git clone https://github.com/PeterLauLukChen/RACO.git && cd RACO && git checkout 84a943c34f38520c7e0c9dd3066517c111b3c8fa` | Apache-2.0 |

> **Notice:** Local outcomes (`supported`, `limited`, `not-supported`) are not an official verdict from challenge controllers or program chairs.
