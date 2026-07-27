# FlashBlock: Attention Caching for Efficient Long-Context Block Diffusion - Technical Reproduction Design

## Paper Overview & Context
- **Title**: FlashBlock: Attention Caching for Efficient Long-Context Block Diffusion
- **Authors**: Zhuokun Chen, Jianfei Cai, Bohan Zhuang
- **Venue**: ICML 2026
- **Paper ID**: `4jfuNNghPS`
- **ArXiv**: `2602.05305v1`
- **Repository Slug**: `flashblock`
- **Target Submission Path**: `submissions/flashblock`
- **Target HF Space**: `wrice/repro-flashblock-4jfunnghps`

---

## Technical Concept & Mathematical Formulation

### 1. Block Diffusion & Attention Decomposition
In block diffusion models, sequence generation is performed block-by-block. At step $s$ of generating block $k$, attention is evaluated for queries $Q_{\text{block}} \in \mathbb{R}^{B \times d}$ (where $B$ is block size) against key-value pairs from:
1. Block-external tokens $\mathcal{J}_{\text{out}}$: Previously generated tokens in blocks $0, \dots, k-1$.
2. Block-internal tokens $\mathcal{J}_{\text{in}}$: Tokens inside the current block $k$.

For a query $q_i$, the scaled dot-product attention output and normalizer are:
$$S_{ij} = \frac{q_i k_j^\top}{\sqrt{d}}$$
$$L_i = \log \sum_{j} \exp(S_{ij})$$
$$A_i = \frac{\sum_{j} \exp(S_{ij}) v_j}{\exp(L_i)}$$

Decomposing into block-external $\mathcal{J}_{\text{out}}$ and block-internal $\mathcal{J}_{\text{in}}$:
$$L_{\text{out}, i} = \log \sum_{j \in \mathcal{J}_{\text{out}}} \exp(S_{ij}), \quad A_{\text{out}, i} = \frac{\sum_{j \in \mathcal{J}_{\text{out}}} \exp(S_{ij}) v_j}{\exp(L_{\text{out}, i})}$$
$$L_{\text{in}, i} = \log \sum_{j \in \mathcal{J}_{\text{in}}} \exp(S_{ij}), \quad A_{\text{in}, i} = \frac{\sum_{j \in \mathcal{J}_{\text{in}}} \exp(S_{ij}) v_j}{\exp(L_{\text{in}, i})}$$

### 2. Cross-Step Stability & Selective Caching
- **Observation**: Key/value states and attention weights for $\mathcal{J}_{\text{out}}$ remain stable across diffusion steps $s \to s+1$ within the block, whereas $\mathcal{J}_{\text{in}}$ changes rapidly as block tokens are denoised/unmasked.
- **Selective Reuse**:
  - At step $s=1$ (or when updated token count $M^s \ge \tau$): Compute full attention. Cache $(A_{\text{out}}^s, L_{\text{out}}^s)$.
  - At step $s+1$ (when $M^{s+1} < \tau$): Reuse cached $(A_{\text{out}}^s, L_{\text{out}}^s)$ without querying the growing KV cache for $\mathcal{J}_{\text{out}}$. Recompute only $(A_{\text{in}}^{s+1}, L_{\text{in}}^{s+1})$ over $\mathcal{J}_{\text{in}}$.

### 3. Log-Space Attention Composition
To combine cached $(A_{\text{out}}^s, L_{\text{out}}^s)$ and recomputed $(A_{\text{in}}^{s+1}, L_{\text{in}}^{s+1})$ with complete numerical stability:
$$m_i = \max(L_{\text{out}, i}^s, L_{\text{in}, i}^{s+1})$$
$$L_{\text{full}, i}^{s+1} = m_i + \log\left(\exp(L_{\text{out}, i}^s - m_i) + \exp(L_{\text{in}, i}^{s+1} - m_i)\right)$$
$$w_{\text{out}, i} = \frac{\exp(L_{\text{out}, i}^s - m_i)}{\exp(L_{\text{full}, i}^{s+1} - m_i)}, \quad w_{\text{in}, i} = \frac{\exp(L_{\text{in}, i}^{s+1} - m_i)}{\exp(L_{\text{full}, i}^{s+1} - m_i)}$$
$$A_{\text{full}, i}^{s+1} = w_{\text{out}, i} A_{\text{out}, i}^s + w_{\text{in}, i} A_{\text{in}, i}^{s+1}$$

---

## Target Reproduction Claims

1. **`cross-step-attention-stability-discrepancy`**:
   - *Claim*: Block-external attention outputs exhibit high cross-step stability (cosine similarity $\ge 0.95$, L1 difference $\le 0.05$) across adjacent diffusion steps within a block, whereas block-internal attention exhibits substantially lower stability ($\le 0.70$).
   - *Independent Test*: Simulate multi-step block diffusion over sequences up to length $N=1024$ and block size $B=16$; measure layer-wise cosine similarity and L1 distance between steps $s$ and $s+1$ for $A_{\text{out}}$ vs $A_{\text{in}}$.

2. **`block-external-attention-caching-speedup`**:
   - *Claim*: Reusing cached block-external attention $(A_{\text{out}}, L_{\text{out}})$ reduces attention computation FLOPs and KV-cache accesses per step from $O(B N)$ to $O(B^2)$, achieving $> 1.30\times$ attention throughput speedup for block size $B \in [4, 8]$ and update threshold $\tau \ge 2$ while maintaining negligible error ($L_1 < 10^{-4}$).
   - *Independent Test*: Benchmark attention compute latency, memory throughput, and FLOP count across context lengths $N \in [256, 4096]$ comparing standard block diffusion vs FlashBlock caching.

3. **`log-space-attention-composition-fidelity`**:
   - *Claim*: FlashBlock's log-space composition of cached $(A_{\text{out}}, L_{\text{out}})$ and recomputed $(A_{\text{in}}, L_{\text{in}})$ achieves exact numerical equivalence ($L_{\infty} < 10^{-5}$, max relative error $< 10^{-5}$) compared to full dense attention computation.
   - *Independent Test*: Perform synthetic and model-based attention forward passes comparing single-pass full attention against decomposed log-space composed attention across diverse scale factors and hidden dimensions $d \in [64, 128, 512]$.

---

## Architecture & Code Structure

```
submissions/flashblock/
├── pyproject.toml
├── README.md
├── app.py
├── generate_evidence.py
├── src/
│   └── flashblock_repro/
│       ├── __init__.py
│       ├── attention.py
│       ├── block_diffusion.py
│       └── metrics.py
└── tests/
    ├── __init__.py
    ├── test_attention.py
    ├── test_block_diffusion.py
    └── test_metrics.py
```

### Module Responsibilities:
- `src/flashblock_repro/attention.py`: Scaled dot-product attention, `BlockCausalAttentionCache`, log-space composition operator, and `FlashBlockAttention` layer.
- `src/flashblock_repro/block_diffusion.py`: `BlockDiffusionModel` implementation with block KV caching and FlashBlock attention caching.
- `src/flashblock_repro/metrics.py`: Metrics for measuring cosine similarity, L1 distance, attention FLOPs, and latency speedups.
- `generate_evidence.py`: Standard evidence generator script outputting `evidence_summary.json` containing testable claim results, environment hardware info, and git commit details.
- `app.py`: Gradio Space application demonstrating FlashBlock attention caching vs dense block diffusion with interactive visualizations and live benchmark runs.

---

## Isolated Worktree Setup

- Branch: `repro/flashblock-4jfunnghps`
- Worktree Path: `.worktrees/flashblock-4jfunnghps`
- CPU & Budget Constraints: Pure CPU execution, paid API cost = $0.00, strict TDD workflow.
