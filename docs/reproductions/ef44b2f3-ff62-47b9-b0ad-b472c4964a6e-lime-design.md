# Reproduction Design: LiME (Lightweight Mixture of Experts for Efficient Multimodal Multi-task Learning)

**Paper ID**: `KRSZj8z5Lr`
**Attempt ID**: `ef44b2f3-ff62-47b9-b0ad-b472c4964a6e`
**ArXiv**: `2604.02338`
**Target Claims**:
1. LiME shares a single PEFT adapter and applies lightweight expert-specific modulation vectors instead of replicating a full adapter per expert (Figure 1).
2. LiME combines zero-parameter routing, adaptive expert selection, n-gram routing granularity, PEFT compatibility, and a shared trainable PEFT module (Table 1).
3. Average benchmark results report LiME variants as competitive with or better than MoE-PEFT baselines while using fewer total trainable parameters (Table 2).
4. Efficiency experiments show LiME variants achieve higher throughput, shorter training time, and up to 4x fewer trainable parameters than corresponding MoE-PEFT methods (Figure 2).
5. LiME's expert modulation is theoretically bounded as an approximation to expert-specific PEFT, and CKA analysis reports similar representations to MoELoRA (Theorem 2, Table 3).

---

## 1. Executive Summary & Strategy

The paper **LiME** introduces a lightweight Mixture-of-Experts PEFT method. Traditional MoE-PEFT approaches replicate an entire LoRA / adapter module per expert, leading to parameter inflation as expert count grows. LiME solves this by maintaining a single shared PEFT adapter (e.g., shared LoRA matrices $A \in \mathbb{R}^{r \times d_{in}}, B \in \mathbb{R}^{d_{out} \times r}$) and modulating its output per expert using element-wise vector scaling $\mathbf{m}_e \in \mathbb{R}^r$ or $\mathbf{v}_e \in \mathbb{R}^{d_{out}}$.

Our reproduction will build a modular Python package (`lime_peft`) and evaluation pipeline verifying:
- **Architecture**: Shared LoRA adapter + expert modulation vectors $\mathbf{m}_e$ vs replicated MoE-LoRA adapters.
- **Parameter Efficiency**: Calculating exact parameter counts for LiME vs MoE-LoRA across $N_e \in \{2, 4, 8, 16\}$ experts.
- **Routing & Modulation Mechanics**: Verification of zero-parameter routing (token feature similarity or hashing), n-gram routing granularity, and adaptive top-$k$ expert selection.
- **Representation Fidelity & Bound**: Empirical evaluation of output representation cosine similarity between LiME modulated outputs and full MoE-LoRA expert outputs under Theorem 2 bounds.

---

## 2. System Architecture

```
submissions/lime-lightweight-mixture-of-experts-for-efficient-multimodal-multi-task-learning/
├── pyproject.toml
├── README.md
├── app.py
├── generate_evidence.py
├── lime_peft/
│   ├── __init__.py
│   ├── lime_layer.py        # Shared PEFT adapter + Expert modulation vectors (m_e, v_e)
│   ├── routing.py           # Zero-parameter routing, adaptive expert selection, n-gram granularity
│   └── metrics.py           # Parameter counting, parameter reduction ratio, CKA / Cosine similarity bounds
├── tests/
│   ├── test_lime_layer.py
│   ├── test_routing.py
│   └── test_metrics.py
├── evidence/
│   └── evidence.json
└── pages/
    └── logbook.md
```

---

## 3. Methodological Details

### 3.1 Shared PEFT Adapter with Expert Modulation (Figure 1)
For an input $X \in \mathbb{R}^{B \times d_{in}}$:
- Shared LoRA projection: $h = X A^T$ where $A \in \mathbb{R}^{r \times d_{in}}$
- Expert Modulation for expert $e$: $\tilde{h}_e = h \odot \mathbf{m}_e$ where $\mathbf{m}_e \in \mathbb{R}^r$
- Expert Output: $y_e = \tilde{h}_e B^T$ where $B \in \mathbb{R}^{d_{out} \times r}$
- MoE Output: $Y = \sum_{e \in \text{TopK}} g_e (y_e)$ where $g_e$ is zero-parameter routing gate.

### 3.2 Parameter Savings (Figure 2 & Table 2)
For $N_e$ experts with rank $r$:
- **MoE-LoRA (baseline)**: $N_e \times (d_{in} r + d_{out} r)$ parameters.
- **LiME (proposed)**: $(d_{in} r + d_{out} r) + N_e \times r$ parameters.
- Parameter reduction ratio: $\frac{\text{Params}_{\text{MoE-LoRA}}}{\text{Params}_{\text{LiME}}} \approx \frac{N_e (d_{in}+d_{out}) r}{(d_{in}+d_{out}) r + N_e r} \approx N_e$ for large $d$. For $N_e=4$, LiME achieves up to $4\times$ parameter reduction.

---

## 4. Empirical Evidence & Verification Protocol

- **Unit Tests**:
  - `test_lime_layer_forward`: Verifies forward pass shapes, shared weights, and expert modulation.
  - `test_zero_param_routing`: Verifies zero extra routing parameters and correct top-$k$ selection.
  - `test_parameter_reduction_ratio`: Verifies up to 4x parameter savings over MoE-LoRA.
- **Deterministic Evidence Generation**:
  - Computes parameter reduction ratios across $N_e \in \{2, 4, 8, 16\}$ experts.
  - Evaluates representation approximation bound (Theorem 2 CKA/Cosine similarity $> 0.95$).
  - Outputs reproducible `evidence/evidence.json` and `pages/logbook.md`.
