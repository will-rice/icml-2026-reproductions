# CapBencher Reproduction Design Specification

**Paper ID:** `oCNT5PcMSQ`
**Title:** How Can I Publish My LLM Benchmark Without Giving the True Answers Away? (CapBencher)
**Authors:** Takashi Ishida, Thanawat Lodkaew, Ikko Yamane
**Upstream Pin:** `ishida-lab/capbencher@9f933d0757549e8e44b72fe2433f568767dab5b6`
**Dedicated Space:** `wrice/repro-capbencher-ocnt5pcmsq`

---

## 1. Objective and Target Claims

The goal of this reproduction is to independently implement the core mathematical formulation, exact binomial contamination testing, monotonic accuracy mapping, and model-merge hacking simulation for CapBencher, and generate clean machine-readable evidence for the ICML 2026 Agent Repro Challenge.

### Target Claims
1. **Bayes Accuracy Capping:** CapBencher caps Bayes accuracy by injecting randomness among logically correct answers, so above-cap performance can signal leakage or gaming (Figure 1).
2. **Monotonic Progress Tracking:** Capped benchmark accuracy remains monotonically related to original benchmark accuracy for tracking and ranking LLM improvement (Figure 2).
3. **Exact Binomial Contamination Test:** The paper uses exact binomial-test p-values rather than asymptotic approximations for contamination detection (Section 4).
4. **Model-Merge Hacking Simulation:** In a model-merge hacking simulation, the merged model's 56.52% accuracy is flagged as contaminated at the 5% significance level (Table 1).

---

## 2. Mathematical Framework & Algorithms

### A. Bayes Accuracy Capping
When a benchmark question $X$ has $K$ logically correct answer choices $F(X) = \{a_1, \dots, a_K\}$, CapBencher randomly selects a single realized solution $Y \in F(X)$ with uniform probability $P(Y=a_j|X) = 1/K$.
The theoretical maximum (Bayes) accuracy achievable by any non-memorizing model is:
$$\alpha_{\text{Bayes}} = \max_{a \in \mathcal{A}} P(Y=a|X) = \frac{1}{K}$$
surpassing $\alpha_{\text{Bayes}}$ provides a mathematically rigorous signal of test-set memorization or data contamination.

### B. Monotonic Score Relationship
By Theorem 1 of the paper, the expected capped accuracy $s_{\text{capped}}(X)$ relates to the original accuracy $s_{\text{orig}}(X)$ via an increasing affine transformation:
$$s_{\text{capped}}(X) = \frac{1}{L} + \frac{L-1}{L} s_{\text{orig}}(X)$$
where $L$ is the number of randomized options ($L = K$). Since $\frac{L-1}{L} > 0$, ranking and monotonic improvement across LLM model generations are strictly preserved.

### C. Exact One-Sided Binomial Contamination Test
For $n$ test questions evaluated under Bayes accuracy cap $\alpha$:
- Null hypothesis $H_0: p_{\text{model}} \le \alpha$ (uncontaminated).
- Alternative hypothesis $H_1: p_{\text{model}} > \alpha$ (contaminated).
Given $k$ correct answers out of $n$:
$$p\text{-value} = \sum_{i=k}^{n} \binom{n}{i} \alpha^i (1-\alpha)^{n-i}$$
If $p\text{-value} \le 0.05$, $H_0$ is rejected at the 5% significance level.

### D. Model-Merge Hacking Simulation
Simulates the model-merge optimization scenario on $n=1000$ questions with $K=2$ ($\alpha = 0.50$).
The merged model achieves $k=565$ correct answers ($56.52\%$ accuracy).
Exact binomial test yields $p \approx 0.0000196 \le 0.05$, confirming contamination detection at $\alpha = 0.05$.

---

## 3. Architecture & Code Layout

```
submissions/capbencher/
├── pyproject.toml
├── capbencher/
│   ├── __init__.py
│   ├── core.py
│   └── simulation.py
├── tests/
│   └── test_capbencher.py
├── evidence/
│   └── bundle.json
└── app.py
```

---

## 4. Verification & Deployment

- Strict TDD: Unit tests written first, asserting exact expected mathematical properties and p-values.
- Hugging Face Space: `wrice/repro-capbencher-ocnt5pcmsq` (Gradio UI presenting interactive binomial test, score mapping, and simulation evidence).
- Bounded Judging & Improvement: Automated polling and candidate improvement verification.
