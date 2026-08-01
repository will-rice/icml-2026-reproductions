# FAC Synthesis Reproduction Report

Paper: 71029 - Less is Enough: Synthesizing Diverse Data in Feature Space of LLMs
Attempt: 19313a84-4cd7-4266-bc6c-8588b3549670
Snapshot: 69c8f39e61726cd423f2bebc01d57b5303b5fcd87e319f3ed7728dfce97b42d1

## Upstream Pins

- arXiv source SHA-256: `31cae9ef0d879106dd296bd9dc4a9981145c33ac18e702637168169d96f8025f`
- GitHub revision: `d3622dec55123c0eff4c079db9e1a59403f08d1b`
- HF demo revision: `2d41421eb288434e2ecd15815ec712e5b3d80033`
- HF dataset API access: `unavailable` (HTTP 401)

## Claim Results

### 010f1831b580319c66761587c1e8cb2ed533475cee7ee199b46c68d4eaaa4f77

Status: `toy`

Feature Activation Coverage measures post-training data diversity in an interpretable LLM feature space rather than using only text-level diversity metrics (Section 3).

Independent toy FAC computation uses the anchor-task feature set as denominator and generated feature intersection as numerator. Official artifacts pin SAE feature-space code paths, but no full SAE activation run was performed.

### 2eb2a85344edfdb5f2e017b8ede642fd6098dc48fe0bfb3506bec207aa5c11c8

Status: `toy`

FAC Synthesis uses sparse autoencoders to identify missing features in a seed dataset and generate synthetic samples that activate those features (Section 4).

Toy set arithmetic reproduces the missing-feature operation. Official scripts expose SAE missing-feature selection and two-stage targeted generation, but large-model synthesis was not run.

### aa863c883e3570fb5243d7d552f94d618ab83d157571a3ce3dec6a6886424789

Status: `inconclusive`

The paper reports consistent diversity and downstream-performance improvements on instruction following, toxicity detection, reward modeling, and behavior steering (Section 5).

The TeX source and official training scripts cover the four named tasks, but GPU fine-tuning and external evaluators were not rerun. Paper-reported values are context only.

### d51e98ace1136e4e71581061b941585a4f682d2aecdc13b3acdc78e5708f9baf

Status: `toy`

Using 100% of selected missing features outperforms lower selected-feature ratios across the four downstream tasks in the reported ablation (Table 7).

The TeX feature-ratio table was parsed and checked for monotonic ordering. This verifies table integrity, not independent performance.

### 6a15477d006c35f360169101410ea481348874a4374e64e3ea15f91a90a490f5

Status: `toy`

The paper identifies a shared interpretable feature space across LLaMA, Mistral, and Qwen that enables cross-model knowledge transfer (Section 6).

Three public SAE checkpoint repositories and model-specific layer settings were pinned. Cross-model transfer performance was not recomputed.
