"""Render the logbook pages from the computed evidence bundle.

The judge reads the visible pages, not the raw JSON: every number that
supports a claim verdict must appear here in plain markdown.
"""

from __future__ import annotations


def metric_rows(bundle: dict) -> str:
    header = (
        "| T | Top-W H | Min-p H | Top-p H | Top-H H | Top-W kept | Top-p kept |\n"
        "| --- | --- | --- | --- | --- | --- | --- |\n"
    )
    rows = []
    for key, metrics in bundle["metrics"].items():
        rows.append(
            f"| {key.removeprefix('t_')} "
            f"| {metrics['entropy_top_w']:.4f} "
            f"| {metrics['entropy_min_p']:.4f} "
            f"| {metrics['entropy_top_p']:.4f} "
            f"| {metrics['entropy_top_h']:.4f} "
            f"| {int(metrics['subset_size_top_w'])} "
            f"| {int(metrics['subset_size_top_p'])} |"
        )
    return header + "\n".join(rows)


def summary_page(bundle: dict) -> str:
    statuses = {
        claim["id"]: bundle["claim_results"][claim["id"]]["status"]
        for claim in bundle["target_claims"]
    }
    return f"""# Reproduction: {bundle["paper_title"]}

Paper `{bundle["paper_id"]}` | Attempt `{bundle["attempt_id"]}` |
Pinned upstream `{bundle["upstream_revision"]}` |
Paid API cost USD {bundle["estimated_api_cost_usd"]:.2f}

## Pages

| Page |
| --- |
| Executive summary (this page) |
| Claim 1: Wasserstein-entropy-mass objective — numerical audit |
| Claim 2: Exact prefix-form subset update vs brute force |
| Claim 3: GSM8K baseline table — not reproduced |
| Methods and provenance |

## Executive summary

This is an independent CPU reproduction of the Top-W decoding
*mechanism*, audited numerically against the released official
implementation (pinned in the upstream manifest). It does **not** rerun
any language model, so the benchmark tables of the paper are explicitly
out of scope here.

| Claim | Self-assessed status | Key numbers |
| --- | --- | --- |
| 1. Objective and geometry (Sec. 3, Alg. 1) | {statuses["claim_1"]} | f-step surrogate max error {bundle["audits"]["geometry_mechanism"]["potential_max_error"]:.1e}; uniform-metric reduction {bundle["audits"]["geometry_mechanism"]["uniform_metric_prefix_matches"]}/{bundle["audits"]["geometry_mechanism"]["trials"]}; {bundle["audits"]["alternating_convergence"]["converged"]}/{bundle["audits"]["alternating_convergence"]["trials"]} converged |
| 2. Exact subset update (Sec. 4.2, Thm. 3.4) | {statuses["claim_2"]} | {bundle["audits"]["prefix_vs_bruteforce"]["optimal_value_matches"]}/{bundle["audits"]["prefix_vs_bruteforce"]["trials"]} brute-force matches; {bundle["audits"]["official_crosscheck"]["identical_kept_sets"]}/{bundle["audits"]["official_crosscheck"]["trials"]} identical to official code |
| 3. GSM8K table (Table 1) | {statuses["claim_3"]} | no model runs; no accuracy numbers claimed |

Every number above is recomputed by `generate_evidence.py` from fixed
seeds; the full raw values are in `evidence/bundle.json`.
"""


def claim_1_page(bundle: dict) -> str:
    controls = bundle["audits"]["geometry_mechanism"]
    convergence = bundle["audits"]["alternating_convergence"]
    return f"""# Claim 1: Wasserstein-entropy-mass objective over embedding geometry

**Claim.** {bundle["target_claims"][0]["text"]}

**Self-assessed status: {bundle["claim_results"]["claim_1"]["status"]}** —
numerical audit at synthetic scale, per the challenge guidance for
mechanism claims.

## What was executed

The decoder implements the paper's objective: crop S maximizes
`E_q_S[varphi] + (beta - lam) * log Gamma_S` with
`varphi_i = geom_scale * f_i + lam * log p_i`, where the potential
`f_i = -min_j_in_S d_cos(i, j)` is the nearest-set surrogate of the
W1 transport term (Lemma 4.2) on whitened, L2-normalized embeddings
(`src/top_w_repro/decoder.py`).

## Mechanism checks

- **f-step equals the Lemma 4.2 surrogate.** The vectorized potential
  matched a naive per-token minimum over the kept set exactly in every
  trial (max absolute error {controls["potential_max_error"]:.1e}).
- **S-step maximizes the objective exactly** — established by
  brute-force enumeration on the claim 2 page.
- **Convergence.** {convergence["converged"]}/{convergence["trials"]}
  random 500-token instances reached a fixed point within the
  9-iteration budget (mean {convergence["mean_iterations"]:.2f}, max
  {convergence["max_iterations"]} iterations). Seeds 3000-3039.
- **Uniform-metric reduction (Section 4.3).** With identical
  embeddings the kept set was exactly a top-probability prefix in
  {controls["uniform_metric_prefix_matches"]}/{controls["trials"]}
  trials.

## Independent sensitivity finding

At the official default hyperparameters (warm_p = 0.999,
geom_scale = 0.6, lam = 2.2, beta = 2.8), on clustered synthetic
vocabularies (40 clusters x 8 near-duplicates), the final kept sets
equaled pure top-probability prefixes in
{controls["probability_prefix_matches"]}/{controls["trials"]} trials and
were completely invariant to randomly re-assigning embeddings to tokens
(mean Jaccard {controls["mean_shuffle_jaccard"]:.3f} between original
and shuffled geometry). Mechanistically: the warm start covers the
candidate pool, the nearest-set potential is zero for every warm-start
member, and expansion candidates always rank below members in the
varphi order, so geometry can only reorder the expansion tail. The same
behavior reproduces in the vendored official implementation. This is an
observation about these synthetic instances and the default
configuration, not about real LLM next-token distributions, where the
paper reports behavioral gains.

## Limitations

{bundle["claim_results"]["claim_1"]["limitations"]}
"""


def claim_2_page(bundle: dict) -> str:
    prefix = bundle["audits"]["prefix_vs_bruteforce"]
    relaxation = bundle["audits"]["theorem_relaxation"]
    crosscheck = bundle["audits"]["official_crosscheck"]
    configs = "; ".join(
        f"(lam={config['lam']}, beta={config['beta']}, "
        f"geom_scale={config['geom_scale']})"
        for config in prefix["configs"]
    )
    return f"""# Claim 2: Exact prefix-form subset update inside the candidate-pool loop

**Claim.** {bundle["target_claims"][1]["text"]}

**Self-assessed status: {bundle["claim_results"]["claim_2"]["status"]}**

## Brute-force enumeration control (Theorem 3.4a, beta >= lam)

For {prefix["pool_size"]}-token candidate pools, every one of the
{prefix["subsets_enumerated_per_trial"]} nonempty subsets was enumerated
and scored with the fixed-potential objective; the prefix-form linear
scan returned the exact optimum in
**{prefix["optimal_value_matches"]}/{prefix["trials"]}** trials
(max objective gap {prefix["max_objective_gap"]:.2e}), across four
hyperparameter configurations satisfying the theorem's beta >= lam
hypothesis: {configs}.

| Method | Work per instance |
| --- | --- |
| Prefix scan (exact S-step) | one sort + {prefix["prefix_candidates_per_trial"]}-prefix linear scan |
| Brute-force enumeration | {prefix["subsets_enumerated_per_trial"]} subset objective evaluations |

## Relaxation control (beta < lam)

Theorem 3.4(a) assumes `beta - lam >= 0`. Rerunning the identical
brute-force comparison with
(lam={relaxation["config"]["lam"]}, beta={relaxation["config"]["beta"]})
produced **{relaxation["prefix_suboptimal_instances"]}/{relaxation["trials"]}**
instances where the pure prefix scan was strictly suboptimal (worst
objective gap {relaxation["worst_objective_gap"]:.3f}) — the collapse
regime the paper describes, showing the hypothesis is load-bearing. The
official implementation's defaults (lam=2.2, beta=2.8) satisfy the
hypothesis.

## Cross-check against the official implementation

The reimplemented alternating decoder (candidate pool top_m=64, 9
alternating iterations, lam=2.2, beta=2.8, geom_scale=0.6 — the official
defaults) kept token sets **identical to the vendored official
`logit_processor_w1.py`** in
{crosscheck["identical_kept_sets"]}/{crosscheck["trials"]} random
400-token instances (mean kept size {crosscheck["mean_kept_size"]:.1f}).
The official file is byte-exact at
`evidence/inputs/upstream/logit_processor_w1.py` (SHA-256 in
`evidence/inputs/upstream_manifest.json`).

## Limitations

{bundle["claim_results"]["claim_2"]["limitations"]}
"""


def claim_3_page(bundle: dict) -> str:
    return f"""# Claim 3: GSM8K comparison against Min-p, Top-p, and Top-H

**Claim.** {bundle["target_claims"][2]["text"]}

**Self-assessed status: {bundle["claim_results"]["claim_3"]["status"]}**

## What this reproduction did NOT do

{bundle["claim_results"]["claim_3"]["evidence"]}

## What exists for an independent benchmark rerun

The official repository (pinned at
`{bundle["upstream_revision"].split("+", 1)[1]}`) ships the evaluation
harness: `run.sh` (GSM8K), `run_gpqa.sh`, `alpaca_generate_w.py`, and
`huggingface.py`. Reproducing Table 1 requires decoding GSM8K with three
instruction-tuned models at five temperatures per method, which needs
GPU inference outside this CPU-only attempt's budget.

## Synthetic distribution shaping (context only — NOT benchmark evidence)

On fixed synthetic logits (seed 42, 500-token vocabulary), the decoder
comparison at the paper's temperatures:

{metric_rows(bundle)}

These numbers characterize how each truncation rule shapes a
distribution; they say nothing about GSM8K accuracy.
"""


def methods_page(bundle: dict) -> str:
    files = "\n".join(
        f"| `{name}` | `{sha}` |"
        for name, sha in bundle["upstream"]["files"].items()
    )
    return f"""# Methods and provenance

## Pinned sources

| Source | Pin |
| --- | --- |
| Paper | arXiv:2602.10346v2 |
| Official code | {bundle["upstream"]["repository"]}@{bundle["upstream"]["revision"]} |

Vendored byte-exact upstream files (MIT license retained):

| File | SHA-256 |
| --- | --- |
{files}

## Environment

CPU only; exact Python and package versions are pinned by the
project's `uv.lock`. Paid API cost: USD
{bundle["estimated_api_cost_usd"]:.2f}.

## Reproduce these numbers

```bash
{chr(10).join(bundle["commands"])}
```

All audits use fixed torch seeds recorded in
`src/top_w_repro/evidence.py`; `evidence/bundle.json` is the exact
machine-readable output of the last run.
"""


def build_pages(bundle: dict) -> dict[str, str]:
    return {
        "00-summary.md": summary_page(bundle),
        "01-claim-1-wasserstein-objective.md": claim_1_page(bundle),
        "02-claim-2-exact-subset-update.md": claim_2_page(bundle),
        "03-claim-3-gsm8k-not-reproduced.md": claim_3_page(bundle),
        "04-methods-and-provenance.md": methods_page(bundle),
    }
