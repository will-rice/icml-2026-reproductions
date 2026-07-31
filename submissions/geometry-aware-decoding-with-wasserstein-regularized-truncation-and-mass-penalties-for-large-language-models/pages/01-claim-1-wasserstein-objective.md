# Claim 1: Wasserstein-entropy-mass objective over embedding geometry

**Claim.** Top-W decoding selects token subsets by optimizing a Wasserstein-entropy-mass objective using embedding-induced geometry (Section 3, Algorithm 1).

**Self-assessed status: verified** —
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
  trial (max absolute error 1.2e-07).
- **S-step maximizes the objective exactly** — established by
  brute-force enumeration on the claim 2 page.
- **Convergence.** 40/40
  random 500-token instances reached a fixed point within the
  9-iteration budget (mean 2.00, max
  2 iterations). Seeds 3000-3039.
- **Uniform-metric reduction (Section 4.3).** With identical
  embeddings the kept set was exactly a top-probability prefix in
  20/20
  trials.

## Independent sensitivity finding

At the official default hyperparameters (warm_p = 0.999,
geom_scale = 0.6, lam = 2.2, beta = 2.8), on clustered synthetic
vocabularies (40 clusters x 8 near-duplicates), the final kept sets
equaled pure top-probability prefixes in
20/20 trials and
were completely invariant to randomly re-assigning embeddings to tokens
(mean Jaccard 1.000 between original
and shuffled geometry). Mechanistically: the warm start covers the
candidate pool, the nearest-set potential is zero for every warm-start
member, and expansion candidates always rank below members in the
varphi order, so geometry can only reorder the expansion tail. The same
behavior reproduces in the vendored official implementation. This is an
observation about these synthetic instances and the default
configuration, not about real LLM next-token distributions, where the
paper reports behavioral gains.

## Limitations

Synthetic logits and synthetic embedding geometry on CPU; no language-model forward passes. The sensitivity finding is specific to these synthetic instances and official default hyperparameters; it does not measure behavior on real LLM next-token distributions.
