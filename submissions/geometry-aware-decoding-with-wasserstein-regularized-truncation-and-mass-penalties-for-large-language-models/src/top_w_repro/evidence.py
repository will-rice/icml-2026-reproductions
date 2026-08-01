"""Numerical audits and the machine-readable evidence bundle.

Every number reported in the logbook pages is computed here from fixed
seeds on CPU. Claim 3 (GSM8K benchmark table) is NOT reproduced: no
model was run, and the bundle says so explicitly.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from top_w_repro.decoder import (
    brute_force_subset_update,
    evaluate_decoding_metrics,
    nearest_set_potential,
    prefix_subset_update,
    subset_objective,
    top_w_mask,
    whiten_and_normalize,
)
from top_w_repro.upstream import load_upstream_module

ROOT = Path(__file__).resolve().parents[2]
PAPER_ID = "HSuU4xBmAv"
ATTEMPT_ID = "c1b6dd10-f227-4d24-89a0-17fb00ec9147"
PAPER_TITLE = (
    "Geometry-Aware Decoding with Wasserstein-Regularized Truncation and "
    "Mass Penalties for Large Language Models"
)
UPSTREAM_REVISION = (
    "arxiv:2602.10346v2"
    "+github.com/arashgholami/top-w-decoding@5949bfae5e6a81bc279c65923f1adc1c9f2e2059"
)
CLAIMS = [
    "Top-W decoding selects token subsets by optimizing a Wasserstein-entropy-mass "
    "objective using embedding-induced geometry (Section 3, Algorithm 1).",
    "The method instantiates a practical alternating decoder with an exact "
    "subset-update step inside a candidate-pool loop (Section 4.2).",
    "Top-W is evaluated against Min-p, Top-p, and Top-H on GSM8K across multiple "
    "temperatures and models (Table 1).",
    "Top-W is evaluated against the same decoding baselines on GPQA across "
    "multiple temperatures and models (Table 2).",
    "Judge-based open-ended evaluations report Top-W wins on more AlpacaEval "
    "and MT-Bench temperature-model tuples than the compared decoding methods "
    "(Figure 1, Figure 2).",
]
# Theorem 3.4(a) proves prefix optimality for beta - lam >= 0; every
# exactness config satisfies that hypothesis. The relaxation control
# below deliberately violates it.
PREFIX_CONFIGS = [
    {"lam": 2.2, "beta": 2.8, "geom_scale": 0.6},
    {"lam": 1.5, "beta": 2.4, "geom_scale": 0.3},
    {"lam": 3.0, "beta": 3.5, "geom_scale": 1.0},
    {"lam": 2.2, "beta": 2.2, "geom_scale": 0.6},
]
RELAXED_CONFIG = {"lam": 2.2, "beta": 1.0, "geom_scale": 0.6}
TEMPERATURES = [0.5, 0.7, 1.0, 1.5, 2.0]


def audit_prefix_vs_bruteforce(trials_per_config: int = 30, pool: int = 10) -> dict:
    """Verify Theorem 3.4: the prefix S-step equals brute-force enumeration.

    Reports deterministic operation counts rather than wall-clock times
    so the bundle is byte-stable across validation reruns.
    """
    total, exact_matches, value_gaps = 0, 0, []
    for config_index, config in enumerate(PREFIX_CONFIGS):
        for trial in range(trials_per_config):
            torch.manual_seed(1000 * config_index + trial)
            probs = torch.softmax(torch.randn(pool) * 2.0, dim=-1)
            embeddings = torch.nn.functional.normalize(
                torch.randn(pool, 8), dim=-1
            )
            potential = config["geom_scale"] * nearest_set_potential(
                embeddings, torch.arange(3)
            )
            prefix = prefix_subset_update(
                probs, potential, lam=config["lam"], beta=config["beta"]
            )
            best_subset, best_value = brute_force_subset_update(
                probs, potential, lam=config["lam"], beta=config["beta"]
            )
            prefix_value = subset_objective(
                probs, potential, prefix, lam=config["lam"], beta=config["beta"]
            )
            gap = abs(prefix_value - best_value)
            value_gaps.append(gap)
            total += 1
            if gap <= 1e-9:
                exact_matches += 1
    return {
        "pool_size": pool,
        "subsets_enumerated_per_trial": 2**pool - 1,
        "prefix_candidates_per_trial": pool,
        "configs": PREFIX_CONFIGS,
        "trials": total,
        "optimal_value_matches": exact_matches,
        "max_objective_gap": max(value_gaps),
        "passed": exact_matches == total,
    }


def audit_theorem_relaxation(trials: int = 120, pool: int = 10) -> dict:
    """Violating the beta >= lam hypothesis must break prefix optimality.

    Theorem 3.4(a) assumes beta - lam >= 0. This control runs the same
    brute-force comparison with beta < lam and counts instances where
    the pure prefix scan is strictly suboptimal, showing the hypothesis
    is load-bearing rather than incidental.
    """
    config = RELAXED_CONFIG
    counterexamples, worst_gap = 0, 0.0
    for trial in range(trials):
        torch.manual_seed(5000 + trial)
        probs = torch.softmax(torch.randn(pool) * 2.0, dim=-1)
        embeddings = torch.nn.functional.normalize(torch.randn(pool, 8), dim=-1)
        potential = config["geom_scale"] * nearest_set_potential(
            embeddings, torch.arange(3)
        )
        prefix = prefix_subset_update(
            probs, potential, lam=config["lam"], beta=config["beta"]
        )
        _, best_value = brute_force_subset_update(
            probs, potential, lam=config["lam"], beta=config["beta"]
        )
        gap = best_value - subset_objective(
            probs, potential, prefix, lam=config["lam"], beta=config["beta"]
        )
        if gap > 1e-6:
            counterexamples += 1
            worst_gap = max(worst_gap, gap)
    return {
        "config": config,
        "trials": trials,
        "prefix_suboptimal_instances": counterexamples,
        "worst_objective_gap": worst_gap,
        "passed": counterexamples >= 1,
    }


def audit_official_crosscheck(trials: int = 25) -> dict:
    """Our alternating decoder must keep the same tokens as the official code."""
    upstream = load_upstream_module()
    matches, kept_sizes = 0, []
    for trial in range(trials):
        torch.manual_seed(2000 + trial)
        logits = torch.randn(400) * 2.0
        embeddings = torch.randn(400, 24)
        ours = top_w_mask(
            logits, embeddings, temperature=0.7, top_m=64, alt_iters=9
        )
        emb = embeddings.numpy(force=True)
        mean = emb.mean(axis=0, keepdims=True)
        var = ((emb - mean) ** 2).mean(axis=0, keepdims=True)
        masked = upstream._topw_mask_logits(
            logits=logits.numpy(force=True).astype(np.float64),
            embeddings_full=emb,
            mean_full=mean.astype(np.float32),
            scale_full=(1.0 / np.sqrt(np.clip(var, 1e-6, None))).astype(
                np.float32
            ),
            temperature=0.7,
            top_m=64,
            init_top_p=0.999,
            alt_iters=9,
            geom_chunk=4096,
            geom_scale=0.6,
            lam_fixed=2.2,
            beta_override=2.8,
        )
        official = sorted(np.flatnonzero(np.isfinite(masked)).tolist())
        kept_sizes.append(len(official))
        if sorted(ours["kept"].tolist()) == official:
            matches += 1
    return {
        "trials": trials,
        "identical_kept_sets": matches,
        "mean_kept_size": sum(kept_sizes) / len(kept_sizes),
        "passed": matches == trials,
    }


def audit_alternating_convergence(trials: int = 40) -> dict:
    """The alternating decoder reaches a fixed point within the budget."""
    iteration_counts, converged = [], 0
    for trial in range(trials):
        torch.manual_seed(3000 + trial)
        logits = torch.randn(500) * 2.0
        embeddings = torch.randn(500, 32)
        result = top_w_mask(
            logits, embeddings, temperature=1.0, top_m=128, alt_iters=9
        )
        iteration_counts.append(result["iterations"])
        converged += int(result["converged"])
    return {
        "trials": trials,
        "converged": converged,
        "max_iterations": max(iteration_counts),
        "mean_iterations": sum(iteration_counts) / trials,
        "passed": converged == trials,
    }


def clustered_vocabulary(
    clusters: int = 40, per_cluster: int = 8, dim: int = 16
) -> torch.Tensor:
    centers = torch.randn(clusters, dim)
    noise = 0.05 * torch.randn(clusters, per_cluster, dim)
    return (centers[:, None, :] + noise).flatten(0, 1)


def audit_geometry_mechanism(trials: int = 20) -> dict:
    """Verify the geometry machinery and measure its selection influence.

    Gates: the f-step must equal the nearest-set W1 surrogate of
    Lemma 4.2 exactly, and the uniform-metric case must reduce to a
    top-probability prefix (Section 4.3). The shuffle comparison is an
    observation, not a gate: it measures whether re-assigning embeddings
    to tokens changes the selected subset at official defaults.
    """
    potential_max_error = 0.0
    uniform_prefix_matches = 0
    shuffle_jaccards = []
    prob_prefix_matches = 0
    for trial in range(trials):
        torch.manual_seed(4000 + trial)
        embeddings = clustered_vocabulary()
        vocab = embeddings.shape[0]
        logits = torch.randn(vocab) * 2.0
        probs = torch.softmax(logits, dim=-1)

        unit = whiten_and_normalize(embeddings)
        kept0 = torch.arange(7)
        fast = nearest_set_potential(unit, kept0)
        naive = torch.tensor(
            [
                -min(
                    float(
                        (1.0 - (unit[i] @ unit[j]).clamp(-1.0, 1.0)).clamp(
                            min=0.0
                        )
                    )
                    for j in kept0.tolist()
                )
                for i in range(vocab)
            ]
        )
        potential_max_error = max(
            potential_max_error, float((fast - naive).abs().max())
        )

        result = top_w_mask(logits, embeddings, temperature=1.0, top_m=128)
        shuffled_result = top_w_mask(
            logits,
            embeddings[torch.randperm(vocab)],
            temperature=1.0,
            top_m=128,
        )
        kept_a = set(result["kept"].tolist())
        kept_b = set(shuffled_result["kept"].tolist())
        shuffle_jaccards.append(len(kept_a & kept_b) / len(kept_a | kept_b))

        order = torch.argsort(probs, descending=True)
        prob_prefix = sorted(order[: len(kept_a)].tolist())
        prob_prefix_matches += int(sorted(kept_a) == prob_prefix)

        uniform = top_w_mask(
            logits, torch.ones(vocab, 16), temperature=1.0, top_m=128
        )
        uniform_prefix = sorted(order[: uniform["kept"].numel()].tolist())
        uniform_prefix_matches += int(
            sorted(uniform["kept"].tolist()) == uniform_prefix
        )

    return {
        "trials": trials,
        "potential_max_error": potential_max_error,
        "uniform_metric_prefix_matches": uniform_prefix_matches,
        "mean_shuffle_jaccard": sum(shuffle_jaccards) / trials,
        "probability_prefix_matches": prob_prefix_matches,
        "passed": (
            potential_max_error <= 1e-6
            and uniform_prefix_matches == trials
        ),
    }


def temperature_sweep() -> dict:
    """Distribution-shaping metrics on synthetic logits at the paper's temperatures."""
    torch.manual_seed(42)
    logits = torch.randn(500) * 2.0
    embeddings = torch.randn(500, 64)
    return {
        f"t_{temperature}": evaluate_decoding_metrics(
            logits, embeddings, temperature=temperature, top_m=128
        )
        for temperature in TEMPERATURES
    }


def build_bundle() -> dict:
    """Compute every audit and assemble the honest evidence bundle."""
    prefix = audit_prefix_vs_bruteforce()
    relaxation = audit_theorem_relaxation()
    crosscheck = audit_official_crosscheck()
    convergence = audit_alternating_convergence()
    controls = audit_geometry_mechanism()
    sweep = temperature_sweep()

    claim_shas = [
        hashlib.sha256(text.encode("utf-8")).hexdigest() for text in CLAIMS
    ]
    claim_results = {
        "claim_1": {
            "status": "verified" if controls["passed"] and convergence["passed"] else "inconclusive",
            "evidence": (
                "Numerical audit of the Wasserstein-entropy-mass objective on "
                "CPU: the f-step equals the Lemma 4.2 nearest-set W1 surrogate "
                f"exactly (max error {controls['potential_max_error']:.1e}); "
                "the S-step provably maximizes the geometry-dependent "
                "fixed-potential objective (see claim 2); the alternating "
                f"decoder converged in {convergence['converged']}"
                f"/{convergence['trials']} trials (mean "
                f"{convergence['mean_iterations']:.2f} iterations); with a "
                "uniform metric the kept set was a top-probability prefix in "
                f"{controls['uniform_metric_prefix_matches']}/{controls['trials']} "
                "trials (Section 4.3 reduction). Sensitivity finding: at the "
                "official defaults (warm_p=0.999, geom_scale=0.6) the final "
                "subsets equaled pure probability prefixes in "
                f"{controls['probability_prefix_matches']}/{controls['trials']} "
                "synthetic trials and were invariant to shuffling the token "
                f"embeddings (mean Jaccard "
                f"{controls['mean_shuffle_jaccard']:.3f}): with a warm start "
                "covering the pool, the nearest-set potential is zero on every "
                "warm-start member, so geometry influences selection only "
                "through expansion candidates."
            ),
            "limitations": (
                "Synthetic logits and synthetic embedding geometry on CPU; no "
                "language-model forward passes. The sensitivity finding is "
                "specific to these synthetic instances and official default "
                "hyperparameters; it does not measure behavior on real LLM "
                "next-token distributions."
            ),
        },
        "claim_2": {
            "status": "verified" if prefix["passed"] and crosscheck["passed"] else "inconclusive",
            "evidence": (
                "Under the theorem's beta >= lam hypothesis, the prefix-form "
                "exact S-step (Theorem 3.4a) matched brute-force enumeration of "
                f"all {prefix['subsets_enumerated_per_trial']} nonempty subsets "
                f"in {prefix['optimal_value_matches']}/{prefix['trials']} "
                f"trials across {len(PREFIX_CONFIGS)} configurations (max "
                f"objective gap {prefix['max_objective_gap']:.2e}; a "
                f"{prefix['prefix_candidates_per_trial']}-step linear scan "
                f"replaces {prefix['subsets_enumerated_per_trial']} subset "
                "evaluations per instance). "
                "Relaxation control: with beta < lam the pure prefix scan was "
                f"strictly suboptimal in "
                f"{relaxation['prefix_suboptimal_instances']}"
                f"/{relaxation['trials']} instances (worst gap "
                f"{relaxation['worst_objective_gap']:.3f}), confirming the "
                "hypothesis is load-bearing. The reimplemented alternating "
                "decoder kept token sets identical to the vendored official "
                "implementation in "
                f"{crosscheck['identical_kept_sets']}/{crosscheck['trials']} "
                "random instances."
            ),
            "limitations": (
                "Brute force is feasible only for 10-token pools; larger pools "
                "rely on the theorem, not enumeration."
            ),
        },
        "claim_3": {
            "status": "unreplicated",
            "evidence": (
                "Not reproduced. The Table 1 comparison requires GSM8K decoding "
                "runs across three instruction-tuned models and five "
                "temperatures. No language model was executed in this "
                "reproduction; no GSM8K accuracy numbers exist here. The "
                "official evaluation harness (run.sh, huggingface.py) is pinned "
                "in the upstream manifest for an independent GPU reproduction."
            ),
            "limitations": (
                "The synthetic temperature sweep in this bundle characterizes "
                "distribution shaping only and is NOT evidence for benchmark "
                "accuracy claims."
            ),
        },
        "claim_4": {
            "status": "unreplicated",
            "evidence": (
                "Not reproduced. The Table 2 comparison requires GPQA decoding "
                "runs across the same instruction-tuned models and "
                "temperatures as Table 1. No language model was executed in "
                "this reproduction; no GPQA accuracy numbers exist here. The "
                "official repository pinned in the upstream manifest ships "
                "run_gpqa.sh as the entry point for an independent GPU "
                "reproduction. The decoding mechanism GPQA would exercise is "
                "the same audited mechanism as claims 1-2: the objective, "
                "exact S-step, and official-code cross-check numbers on the "
                "claim 1 and claim 2 pages are the only mechanism-level "
                "evidence this attempt provides."
            ),
            "limitations": (
                "No GPQA decoding runs were performed; the synthetic "
                "temperature sweep says nothing about GPQA accuracy."
            ),
        },
        "claim_5": {
            "status": "unreplicated",
            "evidence": (
                "Not reproduced. The AlpacaEval and MT-Bench win-rate "
                "comparisons (Figure 1, Figure 2) require open-ended "
                "generation with multiple models and temperatures plus a "
                "judge model. No language model or judge was executed in this "
                "reproduction, and the challenge budget excludes paid judge "
                "APIs (recorded cost USD 0.00). The official repository "
                "pinned in the upstream manifest ships alpaca_generate_w.py "
                "for generation; judge-side evaluation would additionally "
                "require the AlpacaEval and MT-Bench harnesses."
            ),
            "limitations": (
                "No open-ended generations or judge evaluations were "
                "performed; no win-rate numbers exist in this attempt."
            ),
        },
    }

    return {
        "paper_id": PAPER_ID,
        "attempt_id": ATTEMPT_ID,
        "paper_title": PAPER_TITLE,
        "upstream_revision": UPSTREAM_REVISION,
        "estimated_api_cost_usd": 0.0,
        "environment": {
            "device": "cpu",
            "pinned_by": "uv.lock",
        },
        "target_claims": [
            {
                "id": f"claim_{index + 1}",
                "text": text,
                "challenge_claim_sha256": sha,
            }
            for index, (text, sha) in enumerate(zip(CLAIMS, claim_shas))
        ],
        "claim_results": claim_results,
        "audits": {
            "prefix_vs_bruteforce": prefix,
            "theorem_relaxation": relaxation,
            "official_crosscheck": crosscheck,
            "alternating_convergence": convergence,
            "geometry_mechanism": controls,
        },
        "metrics": sweep,
        "upstream": json.loads(
            (ROOT / "evidence" / "inputs" / "upstream_manifest.json").read_text()
        ),
        "commands": [
            "uv run --project . python generate_evidence.py",
            "uv run --project . python -m pytest tests -q",
        ],
    }
