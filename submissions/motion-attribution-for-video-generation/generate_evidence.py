#!/usr/bin/env python3
"""Generate reproduction evidence for Motion Attribution for Video Generation.

Runs deterministic ground-truth experiments and writes evidence_summary.json
plus the judge-visible pages/report.md from the same computed values. VBench
fine-tuning comparisons and the paper's human study are unreplicated: they
require training video generation models and human raters, and their numbers
must never be synthesized. An earlier revision of this package fabricated
them; those values are removed.
"""

import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent / "src"))

from motive.attribution import (
    experiment_dynamics_vs_magnitude,
    experiment_frame_length_bias,
    experiment_motion_mask_localization,
)


def run_evidence() -> dict:
    localization = experiment_motion_mask_localization()
    length_bias = experiment_frame_length_bias()
    dynamics = experiment_dynamics_vs_magnitude()

    unreplicated_vbench = {
        "status": "unreplicated",
        "reason": "requires fine-tuning a video generation model on Motive-selected data and running the VBench harness; not feasible in this CPU-only package and not fabricated",
    }
    unreplicated_human = {
        "status": "unreplicated",
        "reason": "a human pairwise-preference study requires human raters; it cannot be reproduced computationally and must not be synthesized",
    }

    target_claims = [
        {
            "claim": "Motive computes motion-specific data attribution by applying motion-weighted loss masks so gradients emphasize dynamic regions rather than static appearance (Section 3.4).",
            "challenge_claim_sha256": "f2a60a6b6eab09e8593eb79445c51826b5a9f54d58898bdeeb731645ca4ce8e7",
            "status": "partially_reproduced",
            "scope": "toy-scale ground-truth videos",
            "evidence_details": localization,
        },
        {
            "claim": "The method includes a video-specific frame-length bias fix to reduce spurious attribution to longer clips (Section 3.3).",
            "challenge_claim_sha256": "7a535f3a1dbe7198f27fdf3dfda0d03f6481f16a5ec10e9f0b6d55a9b5ecc78d",
            "status": "partially_reproduced",
            "scope": "toy-scale measured gradient norms",
            "evidence_details": {
                key: length_bias[key]
                for key in (
                    "clip_lengths",
                    "raw_scores",
                    "normalized_scores",
                    "raw_growth_ratio_longest_vs_shortest",
                    "normalized_growth_ratio_longest_vs_shortest",
                )
            },
        },
        {
            "claim": "Fine-tuning on Motive-selected data improves VBench motion smoothness and dynamic degree over baselines while using only a fraction of the training data (Table 1).",
            "challenge_claim_sha256": "5717ff1efe49e55f86e7f9ad9a42b8f772cff43145b61db7d6287249e28af09c",
            **unreplicated_vbench,
        },
        {
            "claim": "Human evaluation reports a 74.1% preference win rate for Motive-selected fine-tuning compared with the pretrained base model (Table 2).",
            "challenge_claim_sha256": "c42b1b15ccdaa6a62b2d93bb2865f7741b9223dec64af43daa615b221d545aa5",
            **unreplicated_human,
        },
        {
            "claim": "Motive computes motion-specific influence by detecting motion, forming motion-magnitude patches, and applying motion masks to gradient-based data attribution (Figure 1)",
            "challenge_claim_sha256": "3f4ea59bc4d1902358ebcf59440b816c22b24a2a3cff44e0e000b31c25d736f5",
            "status": "partially_reproduced",
            "scope": "toy-scale ground-truth videos",
            "evidence_details": {
                "mean_mask_in_moving_patches": localization["mean_mask_in_moving_patches"],
                "mean_mask_in_static_patches": localization["mean_mask_in_static_patches"],
                "fraction_of_mask_weight_on_true_motion": localization[
                    "fraction_of_mask_weight_on_true_motion"
                ],
            },
        },
        {
            "claim": "Motive-selected fine-tuning data improves VBench motion smoothness and dynamic degree compared with random and baseline data-selection methods (Table 1)",
            "challenge_claim_sha256": "b791befb7947d259a5936358f6618bb77ecc93f3c563c003c17cc4ab4687c1b1",
            **unreplicated_vbench,
        },
        {
            "claim": "Human pairwise evaluation reports a 74.1% preference win rate for Motive-selected fine-tuning over the pretrained base model (Table 2)",
            "challenge_claim_sha256": "955fbe7fdbeb7fffd0ba876b0255a7cbeeab8954a9ba829554179377f58ba3b6",
            **unreplicated_human,
        },
        {
            "claim": "Frame-length normalization prevents attribution rankings from being biased toward longer clips and yields more coherent top-ranked motion samples (Figure 4)",
            "challenge_claim_sha256": "5f66baf089fb2e5f58e7e8e8b064fab2782282c721cdba8a1d47c2be8a59ed3a",
            "status": "partially_reproduced",
            "scope": "toy-scale ranking with known ground truth",
            "evidence_details": {
                key: length_bias[key]
                for key in (
                    "pair_raw_scores",
                    "pair_normalized_scores",
                    "raw_ranking",
                    "normalized_ranking",
                )
            },
        },
        {
            "claim": "Motive's influence scores are not merely selecting high-motion clips; influential clips are those predicted to improve target motion dynamics (Figure 6)",
            "challenge_claim_sha256": "efb00940faba0794a14bbfa17a7c2fd436d09282fd8313e8eab56799413b8855",
            "status": "partially_reproduced",
            "scope": "toy-scale proxy influence",
            "evidence_details": dynamics,
        },
    ]

    summary = {
        "paper_id": "zAl9heLw4q",
        "title": "Motion Attribution for Video Generation",
        "slug": "motion-attribution-for-video-generation",
        "target_claims": target_claims,
        "all_target_claims_verified": False,
        "provenance": "all numbers computed by generate_evidence.py with pinned seeds; fabricated VBench and human-preference values removed 2026-08-01",
    }

    root = pathlib.Path(__file__).parent
    with open(root / "evidence_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
        f.write("\n")
    (root / "pages").mkdir(exist_ok=True)
    write_report(root / "pages" / "report.md", localization, length_bias, dynamics)
    return summary


def write_report(path: pathlib.Path, localization: dict, length_bias: dict, dynamics: dict) -> None:
    path.write_text(
        f"""# Reproduction Report: Motion Attribution for Video Generation (Motive)

**Paper ID:** `zAl9heLw4q`

## Scope

CPU-only, toy-scale mechanism reproduction on deterministic synthetic videos
with known moving regions. The VBench fine-tuning comparisons (Table 1) and
the 74.1% human-preference study (Table 2) are **unreplicated** - they need
trained video models and human raters. An earlier revision fabricated those
numbers; they have been removed and nothing here restates them.

## Motion-weighted attribution localizes true motion (Sections 3.4, Figure 1): partially reproduced

On a video with one moving square over a static textured background, the
motion mask concentrates on truly-moving patches:

| measurement | value |
| --- | --- |
| mean mask weight on moving patches | {localization['mean_mask_in_moving_patches']} |
| mean mask weight on static patches | {localization['mean_mask_in_static_patches']} |
| fraction of total mask weight on true motion | {localization['fraction_of_mask_weight_on_true_motion']} |

With uniform gradients, motion weighting shrinks the attribution norm from
{localization['unmasked_attribution_norm']} to
{localization['masked_attribution_norm']}, i.e. static-appearance gradients
are suppressed and dynamic regions dominate the score.

## Frame-length bias fix (Section 3.3, Figure 4): partially reproduced

Measured attribution norms grow with clip length on statistically identical
clips (lengths {length_bias['clip_lengths']}): raw scores
{length_bias['raw_scores']} (ratio
{length_bias['raw_growth_ratio_longest_vs_shortest']}x longest/shortest);
after S_raw / sqrt(T/T_ref) normalization:
{length_bias['normalized_scores']} (ratio
{length_bias['normalized_growth_ratio_longest_vs_shortest']}x).

Ranking with known ground truth: a strongly-moving 8-frame clip versus a
weakly-moving 32-frame clip scores {length_bias['pair_raw_scores']} raw
(ranking {length_bias['raw_ranking']}) and
{length_bias['pair_normalized_scores']} normalized (ranking
{length_bias['normalized_ranking']}): normalization recovers the truly more
influential short clip.

## Influence tracks dynamics, not magnitude (Figure 6): partially reproduced

An incoherent jitter clip has higher raw motion energy
({dynamics['jitter_clip_motion_energy']}) than a coherent translating clip
({dynamics['coherent_clip_motion_energy']}), yet the coherent clip's
motion-masked field is far more similar to the target dynamic (cosine
{dynamics['coherent_influence_cosine']} versus
{dynamics['jitter_influence_cosine']}). High motion magnitude alone does not
make a clip influential for a target dynamic, matching the paper's Figure 6
mechanism.

## Unreplicated claims

Table 1 (VBench improvements) and Table 2 (74.1% human preference) are
reported without numbers: producing them requires fine-tuning video
generation models and running a human study. Fabricating them would be
misconduct.

## Reproducibility

`uv run python generate_evidence.py` regenerates `evidence_summary.json` and
this page byte-identically (pinned seeds, no timestamps).
"""
    )


if __name__ == "__main__":
    run_evidence()
    print("Evidence generated.")
