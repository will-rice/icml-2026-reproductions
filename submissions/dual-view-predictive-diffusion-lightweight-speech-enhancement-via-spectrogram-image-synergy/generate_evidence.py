"""Generate evidence summary JSON and judge-visible report for DVPD.

All numbers are computed by the experiments in src/dvpd/model.py with pinned
seeds. Dataset-scale claims (WSJ0-UNI state of the art, OOD generalization,
VBDMD tables) are unreplicated: they require training the full models on real
speech corpora. An earlier revision fabricated the MACs figure (baseline x
0.395) and verified benchmark claims off a random forward pass; those values
are removed.
"""

import json
import sys
from pathlib import Path

src_dir = Path(__file__).parent / "src"
sys.path.insert(0, str(src_dir))

from dvpd.model import (
    experiment_architecture_and_macs,
    experiment_fanc_band_allocation,
    experiment_interaction_coupling,
    experiment_toy_ablation,
)


def ablation_observation(ablation: dict) -> str:
    variants = ablation["heldout_denoised_mse_by_variant"]
    full = variants["full"]
    supported = [
        name
        for name, mse in variants.items()
        if name != "full" and mse > full
    ]
    contradicted = [
        name
        for name, mse in variants.items()
        if name != "full" and mse <= full
    ]
    return (
        f"at toy scale, removing {', '.join(sorted(supported))} worsens held-out MSE "
        f"(supporting their contribution), while removing {', '.join(sorted(contradicted))} "
        "improves it - the FANC contribution claim is not supported at this scale; "
        "TLB could not be ablated"
    )


def main() -> dict:
    architecture = experiment_architecture_and_macs()
    coupling = experiment_interaction_coupling()
    fanc = experiment_fanc_band_allocation()
    ablation = experiment_toy_ablation()

    claims = [
        {
            "claim": "DVPD uses a dual-branch predictive/diffusion architecture that treats spectrograms as both acoustic frequency-domain structures and visual textures (Figure 1; Figure 2).",
            "challenge_claim_sha256": "e65804401638b4d78d121891377320c7e9c30b628db6ee5398fdbd99f1acf8bf",
            "status": "partially_reproduced",
            "scope": "small-scale architecture realization",
            "evidence": {
                "architecture": architecture,
                "cross_branch_coupling": coupling,
            },
        },
        {
            "claim": "The FANC encoder preserves low-frequency harmonics while pruning high-frequency redundancy using frequency-adaptive non-uniform compression (Section 3.3).",
            "challenge_claim_sha256": "666d20e379eaf3e6276e53c997da0973b34e9c7e7adda53320286899edce1bfe",
            "status": "partially_reproduced",
            "scope": "measured resolution and compute allocation per band",
            "evidence": fanc,
        },
        {
            "claim": "DVPD attains state-of-the-art speech enhancement quality on WSJ0-UNI while using 35% of PGUSE parameters and 40% of PGUSE inference MACs (Table 1).",
            "challenge_claim_sha256": "736a19fc99850eaea352bfbad35aa1a6f2eb01bfa6945e03e1dddb7c1b858d7d",
            "status": "unreplicated",
            "reason": "requires faithful full-scale DVPD and PGUSE implementations trained and evaluated on WSJ0-UNI; the small model here is not comparable, and no enhancement-quality metric on real speech exists in this package",
        },
        {
            "claim": "Training only on WSJ0-UNI, DVPD generalizes across multiple out-of-distribution speech enhancement benchmarks better than compared predictive and diffusion baselines (Figure 5).",
            "challenge_claim_sha256": "6488b252bfc515bc18780348702c196bb44a141fc7e992df1f5eed8852ce2665",
            "status": "unreplicated",
            "reason": "requires training on WSJ0-UNI and evaluating on the OOD benchmark suite; not feasible in this CPU-only package",
        },
        {
            "claim": "DVPD improves over compared methods on VBDMD speech denoising and VBDMD-SR speech super-resolution evaluations (Table 2; Table 3).",
            "challenge_claim_sha256": "5abd80ace16b92c56ecc11fbcb636fa728f1e220853c08e5282995effcca1428",
            "status": "unreplicated",
            "reason": "requires the VBDMD and VBDMD-SR corpora and trained baselines; not available to this package",
        },
        {
            "claim": "Ablations show that FANC, the frequency-aware interaction module, LISA, and the TLB strategy each contribute to DVPD performance (Table 4; Table 5).",
            "challenge_claim_sha256": "a9b3354cc3a98a479cb07b0f866b1ccfee08e54c6768b168d637ec85b521f50d",
            "status": "partially_reproduced",
            "scope": "toy-scale trained ablation of three of four components; TLB is not wired into this package's forward pass and could not be ablated",
            "evidence": ablation,
            "observation": ablation_observation(ablation),
        },
    ]

    evidence_summary = {
        "attempt_id": "4d8c65e3-fa13-448b-8111-d40c7d107ce5",
        "paper_id": "3qX5RS8kpJ",
        "slug": "dual-view-predictive-diffusion-lightweight-speech-enhancement-via-spectrogram-image-synergy",
        "upstream_revision": "arxiv:2602.00568",
        "claims": claims,
        "all_target_claims_verified": False,
        "provenance": "all numbers computed by generate_evidence.py with pinned seeds; fabricated MACs figure and prose-verified benchmark claims removed 2026-08-01",
    }

    project_dir = Path(__file__).parent
    evidence_dir = project_dir / "evidence"
    evidence_dir.mkdir(exist_ok=True)
    (evidence_dir / "evidence_summary.json").write_text(
        json.dumps(evidence_summary, indent=2) + "\n"
    )
    (project_dir / "pages").mkdir(exist_ok=True)
    write_report(project_dir / "pages" / "report.md", architecture, coupling, fanc, ablation)
    return evidence_summary


def write_report(path: Path, architecture: dict, coupling: dict, fanc: dict, ablation: dict) -> None:
    variants = ablation["heldout_denoised_mse_by_variant"]
    path.write_text(
        f"""# Reproduction Report: Dual-View Predictive Diffusion (DVPD)

**Paper ID:** `3qX5RS8kpJ`

## Scope

CPU-only, small-scale mechanism reproduction. The architecture, FANC band
allocation, cross-branch interaction, and a trained toy ablation carry real
measured numbers below. The paper's dataset-scale results - WSJ0-UNI state
of the art with 35%/40% efficiency ratios (Table 1), OOD generalization
(Figure 5), and the VBDMD tables (Tables 2-3) - are **unreplicated**. An
earlier revision of this package fabricated the MACs figure and verified
benchmark claims from a random forward pass; those values are removed.

## Dual-branch architecture (Figures 1-2): partially reproduced

A real forward pass preserves the input shape; the small model has
{architecture['toy_model_parameters']:,} parameters and a measured
{architecture['toy_model_conv_macs_per_forward']:,} conv multiply-accumulates
per forward (hook-counted, this implementation only - not comparable to the
paper's models). The dual-view coupling is real and measured: perturbing the
visual branch changes the acoustic output (response norm
{coupling['acoustic_response_to_visual_perturbation']}), and perturbing the
acoustic branch changes the visual output (response norm
{coupling['visual_response_to_acoustic_perturbation']}).

## FANC non-uniform compression (Section 3.3): partially reproduced

Measured band allocation on a 257-bin spectrogram: the low band keeps
{fanc['low_band_representation_rows']} representation rows for
{fanc['low_band_input_rows']} input rows (dense), while the high band
compresses {fanc['high_band_input_rows']} input rows into
{fanc['high_band_representation_rows']} rows (factor
{fanc['high_band_compression_factor']}). Measured compute allocation:
{fanc['macs_per_input_row_low_band']:,} MACs per low-band input row versus
{fanc['macs_per_input_row_high_band']:,} per high-band row - the non-uniform
budget the paper describes.

## Component ablation (Tables 4-5): partially reproduced at toy scale

Each variant trained 120 steps on synthetic harmonic-spectrogram denoising,
evaluated on held-out batches (noisy-input MSE
{ablation['heldout_noisy_input_mse']}):

| variant | held-out MSE |
| --- | --- |
| full model | {variants['full']} |
| no cross-branch interaction | {variants['no_interaction']} |
| no LISA | {variants['no_lisa']} |
| uniform encoder (FANC removed) | {variants['uniform_encoder']} |

Measured outcome, stated plainly: removing LISA or the cross-branch
interaction worsens held-out MSE, supporting their contribution - but
removing FANC *improves* it at this scale, so the FANC contribution claim is
**not supported** by this toy ablation. And `TLBStrategy` is defined in this
package but never connected to the forward pass, so the TLB ablation cannot
be trained here - the paper's four-component ablation is only three-quarters
realizable from this code.

## Unreplicated claims

Tables 1-3 and Figure 5 need full-scale trained models on real speech
corpora (WSJ0-UNI, VBDMD, VBDMD-SR) and faithful baselines. No
enhancement-quality numbers on real speech exist in this package, and none
are invented.

## Reproducibility

`uv run python generate_evidence.py` regenerates
`evidence/evidence_summary.json` and this page byte-identically (pinned
seeds, no timestamps).
"""
    )


if __name__ == "__main__":
    main()
    print("Evidence generated.")
